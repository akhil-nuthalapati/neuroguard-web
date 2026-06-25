"""
NeuroGuard Web — Pure OpenCV fallback (no MediaPipe, no GPU libs needed).
Uses Haar cascades + dlib-style landmark approximation via OpenCV DNN.
Drop-in replacement if MediaPipe fails in headless Docker.
"""

from __future__ import annotations

import base64
import math
import os
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from flask_socketio import SocketIO

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "neuroguard-web-secret")
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

# ── Config ────────────────────────────────────────────────────────────────────
EAR_THR    = float(os.getenv("EAR_THRESHOLD",      "0.25"))
MAR_THR    = float(os.getenv("MAR_THRESHOLD",       "0.60"))
HEAD_THR   = float(os.getenv("HEAD_TILT_THRESHOLD","20"))
BLINK_WIN  = float(os.getenv("BLINK_RATE_WINDOW",  "60"))

# ── OpenCV face/eye/smile cascades (no GL needed) ─────────────────────────────
_CASCADE_BASE = cv2.data.haarcascades  # bundled with opencv-python-headless
_face_cas  = cv2.CascadeClassifier(_CASCADE_BASE + "haarcascade_frontalface_default.xml")
_eye_cas   = cv2.CascadeClassifier(_CASCADE_BASE + "haarcascade_eye_tree_eyeglasses.xml")
_smile_cas = cv2.CascadeClassifier(_CASCADE_BASE + "haarcascade_smile.xml")

# ── Per-session state ─────────────────────────────────────────────────────────
_sessions: dict[str, dict[str, Any]] = {}
_sessions_lock = threading.Lock()


def _new_session() -> dict[str, Any]:
    return {
        "start":        time.time(),
        "blink_times":  deque(),        # timestamps of blinks
        "blink_count":  0,
        "yawn_count":   0,
        "drowsy_frames":0,
        "fatigue_hist": deque(maxlen=120),
        "prev_ear":     1.0,
        "eye_closed_frames": 0,
        "prev_yawning": False,
    }


def _get_session(sid: str) -> dict[str, Any]:
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = _new_session()
        return _sessions[sid]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _dist(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _estimate_ear_from_eye_rect(ex, ey, ew, eh) -> float:
    """Approximate EAR from eye bounding rect: height/width ratio."""
    if ew == 0:
        return 0.30
    return (eh / ew) * 1.5          # calibrated to match landmark EAR range


def _estimate_mar_from_smile(smile_rects, face_w, face_h) -> float:
    """Approximate MAR from smile region height."""
    if not len(smile_rects):
        return 0.20
    sx, sy, sw, sh = smile_rects[0]
    return min(sh / max(face_h * 0.15, 1), 1.5)


def _face_to_pitch_yaw(fx, fy, fw, fh, frame_w, frame_h) -> tuple[float, float]:
    """Rough head pose from face position (center offset from frame center)."""
    cx = fx + fw / 2 - frame_w / 2
    cy = fy + fh / 2 - frame_h / 2
    yaw   =  cx / frame_w * 45.0   # ±22.5° range
    pitch = -cy / frame_h * 30.0
    return pitch, yaw


def _fatigue_to_level(score: float) -> int:
    if score < 30: return 0
    if score < 55: return 1
    if score < 75: return 2
    return 3


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    data      = request.get_json(force=True, silent=True) or {}
    frame_b64 = data.get("frame", "")
    sid       = data.get("session_id", "default")

    # Decode frame
    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid frame"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    state = _get_session(sid)
    now   = time.time()
    h, w  = frame.shape[:2]

    metrics: dict[str, Any] = {
        "ear_avg": 0.30, "mar": 0.20, "pitch": 0.0, "yaw": 0.0,
        "blink_rate": 15.0, "blink_count": state["blink_count"],
        "yawn_count": state["yawn_count"], "fatigue_score": 0.0,
        "alert_level": 0, "face_detected": False,
        "is_drowsy": False, "is_yawning": False, "is_nodding": False,
        "attention": 0.8, "gaze_x": 0.0, "gaze_y": 0.0,
        "is_distracted": False,
        "minutes_to_fatigue": -1, "confidence": 0.0,
        "recommendation": "Drive safely.",
        "session_time": now - state["start"],
        "session_id": sid,
    }

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cas.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces):
        fx, fy, fw, fh = faces[0]
        face_gray = gray[fy:fy+fh, fx:fx+fw]
        metrics["face_detected"] = True

        # ── Eyes ──────────────────────────────────────────────────────────
        eyes = _eye_cas.detectMultiScale(face_gray, 1.1, 5, minSize=(15, 15))
        if len(eyes) >= 2:
            ears = [_estimate_ear_from_eye_rect(*e) for e in eyes[:2]]
            ear  = sum(ears) / len(ears)
        elif len(eyes) == 1:
            ear = _estimate_ear_from_eye_rect(*eyes[0])
        else:
            ear = 0.15          # eyes not detected → likely closed

        metrics["ear_avg"] = round(ear, 4)

        # Blink detection: EAR crosses below threshold then back up
        if state["prev_ear"] >= EAR_THR and ear < EAR_THR:
            state["eye_closed_frames"] = 1
        elif state["prev_ear"] < EAR_THR and ear >= EAR_THR:
            if state["eye_closed_frames"] > 0:
                state["blink_count"] += 1
                state["blink_times"].append(now)
                state["eye_closed_frames"] = 0
        elif ear < EAR_THR:
            state["eye_closed_frames"] += 1
        state["prev_ear"] = ear

        is_drowsy = ear < EAR_THR and state["eye_closed_frames"] > 15
        metrics["is_drowsy"] = is_drowsy

        # ── Blink rate (last 60s) ─────────────────────────────────────────
        cutoff = now - BLINK_WIN
        while state["blink_times"] and state["blink_times"][0] < cutoff:
            state["blink_times"].popleft()
        blink_rate = len(state["blink_times"]) / (BLINK_WIN / 60)
        metrics["blink_rate"]  = round(blink_rate, 1)
        metrics["blink_count"] = state["blink_count"]

        # ── Yawn / Mouth (smile cascade in lower face) ────────────────────
        lower_face = face_gray[fh//2:, :]
        smiles     = _smile_cas.detectMultiScale(lower_face, 1.7, 22)
        mar        = _estimate_mar_from_smile(smiles, fw, fh)
        metrics["mar"]      = round(mar, 4)
        is_yawning = mar > MAR_THR
        metrics["is_yawning"] = is_yawning
        if is_yawning and not state["prev_yawning"]:
            state["yawn_count"] += 1
        state["prev_yawning"] = is_yawning
        metrics["yawn_count"] = state["yawn_count"]

        # ── Head pose ─────────────────────────────────────────────────────
        pitch, yaw = _face_to_pitch_yaw(fx, fy, fw, fh, w, h)
        metrics["pitch"] = round(pitch, 2)
        metrics["yaw"]   = round(yaw,   2)
        metrics["is_nodding"] = abs(pitch) > float(HEAD_THR)

        # ── Gaze (approximate from face position) ─────────────────────────
        gx = (fx + fw/2 - w/2) / (w/2)
        gy = (fy + fh/2 - h/2) / (h/2)
        metrics["gaze_x"] = round(float(gx), 3)
        metrics["gaze_y"] = round(float(gy), 3)
        metrics["is_distracted"] = abs(gx) > 0.25 or abs(gy) > 0.25

        # ── Fatigue score ─────────────────────────────────────────────────
        fat = 0.0
        fat += max(0, (EAR_THR - ear) / EAR_THR) * 40     # low EAR → higher fatigue
        fat += min(state["yawn_count"] * 8, 25)             # yawns
        if blink_rate < 8:  fat += 15                       # too few blinks
        if blink_rate > 25: fat += 10                       # too many blinks (tired eyes)
        fat += min(state["eye_closed_frames"] / 30 * 20, 20)
        fat = min(fat, 100)

        state["fatigue_hist"].append(fat)
        metrics["fatigue_score"] = round(fat, 1)
        metrics["alert_level"]   = _fatigue_to_level(fat)

        # ── Attention ─────────────────────────────────────────────────────
        attn = 1.0
        if is_drowsy:  attn -= 0.4
        if is_yawning: attn -= 0.2
        if metrics["is_distracted"]: attn -= 0.2
        attn = max(0.0, round(attn, 2))
        metrics["attention"] = attn

        # ── Fatigue prediction ────────────────────────────────────────────
        if len(state["fatigue_hist"]) > 10:
            trend  = list(state["fatigue_hist"])
            slope  = (trend[-1] - trend[0]) / len(trend)
            if slope > 0.1 and fat < 80:
                remaining = (80 - fat) / (slope * 60)
                metrics["minutes_to_fatigue"] = max(0, round(remaining, 1))
                metrics["confidence"] = min(0.85, len(trend) / 120)
                if remaining < 5:
                    metrics["recommendation"] = "⚠️ Pull over and rest immediately!"
                elif remaining < 15:
                    metrics["recommendation"] = "Take a break within 15 minutes."
                else:
                    metrics["recommendation"] = "Stay alert. Monitor your fatigue."
            else:
                metrics["minutes_to_fatigue"] = -1
                metrics["recommendation"] = "Fatigue level stable. Drive safely."

    socketio.emit("metrics", metrics)
    return jsonify(metrics)


@app.route("/api/report/<session_id>")
def api_report(session_id: str):
    path = os.path.join("reports", f"report_{session_id}.png")
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return jsonify({"error": "Report not found"}), 404


@app.route("/api/end_session", methods=["POST"])
def end_session():
    data = request.get_json(force=True, silent=True) or {}
    sid  = data.get("session_id", "default")
    with _sessions_lock:
        _sessions.pop(sid, None)
    return jsonify({"status": "ended"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "sessions": len(_sessions), "engine": "opencv-haar"})


@socketio.on("connect")
def on_connect():
    print(f"[WS] connected: {request.sid}")

@socketio.on("disconnect")
def on_disconnect():
    print(f"[WS] disconnected: {request.sid}")


if __name__ == "__main__":
    import eventlet
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5050")))
    print(f"NeuroGuard Web (OpenCV) -> http://0.0.0.0:{port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
