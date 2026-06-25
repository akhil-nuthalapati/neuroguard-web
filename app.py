"""NeuroGuard Web — Flask server with WebRTC frame processing endpoint."""

from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_socketio import SocketIO

from core.eye_detector         import EyeDetector
from core.yawn_detector        import YawnDetector
from core.head_pose_estimator  import HeadPoseEstimator
from core.session_tracker      import SessionTracker
from experimental.attention_scorer import AttentionScorer
from experimental.gaze_tracker     import GazeTracker

load_dotenv()

# ── Flask + SocketIO ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "neuroguard-web-secret")
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker     = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
RunningMode        = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join("models", "face_landmarker.task")

# ── Config from env ───────────────────────────────────────────────────────────
EAR_THR    = float(os.getenv("EAR_THRESHOLD",       "0.25"))
MAR_THR    = float(os.getenv("MAR_THRESHOLD",        "0.7"))
CON_FRAMES = int  (os.getenv("CONSECUTIVE_FRAMES",  "20"))
YAWN_FR    = int  (os.getenv("YAWN_FRAMES",         "15"))
HEAD_THR   = float(os.getenv("HEAD_TILT_THRESHOLD", "20"))
BLINK_WIN  = float(os.getenv("BLINK_RATE_WINDOW",   "60"))

# ── Per-session state (keyed by session_id) ───────────────────────────────────
_sessions: dict[str, dict[str, Any]] = {}
_sessions_lock = threading.Lock()

# ── Landmarker (single shared instance, thread-safe for IMAGE mode) ───────────
_landmarker: FaceLandmarker | None = None
_lm_lock = threading.Lock()


def _get_landmarker() -> FaceLandmarker:
    global _landmarker
    if _landmarker is None:
        opts = FaceLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarker = FaceLandmarker.create_from_options(opts)
    return _landmarker


def _fatigue_to_level(score: float) -> int:
    if score < 30:  return 0
    if score < 55:  return 1
    if score < 75:  return 2
    return 3


def _get_or_create_session(sid: str, driver_id: str = "web_user") -> dict[str, Any]:
    """Get or initialise detector objects for a browser session."""
    with _sessions_lock:
        if sid not in _sessions:
            sess = SessionTracker()
            sess.start_session(driver_id)
            _sessions[sid] = {
                "eye":    EyeDetector(EAR_THR,  CON_FRAMES),
                "yawn":   YawnDetector(MAR_THR, YAWN_FR),
                "head":   HeadPoseEstimator(HEAD_THR),
                "sess":   sess,
                "attn":   AttentionScorer(),
                "gaze":   GazeTracker(),
                "start":  time.time(),
            }
        return _sessions[sid]


def _cleanup_session(sid: str) -> None:
    with _sessions_lock:
        if sid in _sessions:
            try:
                state = _sessions.pop(sid)
                state["sess"].end_session()
            except Exception:
                pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    """
    Receive a base64-encoded JPEG frame from the browser,
    run all detectors, return metrics as JSON.
    """
    data = request.get_json(force=True, silent=True) or {}
    frame_b64 = data.get("frame", "")
    sid        = data.get("session_id", "default")
    driver_id  = data.get("driver_id",  "web_user")

    # Decode base64 frame
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

    state = _get_or_create_session(sid, driver_id)
    eye, yawn, head = state["eye"], state["yawn"], state["head"]
    sess, attn, gaze = state["sess"], state["attn"], state["gaze"]

    metrics: dict[str, Any] = {
        "ear_avg": 0.30, "mar": 0.30, "pitch": 0.0, "yaw": 0.0,
        "blink_rate": 15.0, "blink_count": 0, "yawn_count": 0,
        "fatigue_score": 0.0, "alert_level": 0,
        "session_time": time.time() - state["start"],
        "session_id": sess._session_id,
        "face_detected": False,
    }

    try:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb      = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with _lm_lock:
            result = _get_landmarker().detect(mp_image)

        if result.face_landmarks:
            lm         = result.face_landmarks[0]
            eye_res    = eye.detect(lm)
            yawn_res   = yawn.detect(lm)
            head_res   = head.estimate(frame, lm)
            blink_rate = eye.get_blink_rate(BLINK_WIN)

            attention  = attn.score(eye_res, head_res, {"blink_rate": blink_rate})
            gaze_res   = gaze.estimate_gaze(lm)
            distracted = gaze.detect_distraction()
            prediction = attn.predict_fatigue_onset()

            fatigue = sess.calculate_fatigue_score({
                "ear_avg": eye_res["ear_avg"], "mar": yawn_res["mar"],
                "pitch": head_res["pitch"],    "blink_rate": blink_rate,
            })
            level = _fatigue_to_level(fatigue)
            if head.detect_microsleep():
                level = max(level, 2)

            metrics.update({
                "ear_avg":            eye_res["ear_avg"],
                "mar":                yawn_res["mar"],
                "pitch":              head_res["pitch"],
                "yaw":                head_res["yaw"],
                "blink_rate":         blink_rate,
                "blink_count":        eye_res["blink_count"],
                "yawn_count":         yawn_res["yawn_count"],
                "is_drowsy":          eye_res["is_drowsy"],
                "is_yawning":         yawn_res["is_yawning"],
                "is_nodding":         head_res["is_nodding"],
                "attention":          attention,
                "gaze_x":             gaze_res["gaze_x"],
                "gaze_y":             gaze_res["gaze_y"],
                "is_distracted":      distracted,
                "minutes_to_fatigue": prediction["minutes_to_fatigue"],
                "confidence":         prediction["confidence"],
                "recommendation":     prediction["recommendation"],
                "fatigue_score":      fatigue,
                "alert_level":        level,
                "face_detected":      True,
            })

            sess.update(metrics)

    except Exception as e:
        metrics["error"] = str(e)

    # Broadcast to all connected dashboard tabs
    socketio.emit("metrics", metrics)
    return jsonify(metrics)


@app.route("/api/report/<session_id>")
def api_report(session_id: str):
    """Download PNG report for a session."""
    path = os.path.join("reports", f"report_{session_id}.png")
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return jsonify({"error": "Report not found"}), 404


@app.route("/api/end_session", methods=["POST"])
def end_session():
    """Finalise a session and generate report."""
    data = request.get_json(force=True, silent=True) or {}
    sid  = data.get("session_id", "default")
    with _sessions_lock:
        state = _sessions.get(sid)
    if not state:
        return jsonify({"error": "session not found"}), 404
    summary = state["sess"].end_session()
    report  = state["sess"].generate_report()
    _cleanup_session(sid)
    return jsonify({"summary": summary, "report": report})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "sessions": len(_sessions)})


# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on("process_frame")
def on_process_frame(data):
    """Handle frames via WebSocket for lower latency / higher FPS."""
    frame_b64 = data.get("frame", "")
    sid        = data.get("session_id", "default")
    driver_id  = data.get("driver_id",  "web_user")

    metrics = {"error": "unknown error"}
    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "invalid frame"}

        state = _get_or_create_session(sid, driver_id)
        eye, yawn, head = state["eye"], state["yawn"], state["head"]
        sess, attn, gaze = state["sess"], state["attn"], state["gaze"]

        metrics = {
            "ear_avg": 0.30, "mar": 0.30, "pitch": 0.0, "yaw": 0.0,
            "blink_rate": 15.0, "blink_count": 0, "yawn_count": 0,
            "fatigue_score": 0.0, "alert_level": 0,
            "session_time": time.time() - state["start"],
            "session_id": sess._session_id,
            "face_detected": False,
        }

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb      = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with _lm_lock:
            result = _get_landmarker().detect(mp_image)

        if result.face_landmarks:
            lm         = result.face_landmarks[0]
            eye_res    = eye.detect(lm)
            yawn_res   = yawn.detect(lm)
            head_res   = head.estimate(frame, lm)
            blink_rate = eye.get_blink_rate(BLINK_WIN)

            attention  = attn.score(eye_res, head_res, {"blink_rate": blink_rate})
            gaze_res   = gaze.estimate_gaze(lm)
            distracted = gaze.detect_distraction()
            prediction = attn.predict_fatigue_onset()

            fatigue = sess.calculate_fatigue_score({
                "ear_avg": eye_res["ear_avg"], "mar": yawn_res["mar"],
                "pitch": head_res["pitch"],    "blink_rate": blink_rate,
            })
            level = _fatigue_to_level(fatigue)
            if head.detect_microsleep():
                level = max(level, 2)

            metrics.update({
                "ear_avg":            eye_res["ear_avg"],
                "mar":                yawn_res["mar"],
                "pitch":              head_res["pitch"],
                "yaw":                head_res["yaw"],
                "blink_rate":         blink_rate,
                "blink_count":        eye_res["blink_count"],
                "yawn_count":         yawn_res["yawn_count"],
                "is_drowsy":          eye_res["is_drowsy"],
                "is_yawning":         yawn_res["is_yawning"],
                "is_nodding":         head_res["is_nodding"],
                "attention":          attention,
                "gaze_x":             gaze_res["gaze_x"],
                "gaze_y":             gaze_res["gaze_y"],
                "is_distracted":      distracted,
                "minutes_to_fatigue": prediction["minutes_to_fatigue"],
                "confidence":         prediction["confidence"],
                "recommendation":     prediction["recommendation"],
                "fatigue_score":      fatigue,
                "alert_level":        level,
                "face_detected":      True,
            })
            sess.update(metrics)
    except Exception as e:
        metrics["error"] = str(e)

    # Return directly to the callback
    return metrics


@socketio.on("connect")
def on_connect():
    print(f"Client connected: {request.sid}")

@socketio.on("disconnect")
def on_disconnect():
    print(f"Client disconnected: {request.sid}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import eventlet
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5050")))
    print(f"NeuroGuard Web -> http://localhost:{port}")
    socketio.run(app, host="0.0.0.0", port=port,
                 debug=False, allow_unsafe_werkzeug=True)
