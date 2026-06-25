"""MODULE 5 — SessionTracker: SQLite session persistence, fatigue scoring, and report generation."""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DB_PATH     = os.path.join("data", "sessions.db")
REPORTS_DIR = "reports"


class SessionTracker:
    """
    Tracks driving sessions, calculates fatigue score, and generates matplotlib reports.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path    = db_path
        self._session_id: str | None  = None
        self._driver_id: str          = "default"
        self._start_time: float       = 0.0
        self._frame_metrics: list[dict[str, Any]] = []
        self._alert_count: int        = 0

        os.makedirs("data",    exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    driver_id    TEXT,
                    start_ts     REAL,
                    end_ts       REAL,
                    duration     REAL,
                    avg_fatigue  REAL,
                    alert_count  INTEGER,
                    drowsy_eps   INTEGER
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS frames (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT,
                    ts           REAL,
                    ear          REAL,
                    mar          REAL,
                    pitch        REAL,
                    yaw          REAL,
                    blink_rate   REAL,
                    fatigue      REAL,
                    alert_level  INTEGER
                )
            """)

    def start_session(self, driver_id: str = "default") -> str:
        self._session_id  = str(uuid.uuid4())[:8]
        self._driver_id   = driver_id
        self._start_time  = time.time()
        self._frame_metrics.clear()
        self._alert_count = 0
        return self._session_id

    def calculate_fatigue_score(self, metrics: dict[str, Any]) -> float:
        """Weighted fatigue score 0–100. Weights: EAR(40%) MAR(20%) HeadPose(25%) BlinkRate(15%)"""
        ear_thr   = float(os.getenv("EAR_THRESHOLD",       "0.25"))
        mar_thr   = float(os.getenv("MAR_THRESHOLD",       "0.7"))
        head_thr  = float(os.getenv("HEAD_TILT_THRESHOLD", "20"))

        ear_val   = metrics.get("ear_avg",    0.3)
        mar_val   = metrics.get("mar",        0.3)
        pitch_val = abs(metrics.get("pitch",  0.0))
        blink_val = metrics.get("blink_rate", 15.0)

        ear_comp   = max(0.0, (ear_thr - ear_val) / ear_thr) * 100
        mar_comp   = max(0.0, (mar_val - mar_thr * 0.5) / mar_thr) * 100
        head_comp  = min(100.0, pitch_val / head_thr * 100)
        blink_comp = max(0.0, (15.0 - blink_val) / 15.0) * 100

        score = (ear_comp * 0.40 + mar_comp * 0.20 +
                 head_comp * 0.25 + blink_comp * 0.15)
        return round(min(100.0, max(0.0, score)), 2)

    def update(self, metrics: dict[str, Any]) -> None:
        """Store one frame's metrics to memory (flushed at end_session)."""
        if not self._session_id:
            return

        fatigue = self.calculate_fatigue_score(metrics)
        metrics["fatigue_score"] = fatigue

        if metrics.get("alert_level", 0) > 0:
            self._alert_count += 1

        self._frame_metrics.append({
            "ts":          time.time(),
            "ear":         metrics.get("ear_avg",    0.0),
            "mar":         metrics.get("mar",         0.0),
            "pitch":       metrics.get("pitch",       0.0),
            "yaw":         metrics.get("yaw",         0.0),
            "blink_rate":  metrics.get("blink_rate",  0.0),
            "fatigue":     fatigue,
            "alert_level": metrics.get("alert_level", 0),
        })

    def end_session(self) -> dict[str, Any]:
        """Finalise the session: persist to DB and return summary dict."""
        if not self._session_id or not self._frame_metrics:
            return {}

        end_ts    = time.time()
        duration  = end_ts - self._start_time
        fatigue_vals = [m["fatigue"] for m in self._frame_metrics]
        avg_fatigue  = round(sum(fatigue_vals) / len(fatigue_vals), 2)
        drowsy_eps   = sum(1 for m in self._frame_metrics if m["alert_level"] >= 2)

        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?)",
                (self._session_id, self._driver_id, self._start_time,
                 end_ts, duration, avg_fatigue, self._alert_count, drowsy_eps),
            )
            con.executemany(
                "INSERT INTO frames (session_id,ts,ear,mar,pitch,yaw,blink_rate,fatigue,alert_level)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (self._session_id, m["ts"], m["ear"], m["mar"], m["pitch"],
                     m["yaw"], m["blink_rate"], m["fatigue"], m["alert_level"])
                    for m in self._frame_metrics
                ],
            )

        return {
            "session_id":      self._session_id,
            "driver_id":       self._driver_id,
            "duration":        round(duration, 1),
            "avg_fatigue":     avg_fatigue,
            "alert_count":     self._alert_count,
            "drowsy_episodes": drowsy_eps,
        }

    def generate_report(self, session_id: str | None = None) -> str:
        """Generate a matplotlib PNG report for the session. Returns path to saved file."""
        sid = session_id or self._session_id
        if not sid or not self._frame_metrics:
            return ""

        times   = [m["ts"] - self._frame_metrics[0]["ts"] for m in self._frame_metrics]
        fatigue = [m["fatigue"]    for m in self._frame_metrics]
        blink   = [m["blink_rate"] for m in self._frame_metrics]
        alerts  = [m["alert_level"] for m in self._frame_metrics]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.patch.set_facecolor("#0d0d0d")

        for ax in axes:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")

        axes[0].plot(times, fatigue, color="#00ff88", lw=1.5)
        axes[0].axhline(55, color="orange", lw=1, linestyle="--", label="Danger")
        axes[0].axhline(75, color="red",    lw=1, linestyle="--", label="Critical")
        axes[0].set_title("Fatigue Score Over Time")
        axes[0].set_ylabel("Fatigue %")
        axes[0].legend(facecolor="#1a1a2e", labelcolor="white")

        axes[1].plot(times, blink, color="#00aaff", lw=1.5)
        axes[1].axhline(15, color="yellow", lw=1, linestyle="--", label="Normal (15)")
        axes[1].set_title("Blink Rate (blinks/min)")
        axes[1].set_ylabel("Blinks/min")
        axes[1].legend(facecolor="#1a1a2e", labelcolor="white")

        alert_counts = [0, 0, 0, 0]
        for a in alerts:
            alert_counts[a] += 1
        pie_labels = ["Normal", "Warning", "Danger", "Critical"]
        pie_colors = ["#00c864", "#ffd700", "#ff8c00", "#ff2020"]
        non_zero = [(l, c, v) for l, c, v in zip(pie_labels, pie_colors, alert_counts) if v > 0]
        if non_zero:
            lbls, clrs, vals = zip(*non_zero)
            axes[2].pie(vals, labels=lbls, colors=clrs, autopct="%1.0f%%",
                        textprops={"color": "white"})
        axes[2].set_title("Alert Distribution")

        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, f"report_{sid}.png")
        plt.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
