"""Alert manager for web deployment — visual alerts only (no server audio)."""

from __future__ import annotations
import time
from typing import Any


class AlertManager:
    """Manages alert state and overlay drawing. No audio (browser handles it)."""

    LEVEL_LABELS = {0: "NORMAL", 1: "WARNING", 2: "DANGER", 3: "CRITICAL"}
    LEVEL_COLORS = {0: (0, 255, 120), 1: (0, 200, 255), 2: (0, 130, 255), 3: (0, 60, 255)}

    def __init__(self, cooldown: float = 5.0) -> None:
        self._cooldown    = cooldown
        self._last_alert  = 0.0
        self._last_level  = 0

    def trigger(self, level: int, metrics: dict[str, Any]) -> bool:
        """
        Decide if an alert should fire. Returns True if triggered.
        level: 0=normal, 1=warning, 2=danger, 3=critical
        """
        if level == 0:
            self._last_level = 0
            return False

        now = time.time()
        if now - self._last_alert < self._cooldown and level <= self._last_level:
            return False

        self._last_alert = now
        self._last_level  = level
        return True

    def draw_overlay(self, frame, level: int, metrics: dict[str, Any]):
        """Draw HUD overlay on frame. Returns modified frame."""
        import cv2
        h, w = frame.shape[:2]

        if level > 0:
            color = self.LEVEL_COLORS[level]
            label = self.LEVEL_LABELS[level]
            cv2.rectangle(frame, (0, 0), (w, h), color, 3)
            cv2.putText(frame, label,
                        (w // 2 - 60, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        # HUD info
        fat  = metrics.get("fatigue_score", 0)
        ear  = metrics.get("ear_avg", 0)
        info = f"FAT:{fat:.0f}%  EAR:{ear:.3f}"
        cv2.putText(frame, info, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1)

        return frame
