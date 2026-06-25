"""MODULE 1 — EyeDetector: EAR-based drowsiness and blink rate detection."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np


# MediaPipe Face Mesh eye landmark indices
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(p1 - p2))


class EyeDetector:
    """
    Detects eye closure and blink rate using the Eye Aspect Ratio (EAR).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        ear_threshold: EAR value below which eye is considered closed.
        consecutive_frames: Frames of closure before drowsiness is flagged.
    """

    def __init__(self, ear_threshold: float = 0.25, consecutive_frames: int = 20) -> None:
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames

        self._drowsy_frames: int = 0
        self._blink_timestamps: deque[float] = deque()
        self._eye_was_closed: bool = False
        self._total_blinks: int = 0

    # ------------------------------------------------------------------
    def calculate_ear(self, landmarks: list[Any], eye_indices: list[int]) -> float:
        """
        Compute Eye Aspect Ratio for the given landmark indices.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList (iterable of landmark objects).
            eye_indices: 6 indices [p1..p6] for EAR computation.

        Returns:
            EAR value as a float.
        """
        pts = np.array(
            [[landmarks[i].x, landmarks[i].y] for i in eye_indices],
            dtype=np.float32,
        )
        # Vertical distances
        v1 = _dist(pts[1], pts[5])
        v2 = _dist(pts[2], pts[4])
        # Horizontal distance
        h  = _dist(pts[0], pts[3])
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    # ------------------------------------------------------------------
    def detect(self, landmarks: list[Any]) -> dict[str, Any]:
        """
        Run full eye analysis for one frame.

        Args:
            landmarks: MediaPipe face mesh landmark list.

        Returns:
            dict with keys: ear_left, ear_right, ear_avg, is_drowsy,
                            drowsy_frames, blink_count, confidence.
        """
        ear_left  = self.calculate_ear(landmarks, LEFT_EYE)
        ear_right = self.calculate_ear(landmarks, RIGHT_EYE)
        ear_avg   = (ear_left + ear_right) / 2.0

        eye_closed = ear_avg < self.ear_threshold

        # Blink detection: transition from closed → open
        if self._eye_was_closed and not eye_closed:
            self._total_blinks += 1
            self._blink_timestamps.append(time.time())

        self._eye_was_closed = eye_closed

        if eye_closed:
            self._drowsy_frames += 1
        else:
            self._drowsy_frames = 0

        is_drowsy  = self._drowsy_frames >= self.consecutive_frames
        confidence = min(1.0, self._drowsy_frames / max(1, self.consecutive_frames))

        return {
            "ear_left":     round(ear_left, 4),
            "ear_right":    round(ear_right, 4),
            "ear_avg":      round(ear_avg, 4),
            "is_drowsy":    is_drowsy,
            "drowsy_frames": self._drowsy_frames,
            "blink_count":  self._total_blinks,
            "confidence":   round(confidence, 3),
        }

    # ------------------------------------------------------------------
    def get_blink_rate(self, window_seconds: float = 60.0) -> float:
        """
        Calculate blinks per minute over the last `window_seconds`.

        Args:
            window_seconds: Rolling window size in seconds.

        Returns:
            Blinks-per-minute (float).
        """
        now = time.time()
        cutoff = now - window_seconds
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()
        rate = len(self._blink_timestamps) / (window_seconds / 60.0)
        return round(rate, 2)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all counters and history."""
        self._drowsy_frames = 0
        self._blink_timestamps.clear()
        self._eye_was_closed = False
        self._total_blinks = 0
