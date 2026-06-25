"""EXPERIMENTAL X2 — GazeTracker: iris-based gaze deviation and distraction detection."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np


# MediaPipe iris landmark indices (only available with refine_landmarks=True)
LEFT_IRIS  = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]

# Left/right eye outer corners for normalization
LEFT_EYE_CORNERS  = (362, 263)   # left corner, right corner of left eye
RIGHT_EYE_CORNERS = (33, 133)    # left corner, right corner of right eye


class GazeTracker:
    """
    Tracks gaze direction using MediaPipe iris landmarks.
    Detects distraction when gaze deviates from center for > threshold_seconds.

    Requires mediapipe FaceMesh initialized with refine_landmarks=True.
    """

    HISTORY_LEN = 90  # ~3 seconds at 30 fps

    def __init__(self) -> None:
        self._gaze_x_history: deque[float] = deque(maxlen=self.HISTORY_LEN)
        self._gaze_y_history: deque[float] = deque(maxlen=self.HISTORY_LEN)
        self._away_start: float | None     = None

    # ------------------------------------------------------------------
    def _pupil_center(self, landmarks: list[Any], indices: list[int]) -> np.ndarray:
        """Return mean x,y of the given landmark indices."""
        pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices], dtype=np.float32)
        return pts.mean(axis=0)

    # ------------------------------------------------------------------
    def estimate_gaze(self, landmarks: list[Any]) -> dict[str, Any]:
        """
        Estimate gaze direction from iris position relative to eye corners.

        Args:
            landmarks: MediaPipe face mesh normalized landmark list
                       (must have refine_landmarks=True for iris).

        Returns:
            dict with gaze_x, gaze_y, is_looking_away, deviation_angle.
        """
        # Iris centers
        left_iris  = self._pupil_center(landmarks, LEFT_IRIS)
        right_iris = self._pupil_center(landmarks, RIGHT_IRIS)

        # Eye corners for normalization
        l_left  = np.array([landmarks[LEFT_EYE_CORNERS[0]].x,  landmarks[LEFT_EYE_CORNERS[0]].y])
        l_right = np.array([landmarks[LEFT_EYE_CORNERS[1]].x,  landmarks[LEFT_EYE_CORNERS[1]].y])
        r_left  = np.array([landmarks[RIGHT_EYE_CORNERS[0]].x, landmarks[RIGHT_EYE_CORNERS[0]].y])
        r_right = np.array([landmarks[RIGHT_EYE_CORNERS[1]].x, landmarks[RIGHT_EYE_CORNERS[1]].y])

        def normalise(iris: np.ndarray, corner_l: np.ndarray, corner_r: np.ndarray) -> tuple[float, float]:
            w = float(np.linalg.norm(corner_r - corner_l)) + 1e-6
            rel = iris - corner_l
            nx = float(rel[0]) / w - 0.5   # -0.5 left … +0.5 right
            ny = float(rel[1]) / (w * 0.4) - 0.5
            return nx, ny

        lx, ly = normalise(left_iris,  l_left, l_right)
        rx, ry = normalise(right_iris, r_left, r_right)

        gaze_x = round((lx + rx) / 2.0, 4)
        gaze_y = round((ly + ry) / 2.0, 4)

        self._gaze_x_history.append(gaze_x)
        self._gaze_y_history.append(gaze_y)

        deviation = float(np.sqrt(gaze_x ** 2 + gaze_y ** 2))
        angle_deg = round(deviation * 45.0, 2)   # approx mapping
        is_away   = deviation > 0.25

        return {
            "gaze_x":         gaze_x,
            "gaze_y":         gaze_y,
            "is_looking_away": is_away,
            "deviation_angle": angle_deg,
        }

    # ------------------------------------------------------------------
    def detect_distraction(self, threshold_seconds: float = 2.0) -> bool:
        """
        Return True if the driver has been looking away for > threshold_seconds.

        Args:
            threshold_seconds: Duration of continuous gaze deviation to flag.

        Returns:
            True if distracted.
        """
        if not self._gaze_x_history:
            return False

        gx = np.array(self._gaze_x_history)
        gy = np.array(self._gaze_y_history)
        away_flags = (np.sqrt(gx ** 2 + gy ** 2) > 0.25)

        # Check trailing run of away frames (30 fps assumed)
        fps = 30
        needed = int(threshold_seconds * fps)
        if len(away_flags) < needed:
            return False

        return bool(away_flags[-needed:].all())
