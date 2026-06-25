"""MODULE 2 — YawnDetector: MAR-based yawn detection."""

from __future__ import annotations

from typing import Any

import numpy as np


# Key MediaPipe landmark indices used for MAR
# Upper lip: 13, Lower lip: 14, Far lower: 17
# Left corner: 61, Right corner: 291
MOUTH_UPPER = [13, 14]
MOUTH_LOWER = [17, 312]
MOUTH_LEFT  = 61
MOUTH_RIGHT = 291

# Full set of outer lip landmarks (for drawing)
OUTER_MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


class YawnDetector:
    """
    Detects yawning using Mouth Aspect Ratio (MAR).

    MAR = vertical_open / horizontal_width
        = (||upper_lip - lower_lip||) / (||left_corner - right_corner||)

    Args:
        mar_threshold: MAR above which mouth is considered open (yawning).
        yawn_frames: Consecutive frames of open mouth before counting as yawn.
    """

    def __init__(self, mar_threshold: float = 0.7, yawn_frames: int = 15) -> None:
        self.mar_threshold = mar_threshold
        self.yawn_frames   = yawn_frames

        self._yawn_frames_counter: int = 0
        self._yawn_count: int = 0
        self._was_yawning: bool = False

    # ------------------------------------------------------------------
    def calculate_mar(self, landmarks: list[Any]) -> float:
        """
        Compute Mouth Aspect Ratio.

        Args:
            landmarks: MediaPipe face mesh normalized landmark list.

        Returns:
            MAR value as float.
        """
        upper = np.array([landmarks[13].x, landmarks[13].y], dtype=np.float32)
        lower = np.array([landmarks[17].x, landmarks[17].y], dtype=np.float32)
        left  = np.array([landmarks[MOUTH_LEFT].x,  landmarks[MOUTH_LEFT].y],  dtype=np.float32)
        right = np.array([landmarks[MOUTH_RIGHT].x, landmarks[MOUTH_RIGHT].y], dtype=np.float32)

        vertical   = _dist(upper, lower)
        horizontal = _dist(left,  right)

        if horizontal < 1e-6:
            return 0.0
        return round(vertical / horizontal, 4)

    # ------------------------------------------------------------------
    def detect(self, landmarks: list[Any]) -> dict[str, Any]:
        """
        Run yawn analysis for one frame.

        Args:
            landmarks: MediaPipe face mesh landmark list.

        Returns:
            dict with keys: mar, is_yawning, yawn_frames, yawn_count, confidence.
        """
        mar       = self.calculate_mar(landmarks)
        is_open   = mar > self.mar_threshold

        if is_open:
            self._yawn_frames_counter += 1
        else:
            # Yawn ended — count it if it was long enough
            if self._was_yawning:
                self._yawn_count += 1
            self._yawn_frames_counter = 0

        is_yawning = self._yawn_frames_counter >= self.yawn_frames
        self._was_yawning = is_yawning

        confidence = min(1.0, self._yawn_frames_counter / max(1, self.yawn_frames))

        return {
            "mar":        mar,
            "is_yawning": is_yawning,
            "yawn_frames": self._yawn_frames_counter,
            "yawn_count": self._yawn_count,
            "confidence": round(confidence, 3),
        }

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset yawn state."""
        self._yawn_frames_counter = 0
        self._yawn_count = 0
        self._was_yawning = False
