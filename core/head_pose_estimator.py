"""MODULE 3 — HeadPoseEstimator: solvePnP-based head pose and microsleep detection."""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np


# 3-D model reference points (canonical face model, mm units)
MODEL_POINTS_3D = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip
    [0.0,  -330.0, -65.0  ],   # Chin
    [-225.0, 170.0, -135.0],   # Left eye outer corner
    [225.0,  170.0, -135.0],   # Right eye outer corner
    [-150.0,-150.0, -125.0],   # Left mouth corner
    [150.0, -150.0, -125.0],   # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
POSE_LM_INDICES = [1, 152, 263, 33, 287, 57]


class HeadPoseEstimator:
    """
    Estimates head pose (pitch, yaw, roll) from MediaPipe landmarks using solvePnP.

    Args:
        tilt_threshold: Pitch/yaw degrees beyond which nodding is flagged.
    """

    def __init__(self, tilt_threshold: float = 20.0) -> None:
        self.tilt_threshold = tilt_threshold
        self._pitch_history: deque[float] = deque(maxlen=60)

    # ------------------------------------------------------------------
    def _get_camera_matrix(self, frame_shape: tuple[int, ...]) -> np.ndarray:
        h, w = frame_shape[:2]
        focal = w
        return np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1    ],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    def estimate(self, frame: np.ndarray, landmarks: list[Any]) -> dict[str, Any]:
        """
        Compute Euler angles for the detected face.

        Args:
            frame: Current BGR video frame (used for camera intrinsics).
            landmarks: MediaPipe face mesh landmark list.

        Returns:
            dict with keys: pitch, yaw, roll, is_nodding, tilt_direction, confidence.
        """
        h, w = frame.shape[:2]
        image_points = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in POSE_LM_INDICES],
            dtype=np.float64,
        )
        cam_matrix = self._get_camera_matrix(frame.shape)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                    "is_nodding": False, "tilt_direction": "center", "confidence": 0.0}

        rot_mat, _ = cv2.Rodrigues(rvec)
        pose_mat   = cv2.hconcat([rot_mat, tvec])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(
            cv2.hconcat([rot_mat, tvec.reshape(3, 1)])
        )

        pitch = float(euler[0][0])
        yaw   = float(euler[1][0])
        roll  = float(euler[2][0])

        self._pitch_history.append(pitch)

        is_nodding = abs(pitch) > self.tilt_threshold

        if abs(pitch) > abs(yaw):
            tilt_direction = "down" if pitch > 0 else "up"
        else:
            tilt_direction = "right" if yaw > 0 else "left"

        confidence = min(1.0, max(abs(pitch), abs(yaw)) / (self.tilt_threshold * 2))

        return {
            "pitch":          round(pitch, 2),
            "yaw":            round(yaw, 2),
            "roll":           round(roll, 2),
            "is_nodding":     is_nodding,
            "tilt_direction": tilt_direction,
            "confidence":     round(confidence, 3),
        }

    # ------------------------------------------------------------------
    def detect_microsleep(self, window: int = 30) -> bool:
        """
        Detect microsleep signature: head drops then snaps back within ~1 second.

        Args:
            window: Number of recent frames to analyse.

        Returns:
            True if microsleep pattern detected.
        """
        if len(self._pitch_history) < window:
            return False

        recent = list(self._pitch_history)[-window:]
        peak   = max(recent)
        last   = recent[-1]

        # Drop > threshold and subsequent recovery to near-center
        return peak > self.tilt_threshold and abs(last) < self.tilt_threshold * 0.5

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset pitch history."""
        self._pitch_history.clear()
