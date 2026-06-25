"""EXPERIMENTAL X1 — AttentionScorer: ML-based fatigue onset prediction."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class AttentionScorer:
    """
    Experimental attention scorer using a weighted ensemble of sensor inputs
    and linear regression on a rolling window to predict fatigue onset.

    All computation is pure NumPy — no external model file required.
    """

    # Rolling window: ~10 seconds at 30 fps
    WINDOW = 300

    def __init__(self) -> None:
        self._score_history: deque[float] = deque(maxlen=self.WINDOW)
        self._time_history:  deque[float] = deque(maxlen=self.WINDOW)
        self._frame_index:   int          = 0

    # ------------------------------------------------------------------
    def score(
        self,
        eye_metrics:   dict[str, Any],
        head_metrics:  dict[str, Any],
        blink_metrics: dict[str, Any],
    ) -> float:
        """
        Compute attention score 0.0 (distracted) – 1.0 (fully attentive).

        Args:
            eye_metrics:   Output of EyeDetector.detect().
            head_metrics:  Output of HeadPoseEstimator.estimate().
            blink_metrics: Dict with blink_rate key.

        Returns:
            Attention score in [0, 1].
        """
        ear        = eye_metrics.get("ear_avg", 0.3)
        pitch      = abs(head_metrics.get("pitch", 0.0))
        blink_rate = blink_metrics.get("blink_rate", 15.0)

        # Normalise to 0–1 contributions (1 = good)
        ear_score   = min(1.0, ear / 0.35)
        head_score  = max(0.0, 1.0 - pitch / 30.0)
        blink_score = 1.0 if 10 <= blink_rate <= 25 else max(0.0, 1.0 - abs(blink_rate - 17.5) / 17.5)

        attention = (ear_score * 0.50 + head_score * 0.30 + blink_score * 0.20)
        attention = round(float(np.clip(attention, 0.0, 1.0)), 4)

        self._score_history.append(attention)
        self._time_history.append(float(self._frame_index))
        self._frame_index += 1

        return attention

    # ------------------------------------------------------------------
    def predict_fatigue_onset(self, fps: float = 30.0) -> dict[str, Any]:
        """
        Predict minutes until fatigue using linear regression on score trend.

        Args:
            fps: Frames per second (used to convert frames → time).

        Returns:
            Dict with minutes_to_fatigue, confidence, recommendation.
        """
        n = len(self._score_history)
        if n < 30:
            return {
                "minutes_to_fatigue": -1,
                "confidence": 0.0,
                "recommendation": "Not enough data",
            }

        x = np.array(self._time_history, dtype=np.float64)
        y = np.array(self._score_history, dtype=np.float64)

        # Linear regression: y = slope * x + intercept
        x_mean, y_mean = x.mean(), y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom < 1e-9:
            slope = 0.0
        else:
            slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)

        intercept = y_mean - slope * x_mean
        current_score = float(y[-1])

        # Fatigue threshold: attention < 0.4
        fatigue_thr = 0.4
        if slope >= 0 or current_score <= fatigue_thr:
            minutes = 0.0
            confidence = 0.9 if current_score <= fatigue_thr else 0.1
        else:
            # frames until score hits threshold
            frames_left = (fatigue_thr - intercept) / slope - x[-1]
            minutes     = max(0.0, round(frames_left / fps / 60.0, 1))
            # Confidence based on R-squared
            y_pred = slope * x + intercept
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y_mean) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
            confidence = round(float(np.clip(r2, 0.0, 1.0)), 3)

        if minutes == 0:
            rec = "⚠ Fatigue detected — stop driving"
        elif minutes < 5:
            rec = "⚠ Fatigue imminent — take a break soon"
        elif minutes < 15:
            rec = "ℹ Mild fatigue trend — stay alert"
        else:
            rec = "✓ Attention level normal"

        return {
            "minutes_to_fatigue": minutes,
            "confidence":         confidence,
            "recommendation":     rec,
        }
