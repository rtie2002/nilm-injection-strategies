"""EWMA post-processing for augmented Geng models (EasyS2S_test.py)."""

from __future__ import annotations

import numpy as np


def conditional_ema(predictions: np.ndarray, threshold: float, alpha: float = 0.9) -> np.ndarray:
    smoothed = np.zeros_like(predictions, dtype=np.float64)
    active = False
    last_valid = 0.0
    consecutive_above = 0

    for t, value in enumerate(predictions):
        if value > threshold:
            consecutive_above += 1
            if consecutive_above >= 1:
                if not active:
                    smoothed[t] = value
                    last_valid = value
                    active = True
                else:
                    smoothed[t] = alpha * value + (1.0 - alpha) * last_valid
                    last_valid = smoothed[t]
            else:
                smoothed[t] = value
        else:
            smoothed[t] = value
            active = False
            consecutive_above = 0

    return smoothed
