"""Public feature operations required by abnormality localization."""

from __future__ import annotations

import numpy as np


def position_summary(positions: np.ndarray) -> np.ndarray:
    """Summarize each path by mean, standard deviation, and displacement."""
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("positions must have shape [batch, path_points, 3]")
    mean = values.mean(axis=1)
    standard_deviation = values.std(axis=1)
    displacement = values[:, -1, :] - values[:, 0, :]
    return np.concatenate([mean, standard_deviation, displacement], axis=1).astype(np.float32)
