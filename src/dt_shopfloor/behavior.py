"""Equipment behavior-path consistency checks."""

from __future__ import annotations

import numpy as np

from .interfaces import ACTION_MOVE, ACTION_NONE, ACTION_UNLOAD, ACTION_WAKEUP, STATE_DORMANCY, STATE_TRANSFER


LEGAL_TRANSITIONS = {
    (STATE_DORMANCY, ACTION_WAKEUP, STATE_TRANSFER),
    (STATE_TRANSFER, ACTION_MOVE, STATE_TRANSFER),
    (STATE_TRANSFER, ACTION_UNLOAD, STATE_DORMANCY),
    (STATE_DORMANCY, ACTION_NONE, STATE_DORMANCY),
}


def mismatch_count(states: np.ndarray, actions: np.ndarray, length: int) -> int:
    """Count deviations from the established AGV behavior path."""
    mismatches = 0
    for index in range(int(length)):
        transition = (int(states[index]), int(actions[index]), int(states[index + 1]))
        mismatches += int(transition not in LEGAL_TRANSITIONS)
    return mismatches


def batch_mismatch_counts(
    states: np.ndarray, actions: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Evaluate behavior consistency for a batch of AGV paths."""
    return np.asarray(
        [mismatch_count(s, a, int(length)) for s, a, length in zip(states, actions, lengths)],
        dtype=np.int64,
    )
