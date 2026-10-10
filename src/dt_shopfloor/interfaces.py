"""Public, data-source-independent input interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EVENT_PAD, EVENT_E1, EVENT_E2, EVENT_E3 = 0, 1, 2, 3
STATE_DORMANCY, STATE_TRANSFER = 0, 1
ACTION_NONE, ACTION_WAKEUP, ACTION_MOVE, ACTION_UNLOAD = 0, 1, 2, 3


@dataclass(frozen=True)
class DTSBatch:
    """Batch of caller-supplied event, position, state, and action arrays."""

    events: np.ndarray
    sequence_lengths: np.ndarray
    positions: np.ndarray
    states: np.ndarray
    actions: np.ndarray

    def validate(self) -> "DTSBatch":
        """Validate shapes without assuming an industrial data source."""
        events = np.asarray(self.events)
        lengths = np.asarray(self.sequence_lengths).reshape(-1)
        positions = np.asarray(self.positions)
        states = np.asarray(self.states)
        actions = np.asarray(self.actions)
        if events.ndim != 2:
            raise ValueError("events must have shape [batch, event_steps]")
        batch_size, event_steps = events.shape
        if lengths.shape != (batch_size,):
            raise ValueError("sequence_lengths must contain one value per sample")
        if positions.ndim != 3 or positions.shape[0] != batch_size or positions.shape[2] != 3:
            raise ValueError("positions must have shape [batch, path_points, 3]")
        if states.shape != (batch_size, event_steps + 1):
            raise ValueError("states must have one more step than events")
        if actions.shape != events.shape:
            raise ValueError("actions must align with events")
        if np.any(lengths < 1) or np.any(lengths > event_steps):
            raise ValueError("sequence_lengths contain an invalid value")
        if not np.isfinite(positions).all():
            raise ValueError("positions contain non-finite values")
        return self

    def __len__(self) -> int:
        return int(np.asarray(self.events).shape[0])
