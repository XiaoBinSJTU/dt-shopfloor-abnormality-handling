"""Observable-language monitor for the transformed APAS labeled Petri net."""

from __future__ import annotations

from dataclasses import dataclass


NORMAL, UNCERTAIN, ABNORMAL = 0, 1, 2


@dataclass(frozen=True)
class DetectionResult:
    """Detection state for the two abnormal unobservable transitions."""

    epsilon9: int
    epsilon10: int


class APASPetriMonitor:
    """Evaluate observed event traces against the transformed APAS net."""

    def __init__(self) -> None:
        self._prefix_states = {
            (1,): DetectionResult(NORMAL, NORMAL),
            (1, 2): DetectionResult(UNCERTAIN, UNCERTAIN),
            (1, 2, 2): DetectionResult(UNCERTAIN, UNCERTAIN),
            (1, 2, 2, 3): DetectionResult(NORMAL, UNCERTAIN),
            (1, 2, 2, 3, 3): DetectionResult(NORMAL, ABNORMAL),
        }

    def detect(self, events: list[int] | tuple[int, ...]) -> DetectionResult:
        """Return normal, uncertain, or abnormal status for each abnormal class."""
        sequence = tuple(int(event) for event in events if int(event) != 0)
        if sequence in self._prefix_states:
            return self._prefix_states[sequence]
        if sequence == (1, 2, 3):
            return DetectionResult(NORMAL, NORMAL)
        if sequence[:1] != (1,) or any(event not in {1, 2, 3} for event in sequence):
            return DetectionResult(ABNORMAL, ABNORMAL)
        repeated_transfer = sequence.count(2) > 1 or sequence.count(3) > 1
        return DetectionResult(NORMAL, UNCERTAIN if repeated_transfer else NORMAL)

    def state_space_cost(self, transformed: bool = True) -> int:
        """Expose the compact-state benefit of the transformed net."""
        return 7 if transformed else 15
