"""Run the core algorithms on tiny synthetic inputs for demonstration only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dt_shopfloor import DTSBatch, MultiDimensionalDTSModel, rank_modules


def _base_path(line: float, offset: float = 0.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 8)
    y = np.full(8, line + offset)
    z = np.zeros(8)
    return np.stack([x, y, z], axis=1)


def _normal_batch(lines: list[float], offsets: list[float]) -> DTSBatch:
    size = len(lines)
    events = np.tile(np.asarray([1, 2, 3, 0, 0]), (size, 1))
    lengths = np.full(size, 3)
    positions = np.stack([_base_path(line, offset) for line, offset in zip(lines, offsets)])
    states = np.tile(np.asarray([0, 1, 1, 0, 0, 0]), (size, 1))
    actions = np.tile(np.asarray([1, 2, 3, 0, 0]), (size, 1))
    return DTSBatch(events, lengths, positions, states, actions)


def _query_batch() -> DTSBatch:
    normal = _base_path(0.0)
    cross_line = _base_path(2.0)
    cross_line[:, 1] = np.linspace(2.0, 0.0, cross_line.shape[0])
    return DTSBatch(
        events=np.asarray([[1, 2, 3, 0, 0], [1, 2, 2, 3, 3]]),
        sequence_lengths=np.asarray([3, 5]),
        positions=np.stack([normal, cross_line]),
        states=np.asarray(
            [
                [0, 1, 1, 0, 0, 0],
                [0, 1, 0, 1, 1, 0],
            ]
        ),
        actions=np.asarray(
            [
                [1, 2, 3, 0, 0],
                [1, 2, 3, 2, 3],
            ]
        ),
    )


def main() -> None:
    """Exercise semantic matching, fitting, and multi-channel inference."""
    scores, selected = rank_modules(
        query_quant=np.asarray([6.0]),
        query_qual=np.asarray([1]),
        query_semantic=np.asarray([1, 1, 1, 0]),
        module_quant=np.asarray([[4.0], [6.0], [5.0]]),
        module_qual=np.asarray([[0], [1], [2]]),
        module_semantic=np.asarray([[1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 1, 0]]),
        weights=[0.35, 0.25, 0.40],
    )
    train = _normal_batch([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [-0.02, 0.02] * 3)
    validation = _normal_batch([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    model = MultiDimensionalDTSModel(seed=2023).fit(train, validation)
    query = _query_batch()
    detections = model.predict(query)
    signals = model.signals(query)

    assert scores.shape == (3,) and np.isfinite(scores).all()
    assert selected == 1
    assert detections.shape == (2,) and set(detections.tolist()).issubset({0, 1})
    assert signals.petri_status.shape == detections.shape
    print("Semantic matching: passed")
    print("Core inference: passed")
    print(f"Synthetic samples checked: {len(query)}")
    print(f"Detected abnormal cases: {int(detections.sum())}")


if __name__ == "__main__":
    main()
