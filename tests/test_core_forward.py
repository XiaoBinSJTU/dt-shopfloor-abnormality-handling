"""Core forward-path tests using synthetic inputs only."""

import numpy as np

from dt_shopfloor import DTSBatch, MultiDimensionalDTSModel


def _batch(lines: list[float]) -> DTSBatch:
    size = len(lines)
    x = np.linspace(0.0, 1.0, 6)
    positions = np.stack([np.stack([x, np.full(6, line), np.zeros(6)], axis=1) for line in lines])
    return DTSBatch(
        events=np.tile(np.asarray([1, 2, 3, 0, 0]), (size, 1)),
        sequence_lengths=np.full(size, 3),
        positions=positions,
        states=np.tile(np.asarray([0, 1, 1, 0, 0, 0]), (size, 1)),
        actions=np.tile(np.asarray([1, 2, 3, 0, 0]), (size, 1)),
    )


def test_core_fit_and_forward_are_finite():
    train = _batch([0.0, 0.01, 1.0, 1.01, 2.0, 2.01])
    validation = _batch([0.0, 1.0, 2.0])
    model = MultiDimensionalDTSModel(seed=2023).fit(train, validation)
    predictions = model.predict(validation)
    signals = model.signals(validation)
    assert predictions.shape == (3,)
    assert np.isfinite(model.localizer.centers).all()
    assert signals.petri_status.shape == (3,)
