"""Core semantic, behavior, and Petri-net logic tests."""

import numpy as np

from dt_shopfloor import ABNORMAL, NORMAL, UNCERTAIN, APASPetriMonitor, mismatch_count, rank_modules


def test_semantic_matching_selects_the_best_module():
    scores, selected = rank_modules(
        np.asarray([6.0]),
        np.asarray([1]),
        np.asarray([1, 1, 1]),
        np.asarray([[4.0], [6.0]]),
        np.asarray([[0], [1]]),
        np.asarray([[1, 0, 0], [1, 1, 1]]),
        [0.35, 0.25, 0.40],
    )
    assert selected == 1
    assert scores[1] > scores[0]


def test_behavior_mismatch_is_computed():
    states = np.asarray([0, 1, 0, 1])
    actions = np.asarray([1, 2, 3])
    assert mismatch_count(states, actions, 3) == 2


def test_petri_monitor_matches_documented_prefix_states():
    monitor = APASPetriMonitor()
    assert monitor.detect([1]).epsilon10 == NORMAL
    assert monitor.detect([1, 2]).epsilon10 == UNCERTAIN
    assert monitor.detect([1, 2, 2, 3, 3]).epsilon10 == ABNORMAL
