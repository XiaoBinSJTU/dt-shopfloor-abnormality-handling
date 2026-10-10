"""Semantic matching corresponding to the paper's equations 27-32."""

from __future__ import annotations

import numpy as np


def quantitative_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute normalized quantitative similarity with zero-safe denominators."""
    query = np.asarray(query, dtype=float).reshape(1, -1)
    candidates = np.asarray(candidates, dtype=float)
    denominator = np.maximum(np.maximum(np.abs(query), np.abs(candidates)), 1e-12)
    return np.clip(1.0 - np.mean(np.abs(candidates - query) / denominator, axis=1), 0.0, 1.0)


def qualitative_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute exact-match similarity for qualitative attributes."""
    query = np.asarray(query).reshape(1, -1)
    return np.mean(np.asarray(candidates) == query, axis=1).astype(float)


def semantic_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute Jaccard similarity for semantic attribute sets."""
    query_mask = np.asarray(query).reshape(1, -1).astype(bool)
    candidate_mask = np.asarray(candidates).astype(bool)
    intersection = np.logical_and(candidate_mask, query_mask).sum(axis=1)
    union = np.logical_or(candidate_mask, query_mask).sum(axis=1)
    return np.divide(intersection, union, out=np.ones_like(intersection, dtype=float), where=union > 0)


def rank_modules(
    query_quant: np.ndarray,
    query_qual: np.ndarray,
    query_semantic: np.ndarray,
    module_quant: np.ndarray,
    module_qual: np.ndarray,
    module_semantic: np.ndarray,
    weights: list[float] | tuple[float, float, float],
) -> tuple[np.ndarray, int]:
    """Return weighted similarity scores and the best reusable module index."""
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.shape != (3,) or not np.isclose(weight_array.sum(), 1.0):
        raise ValueError("Semantic matching weights must be three values summing to one")
    components = np.stack(
        [
            quantitative_similarity(query_quant, module_quant),
            qualitative_similarity(query_qual, module_qual),
            semantic_similarity(query_semantic, module_semantic),
        ],
        axis=1,
    )
    scores = components @ weight_array
    return scores, int(np.argmax(scores))
