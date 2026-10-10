"""Core multi-dimensional abnormality detection and localization logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .behavior import batch_mismatch_counts
from .features import position_summary
from .interfaces import DTSBatch
from .petri_net import ABNORMAL, UNCERTAIN, APASPetriMonitor


class NumpyKMeans:
    """Deterministic K-means used for behavior-path localization."""

    def __init__(self, clusters: int = 3, max_iterations: int = 100, seed: int = 2023) -> None:
        self.clusters = clusters
        self.max_iterations = max_iterations
        self.seed = seed
        self.centers: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "NumpyKMeans":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < self.clusters:
            raise ValueError("K-means requires a two-dimensional array with enough rows")
        rng = np.random.default_rng(self.seed)
        centers = values[rng.choice(values.shape[0], self.clusters, replace=False)].copy()
        for _ in range(self.max_iterations):
            distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
            assignments = np.argmin(distances, axis=1)
            updated = centers.copy()
            for cluster in range(self.clusters):
                members = values[assignments == cluster]
                if members.size:
                    updated[cluster] = members.mean(axis=0)
            if np.allclose(updated, centers, atol=1e-7):
                centers = updated
                break
            centers = updated
        self.centers = centers
        return self

    def distance(self, values: np.ndarray) -> np.ndarray:
        if self.centers is None:
            raise RuntimeError("K-means must be fitted before distance calculation")
        values = np.asarray(values, dtype=np.float64)
        distances = np.linalg.norm(values[:, None, :] - self.centers[None, :, :], axis=2)
        return distances.min(axis=1)


@dataclass(frozen=True)
class DetectionSignals:
    """Independent Petri-net, behavior, and spatial evidence channels."""

    petri_status: np.ndarray
    behavior_flag: np.ndarray
    spatial_flag: np.ndarray


class MultiDimensionalDTSModel:
    """Fuse Petri-net, behavior-path, and spatial localization evidence."""

    def __init__(self, seed: int = 2023) -> None:
        self.monitor = APASPetriMonitor()
        self.localizer = NumpyKMeans(clusters=3, seed=seed)
        self.spatial_threshold: float | None = None

    def fit(
        self,
        normal_train: DTSBatch,
        normal_validation: DTSBatch | None = None,
    ) -> "MultiDimensionalDTSModel":
        """Fit spatial clusters using caller-verified normal paths only."""
        normal_train.validate()
        if len(normal_train) < 3:
            raise ValueError("At least three normal paths are required")
        train_summary = position_summary(normal_train.positions)
        self.localizer.fit(train_summary)
        reference = self.localizer.distance(train_summary)
        if normal_validation is not None:
            normal_validation.validate()
            reference = self.localizer.distance(position_summary(normal_validation.positions))
        self.spatial_threshold = float(np.quantile(reference, 0.99) + max(reference.std(), 1e-6))
        return self

    def signals(self, batch: DTSBatch) -> DetectionSignals:
        """Compute the three evidence channels without retaining caller data."""
        batch.validate()
        if self.spatial_threshold is None:
            raise RuntimeError("Model must be fitted before inference")
        petri_status = []
        for events, length in zip(batch.events, batch.sequence_lengths):
            result = self.monitor.detect(tuple(events[: int(length)].tolist()))
            petri_status.append(result.epsilon10)
        behavior = batch_mismatch_counts(batch.states, batch.actions, batch.sequence_lengths) > 0
        spatial_distance = self.localizer.distance(position_summary(batch.positions))
        return DetectionSignals(
            petri_status=np.asarray(petri_status, dtype=np.int64),
            behavior_flag=behavior.astype(bool),
            spatial_flag=(spatial_distance > self.spatial_threshold),
        )

    def predict(self, batch: DTSBatch, variant: str = "full") -> np.ndarray:
        """Return binary detection codes for the full method or a documented switch."""
        signals = self.signals(batch)
        certain = signals.petri_status == ABNORMAL
        uncertain = signals.petri_status == UNCERTAIN
        behavior = signals.behavior_flag
        spatial = signals.spatial_flag
        if variant == "full":
            prediction = certain | (uncertain & (behavior | spatial)) | (behavior & spatial)
        elif variant == "without_petri":
            prediction = behavior & spatial
        elif variant == "without_behavior":
            prediction = certain | (uncertain & spatial)
        elif variant == "without_spatial":
            prediction = certain | (uncertain & behavior)
        else:
            raise ValueError(f"Unknown model variant: {variant}")
        return prediction.astype(np.int64)
