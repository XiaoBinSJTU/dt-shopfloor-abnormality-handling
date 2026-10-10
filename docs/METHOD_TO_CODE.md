# Method-to-Code Map

This document maps the paper's computational method steps to the public reference implementation. 

## 1. Semantic representation and matching

**Code:** `src/dt_shopfloor/semantic_matching.py`

- `quantitative_similarity`: normalized numerical-attribute similarity.
- `qualitative_similarity`: exact-match similarity for categorical attributes.
- `semantic_similarity`: Jaccard similarity for semantic attribute sets.
- `rank_modules`: weighted combination and reusable-module ranking.


## 2. Equipment behavior-path modeling

**Code:** `src/dt_shopfloor/behavior.py`

- `LEGAL_TRANSITIONS`: public state-action-state rules used by the AGV example.
- `mismatch_count`: deviation count for one behavior path.
- `batch_mismatch_counts`: vector of deviation counts for a batch.

State and action identifiers are defined in `src/dt_shopfloor/interfaces.py`.

## 3. Labeled Petri-net abnormality detection

**Code:** `src/dt_shopfloor/petri_net.py`

- `APASPetriMonitor.detect`: maps an observable event prefix to the status of `epsilon9` and `epsilon10`.
- `APASPetriMonitor.state_space_cost`: reports the node counts represented by the original and transformed reference monitors.

Status codes are `NORMAL=0`, `UNCERTAIN=1`, and `ABNORMAL=2`. The case-study prefix checks are covered by unit tests.

## 4. Behavior-path localization

**Code:** `src/dt_shopfloor/features.py` and `src/dt_shopfloor/core.py`

- `position_summary`: mean, standard deviation, and displacement for each three-dimensional path.
- `NumpyKMeans.fit`: deterministic clustering of caller-verified normal path summaries.
- `NumpyKMeans.distance`: distance to the nearest normal cluster.

The public interface requires the caller to identify normal fitting paths. 

## 5. Multi-dimensional abnormality decision

**Code:** `src/dt_shopfloor/core.py::MultiDimensionalDTSModel`

1. `fit` learns spatial clusters and a validation-based distance threshold from normal-only inputs.
2. `signals` computes Petri-net, behavior, and spatial evidence independently.
3. `predict` combines the evidence into binary detection codes.

The optional `without_petri`, `without_behavior`, and `without_spatial` switches expose the contribution of each code path for interface inspection. 

## 6. Public input interface

**Code:** `src/dt_shopfloor/interfaces.py::DTSBatch`

Expected shapes:

| Field | Shape | Meaning |
| --- | --- | --- |
| `events` | `[batch, event_steps]` | Integer observable-event identifiers with zero padding |
| `sequence_lengths` | `[batch]` | Valid event count for each sample |
| `positions` | `[batch, path_points, 3]` | Three-dimensional path coordinates |
| `states` | `[batch, event_steps + 1]` | Equipment state identifiers |
| `actions` | `[batch, event_steps]` | Equipment action identifiers |

