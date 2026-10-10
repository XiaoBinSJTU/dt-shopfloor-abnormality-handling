# Multi-dimensional Modeling and Abnormality Handling of Digital Twin Shop Floor

[Bin Xiao's GitHub profile](https://github.com/XiaoBinSJTU) · [Paper DOI](https://doi.org/10.1016/j.jii.2023.100492)

## Paper Information

**Bin Xiao, Qinglin Qi, and Fei Tao.** “Multi-dimensional modeling and abnormality handling of digital twin shop floor.” *Journal of Industrial Information Integration*, 35 (2023), 100492.

## Graphical Abstract

![Graphical Abstract](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/02-shopfloor-modeling-abnormality.svg)

## Research Highlights

Digital twin shop floors require consistent geometric, behavioral, and process-level representations. This reference implementation exposes semantic model matching, equipment state-action consistency checks, labeled Petri-net observation logic, K-means path localization, and multi-channel abnormality decisions. It is designed for method inspection and interface validation with synthetic inputs, without releasing industrial records, trained parameters, or paper-level experiment outputs.

## Method Overview

### 1. Geometric model semantic matching

Quantitative, qualitative, and semantic attributes are compared separately and combined through configurable weights. The implementation follows the similarity structure described in Equations 27-32 and returns ranked reusable modules.

### 2. Equipment behavior-path consistency

AGV production behavior is represented by state-action-state transitions. A behavior path is checked against legal wake-up, transfer, unloading, and dormancy transitions to localize deviations from the expected process.

### 3. Labeled Petri-net observation

The APAS monitor maps observable event prefixes to normal, uncertain, or abnormal states for the two abnormal unobservable transitions discussed in the case study. The compact monitor also exposes the original-versus-transformed node counts used by this reference implementation.

### 4. Spatial localization and evidence fusion

Deterministic K-means models summaries of caller-verified normal paths. Petri-net status, behavior consistency, and spatial outlier evidence are then combined by the reference decision logic. Optional evidence switches are retained for code inspection only; no paper comparison or ablation results are included.

## Method-to-Code Map

| Paper method step | Public code entry | Purpose |
| --- | --- | --- |
| Semantic matching | `src/dt_shopfloor/semantic_matching.py::rank_modules` | Weighted quantitative, qualitative, and semantic similarity |
| Equipment behavior model | `src/dt_shopfloor/behavior.py::mismatch_count` | State-action-state consistency checking |
| Labeled Petri-net detection | `src/dt_shopfloor/petri_net.py::APASPetriMonitor` | Observable-prefix abnormality status |
| Position-path localization | `src/dt_shopfloor/core.py::NumpyKMeans` | Normal-path clustering and distance calculation |
| Multi-dimensional decision | `src/dt_shopfloor/core.py::MultiDimensionalDTSModel` | Fit, evidence extraction, and detection |
| Data-source-independent input | `src/dt_shopfloor/interfaces.py::DTSBatch` | Validated array interface without private loaders |

See [`docs/METHOD_TO_CODE.md`](docs/METHOD_TO_CODE.md) for inputs, outputs, and implementation notes.

## Quick Start

The following commands are intended for Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python examples\minimal_demo.py
python -m pytest -q
```

## Citation

```bibtex
@article{xiao2023multidimensional,
  title   = {Multi-dimensional modeling and abnormality handling of digital twin shop floor},
  author  = {Xiao, Bin and Qi, Qinglin and Tao, Fei},
  journal = {Journal of Industrial Information Integration},
  volume  = {35},
  pages   = {100492},
  year    = {2023},
  doi     = {10.1016/j.jii.2023.100492}
}
```

## Author

Research profile and related publications: [XiaoBinSJTU](https://github.com/XiaoBinSJTU).

