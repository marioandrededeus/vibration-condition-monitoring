---

# Vibration Condition Monitoring – Frequency-Domain Carpet Detection and Looseness Classification

---

## Overview

This repository implements a vibration-based condition monitoring pipeline structured as a two-stage analytical system:

* **Part 1:** Frequency-domain carpet detection
* **Part 2:** Structural looseness classification using metadata integration and supervised learning

The project combines:

* Digital Signal Processing (DSP)
* Physically grounded modeling
* Structured metadata integration
* Machine learning validation
* Modular, production-oriented design

Before implementing detection or classification logic, signal consistency and spectral assumptions are validated to ensure modeling decisions are grounded in signal processing fundamentals.

---

# Scientific Motivation

The broadband detection methodology adopted in this repository is conceptually inspired by prior research on lubrication-related bearing faults.

Xu et al., *“Vibration-based identification of lubrication starved bearing conditions”* (Mechanical Systems and Signal Processing), demonstrate that starved lubrication leads to increased broadband energy and elevated RMS levels. However, their study also shows that RMS and global vibration levels vary significantly with rotational speed, making purely time-domain metrics unreliable when speed is not controlled.

In other words:

* Lubrication degradation increases broadband spectral energy.
* Rotational speed variations can produce similar increases in global vibration levels.
* Without speed information, distinguishing operational effects from degradation becomes less precise if relying solely on time-domain statistics.

Because rotational speed is not provided in Part 1 of this dataset, the methodology prioritizes:

* Frequency-domain analysis
* Relative noise-floor modeling in dB
* Broadband elevation detection above 1000 Hz
* Robust baseline estimation rather than absolute energy thresholds

This approach transfers the conceptual insight from Xu et al., that broadband spectral behavior is more robust than raw RMS, while acknowledging the limitation imposed by missing operational variables.

---

# Repository Structure

```
vibration-condition-monitoring/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── src/
│   └── tractian_cm/
│       ├── io/               # Data loading and validation
│       │   ├── loader.py
│       │   └── metadata_part2.py
│       │
│       ├── dsp/              # Spectral analysis utilities (FFT, PSD, etc.)
│       │
│       ├── part1/            # Carpet detection logic
│       │
│       └── part2/            # Looseness classification pipeline
│           ├── sample_index.py
│           └── (modeling modules to be added)
│
├── notebooks/
│   ├── 01_part1_sanity_and_psd.ipynb
│   └── 01_part2_pipeline.ipynb
│
├── reports/
│   └── part2/
│       └── 00_plan.md
│
└── data/                     # Local datasets (not versioned)
```

---

# Part 1 – Carpet Detection Strategy

Carpet regions are modeled as:

* Broadband spectral elevation
* Persistent increase in spectral noise floor
* Occurrence in the frequency region above 1000 Hz

### Detection Pipeline

1. Validate sampling consistency.
2. Estimate PSD using Welch’s method.
3. Restrict analysis to frequencies ≥ 1000 Hz.
4. Estimate robust spectral baseline using rolling median in dB.
5. Detect regions exceeding baseline by a relative threshold.
6. Enforce minimum bandwidth constraints.

This approach prioritizes interpretability and physical consistency over heuristic peak detection.

---

# Part 2 – Structural Looseness Classification

The second stage focuses on supervised classification of structural looseness using:

* Raw tri-axial acceleration signals
* Metadata-driven axis-to-orientation mapping
* Time-domain and frequency-domain feature extraction
* Structured validation and reproducible modeling

Unlike Part 1, Part 2 introduces:

* Explicit target labels (`healthy` vs `structural_looseness`)
* Sensor-dependent orientation mapping (axisX/axisY/axisZ → horizontal/vertical/axial)
* Rotational speed (rpm) information
* Supervised modeling framework

---

## Phase-Based Development Strategy (Part 2)

### Phase 1 – Data Preparation (Commit 1)

Implemented in:

* `src/tractian_cm/io/metadata_part2.py`
* `src/tractian_cm/part2/sample_index.py`
* `notebooks/01_part2_pipeline.ipynb`
* `reports/part2/00_plan.md`

This phase includes:

* Parsing and validating metadata (`part_3_metadata.csv`, `test_metadata.csv`)
* Strict validation of orientation mapping:

  * Keys must be `axisX`, `axisY`, `axisZ`
  * Values must be exactly one of each: `horizontal`, `vertical`, `axial`
* Construction of a unified **samples index**, including:

  * `train_labeled`
  * `train_unlabeled`
  * `test`

No signal modeling or feature extraction is introduced in this phase.

The objective is to ensure:

* Schema consistency
* Correct metadata integration
* Clean separation between I/O and modeling logic

---

### Phase 2 – Exploratory Data Analysis (Upcoming)

Planned deliverables:

* Class balance analysis
* Distribution of rpm and sensor_id
* Univariate feature analysis
* Bivariate feature exploration
* Spectral comparison between healthy and looseness samples
* Insight-driven feature selection

All feature extraction will reuse centralized DSP utilities.

---

### Phase 3 – Modeling and Validation (Upcoming)

Planned modeling strategy:

* Physics-informed heuristics (baseline)
* Logistic Regression
* Random Forest
* Optional deep learning (Keras/TensorFlow) if justified by data volume

Validation strategy:

* Group-aware cross-validation (e.g., by sensor_id)
* Confusion matrix
* ROC AUC
* PR AUC
* Threshold calibration
* Optional probability calibration

Model interface must follow the provided template:

```python
class LoosenessModel:
    def predict(self, wave_hor, wave_ver, wave_axi) -> bool
    def score(self, wave_hor, wave_ver, wave_axi) -> float
```

The model will receive **orientation-mapped signals**, not raw axis signals.

---

# Design Principles

* Separate physical signal validation from machine learning.
* Avoid absolute thresholds when signal energy varies across samples.
* Use relative spectral elevation for broadband detection.
* Maintain modular and deployable project structure.
* Centralize spectral computation for consistency across notebooks and modules.
* Enforce strict metadata validation before model execution.
* Ensure that webapp integration reuses the same feature pipeline as training.

---

# Environment Setup

Create and activate virtual environment (Windows example):

```
python -m venv tractianenv
tractianenv\Scripts\activate
```

Install project in editable mode:

```
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

---

# References

Xu, X., Liao, X., Zhou, T., He, Z., & Hu, H.
“Vibration-based identification of lubrication starved bearing conditions.”
Measurement, 226:114156 (2024)
[https://doi.org/10.1016/j.measurement.2024.114156](https://doi.org/10.1016/j.measurement.2024.114156)

This work provides conceptual grounding for frequency-domain broadband analysis as a robust alternative to purely time-domain metrics.

---