Perfeito. Abaixo está o **README completo revisado**, já incorporando:

* ✔ Ajuste `loaders.py`
* ✔ Bloco físico explícito sobre looseness
* ✔ Refinamento técnico do sampling-rate section
* ✔ Pequeno upgrade de linguagem (mais executivo e preciso)
* ✔ Narrativa mais fluida

Você pode substituir o README atual por este.

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
* Supervised learning validation
* Modular, production-oriented architecture

Before implementing detection or classification logic, signal consistency and spectral assumptions are validated to ensure modeling decisions are grounded in signal processing fundamentals.

---

# Scientific Motivation

The broadband detection methodology adopted in Part 1 is conceptually inspired by prior research on lubrication-related bearing faults.

Xu et al., *“Vibration-based identification of lubrication starved bearing conditions”* (Mechanical Systems and Signal Processing), demonstrate that starved lubrication leads to increased broadband energy and elevated RMS levels. However, their study also shows that RMS and global vibration levels vary significantly with rotational speed, making purely time-domain metrics unreliable when speed is not controlled.

In summary:

* Lubrication degradation increases broadband spectral energy.
* Rotational speed variations can produce similar increases in vibration levels.
* Without speed information, distinguishing operational effects from degradation becomes less precise if relying solely on time-domain statistics.

Because rotational speed is not provided in Part 1 of this dataset, the methodology prioritizes:

* Frequency-domain analysis
* Relative noise-floor modeling in dB
* Broadband elevation detection above 1000 Hz
* Robust baseline estimation instead of absolute thresholds

This transfers the conceptual insight from Xu et al. that broadband spectral behavior is more robust than raw RMS, while acknowledging dataset limitations.

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
│       │   ├── loaders.py
│       │   └── metadata_part2.py
│       │
│       ├── dsp/              # Spectral analysis utilities (FFT, PSD, etc.)
│       │
│       ├── part1/            # Carpet detection logic
│       │
│       └── part2/            # Looseness classification pipeline
│           ├── sample_index.py
│           ├── orientation.py
│           ├── features.py
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

### Physical Motivation for Feature Design

Structural looseness is known to generate:

* Low-frequency structural excitation
* Harmonic amplification at multiples of the rotational frequency (1×, 2×, 3×)
* Possible sub-harmonic components (e.g., 0.5×) due to nonlinear mechanical contact

Feature engineering is therefore designed to capture:

* Rotation-synchronous components tied to rpm
* Low-frequency band energy ratios
* Harmonic amplitude ratios
* Directionally aggregated structural responses

This ensures that modeling remains physically interpretable and not purely statistical.

---

## Phase-Based Development Strategy (Part 2)

### Phase 1 – Data Preparation

Implemented in:

* `metadata_part2.py`
* `sample_index.py`
* `loaders.py`
* `orientation.py`
* `features.py`

This phase includes:

* Parsing and validating metadata (`part_3_metadata.csv`, `test_metadata.csv`)
* Strict validation of orientation mapping
* Construction of unified sample index
* Resolution-aware spectral feature extraction
* Generation of a structured feature matrix

The objective is to ensure:

* Schema consistency
* Correct metadata integration
* Clean separation between I/O and modeling logic
* Physically grounded feature extraction

---

### Sampling Rate Considerations

Sanity checks revealed a structural difference between training and test datasets:

* Training set: ~4 kHz sampling rate, 2048 samples (~0.5 s window)
* Test set: ~8 kHz sampling rate, 16384 samples (~2.0 s window)

This implies differences in:

* Spectral resolution (Δf = fs / N)
* Nyquist frequency
* Frequency coverage
* Time-window duration

High-frequency broadband metrics are therefore not directly comparable across splits.

### Mitigation Strategy

To ensure robustness and cross-dataset generalization:

* Harmonic detection is resolution-aware (tolerance ≥ 2 × spectral bin width)
* Features are anchored to physical rotational frequency (rpm-based)
* Low-frequency band ratios are prioritized
* Relative harmonic metrics are preferred over absolute broadband energy

Feature extraction is therefore **frequency-anchored rather than bin-index dependent**, ensuring comparability despite sampling mismatch.

---

### Phase 2 – Exploratory Data Analysis (Upcoming)

Planned deliverables:

* Class balance analysis
* Distribution of rpm and sensor_id
* Univariate feature analysis
* Bivariate feature exploration
* Spectral comparison between healthy and looseness samples
* Insight-driven feature refinement

---

### Phase 3 – Modeling and Validation (Upcoming)

Planned modeling strategy:

* Physics-informed baseline heuristics
* Logistic Regression
* Random Forest
* Optional deep learning (Keras/TensorFlow) if justified

Validation strategy:

* Group-aware cross-validation (e.g., by sensor_id)
* Confusion matrix
* ROC AUC
* PR AUC
* Threshold calibration
* Optional probability calibration

Model interface must follow:

```python
class LoosenessModel:
    def predict(self, wave_hor, wave_ver, wave_axi) -> bool
    def score(self, wave_hor, wave_ver, wave_axi) -> float
```

The model will receive **orientation-mapped signals**, not raw axis signals.

---

# Design Principles

* Separate physical signal validation from machine learning
* Avoid absolute thresholds when signal energy varies
* Use rotation-anchored features instead of bin-indexed features
* Prefer relative spectral metrics over raw amplitudes
* Maintain modular, production-oriented structure
* Centralize spectral computation
* Enforce strict metadata validation
* Ensure feature pipeline parity between training and deployment

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

Randall, R. B.
Vibration-based Condition Monitoring: Industrial, Aerospace and Automotive Applications.
John Wiley & Sons (2011)
https://doi.org/10.1002/9780470977668

Randall provides a comprehensive treatment of harmonic amplification, sub-harmonic components, and nonlinear vibration signatures associated with structural looseness and other rotating machinery faults.

---

# Final Assessment

The repository emphasizes:

* Physical interpretability
* Resolution-aware spectral modeling
* Domain-shift awareness
* Modular architecture
* Reproducible validation

The goal is not merely to build a classifier, but to construct a physically grounded and production-ready vibration monitoring pipeline.

---