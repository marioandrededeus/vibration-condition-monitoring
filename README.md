# Vibration Condition Monitoring

## Frequency-Domain Carpet Detection and Structural Looseness Diagnosis

---

## Overview

This repository implements a vibration-based condition monitoring system structured as a two-stage analytical pipeline:

* **Part 1:** Broadband “carpet” detection in the frequency domain
* **Part 2:** Structural looseness diagnosis using rotation-synchronous harmonic behavior

The project combines:

* Digital Signal Processing (DSP)
* Physics-informed feature engineering
* Metadata integration and validation
* Controlled supervised model benchmarking
* Domain-shift awareness
* Production-oriented architecture
* Streamlit web application deployment

The objective is not merely to train a classifier, but to construct a **physically interpretable, resolution-aware, and deployable vibration monitoring solution**.

---

# Scientific Motivation

## Part 1 – Broadband Carpet Detection

The broadband detection methodology is conceptually inspired by research on lubrication-related bearing faults.

Xu et al., *“Vibration-based identification of lubrication starved bearing conditions”* (Measurement, 2024), demonstrate that lubrication starvation increases broadband spectral energy and RMS levels. However, RMS varies strongly with rotational speed, making purely time-domain metrics unreliable without speed control.

Because rotational speed is not provided in Part 1:

* Detection prioritizes frequency-domain analysis
* Relative spectral elevation (in dB) is used instead of absolute RMS
* Carpet regions are detected above 1000 Hz
* A robust rolling-median baseline is estimated

This ensures robustness despite missing operational variables.

---

# Repository Structure

```
vibration-condition-monitoring/
│
├── README.md
├── pyproject.toml
│
├── notebooks/
│ ├── 01_part1_sanity_and_psd.ipynb
│ └── 01_part2_pipeline.ipynb
│
└── src/
└── tractian_cm/
├── io/
│ ├── loaders.py
│ └── metadata_part2.py
│
├── dsp/
│ └── (spectral utilities: FFT/PSD helpers)
│
├── part1/
│ └── (carpet detection modules)
│
└── part2/
├── sample_index.py
├── orientation.py
├── features.py
└── (modeling / heuristic / artifacts)
```

---

# Part 1 – Carpet Detection Strategy

Carpet regions are modeled as:

* Broadband spectral elevation
* Persistent increase in spectral noise floor
* Occurrence above 1000 Hz

### Detection Pipeline

1. Validate sampling consistency
2. Estimate PSD using Welch’s method
3. Restrict analysis to ≥ 1000 Hz
4. Estimate rolling median baseline (dB)
5. Detect spectral elevation above relative threshold
6. Enforce minimum bandwidth constraints

This approach emphasizes interpretability over heuristic peak detection.

---

# Part 2 – Structural Looseness Diagnosis

## Physical Context

Structural looseness involves reduced clamping force in non-rotating components (e.g., motor bases, couplings, supports).

Unlike lubrication faults (high-frequency broadband energy), looseness primarily modifies:

* System stiffness
* Low-frequency structural response
* Harmonic content of vibration

Expected vibration signatures include:

* Increased amplitude at 1× rotational frequency
* Harmonic amplification at 2× and 3×
* Possible sub-harmonics (0.5×)
* Increased crest factor due to nonlinear contact

---

## Sampling-Rate Domain Shift

Sanity checks revealed a structural difference between train and test datasets:

| Split | Samples | fs (Hz) | Duration |
| ----- | ------- | ------- | -------- |
| Train | 2048    | ~4 kHz  | ~0.5 s   |
| Test  | 16384   | ~8 kHz  | ~2.0 s   |

This creates domain shift in:

* Spectral resolution (Δf)
* Nyquist frequency
* Time window length

### Mitigation Strategy

Feature engineering was designed to be:

* **Frequency-anchored** (harmonics tied to rpm)
* Resolution-aware (tolerance ≥ 2 × spectral bin width)
* Ratio-based (2×/1×, band ratios)
* Physically interpretable

This prevents overfitting to spectral resolution artifacts.

---

# Exploratory Findings

## Inverted Class Imbalance

Looseness represents ~67% of the dataset.
Optimizing recall alone would lead to degenerate always-positive models.

Model selection therefore prioritized:

* F1-score
* PR-AUC
* Recall monitored as safety constraint

---

## RPM Bias (1595 Regime)

All samples at rpm=1595 were labeled looseness.

This suggested:

* Potential dataset bias
* Hidden assembly conditions not represented by available features

To investigate, models were evaluated in two scenarios:

1. Full dataset
2. Restricted dataset (rpm=1598 only)

---

## Key Insight – Harmonic Interaction

When restricting to rpm=1598:

* Recall reached 1.0 across models
* Feature importance shifted significantly
* 2×/1× alone was insufficient
* Crest factor and 3×/1× gained relevance

This revealed that looseness behavior is captured through **combined harmonic amplification and nonlinear response**, not a single metric.

---

# Model Benchmarking

Three approaches were evaluated:

1. Single-feature physics heuristic (2×/1× threshold)
2. Logistic Regression
3. Random Forest (RandomizedSearchCV)

Validation protocol:

* Stratified 70/30 train-test split
* 5-fold CV on training data
* Threshold tuning on training folds only
* Strict data-leakage prevention
* Final evaluation on unseen holdout

---

# Final Model Selection

Although Random Forest achieved slightly higher ROC-AUC and PR-AUC, gains in F1 were marginal.

A refined **multi-feature physics-informed heuristic**, combining:

* 2×/1× harmonic ratio
* 3×/1× harmonic ratio
* Crest factor

achieved performance comparable to Random Forest while offering:

* Full interpretability
* Direct physical meaning
* Lower computational cost
* Simpler deployment
* No black-box behavior

Given the objectives of robustness, transparency, and physical coherence, the enhanced heuristic model was selected for deployment.

Random Forest remains as a benchmarking reference.

---

# Streamlit Web Application

A Streamlit application was implemented to demonstrate deployment readiness.

The app includes:

### Part 1

* PSD visualization
* Broadband carpet detection
* Interactive signal analysis

### Part 2

* Tri-axial signal upload
* Metadata parsing (orientation + rpm)
* Physics-based feature extraction
* Looseness prediction with probability score

The same feature extraction pipeline used in training is reused in the app, ensuring consistency between research and deployment.

---

# Sensor-Level Considerations

`sensor_id` was not explicitly modeled.

Potential future improvements:

* Group-aware cross-validation (GroupKFold)
* Cross-sensor robustness evaluation
* Sensor-level normalization strategies

This ensures generalization beyond laboratory conditions.

---

# Design Principles

* Physics before machine learning
* Investigate bias before model selection
* Avoid leakage at all stages
* Prefer frequency-anchored features
* Prioritize interpretability
* Ensure training–deployment parity

---

# Environment Setup

Create environment:

```
python -m venv tractianenv
tractianenv\Scripts\activate
```

Install:

```
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

Run app:

```
streamlit run app/streamlit_app.py
```

---

# References

Xu, X., Liao, X., Zhou, T., He, Z., & Hu, H.
“Vibration-based identification of lubrication starved bearing conditions.”
Measurement, 226:114156 (2024)
[https://doi.org/10.1016/j.measurement.2024.114156](https://doi.org/10.1016/j.measurement.2024.114156)

Randall, R. B.
*Vibration-based Condition Monitoring: Industrial, Aerospace and Automotive Applications.*
John Wiley & Sons (2011)
[https://doi.org/10.1002/9780470977668](https://doi.org/10.1002/9780470977668)

---

# Final Assessment

This project demonstrates:

* Detection of dataset bias (rpm regime)
* Domain-shift mitigation (sampling-rate differences)
* Controlled statistical benchmarking
* Physically grounded feature engineering
* Interpretability-driven model selection
* Production-level integration

The outcome is not simply a classifier, but a **robust, physically interpretable, and deployable vibration diagnostic pipeline**.