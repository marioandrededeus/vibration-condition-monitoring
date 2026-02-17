# Vibration Condition Monitoring

## Broadband Carpet Detection & Structural Looseness Diagnosis

---

# Executive Summary

This repository implements a two-stage vibration diagnostic system:

* **Part 1:** Broadband carpet detection in the frequency domain
* **Part 2:** Structural looseness diagnosis using harmonic behavior

The solution prioritizes:

* Physics-informed features over black-box modeling
* Robustness under sampling-rate domain shift
* Explicit bias investigation (rpm regime)
* Deterministic and deployable architecture

Although tree-based models achieved slightly higher ROC-AUC, a multi-feature harmonic heuristic was selected for deployment due to interpretability, robustness, and lower operational complexity.

---

# Solution Architecture

```
vibration-condition-monitoring/
│
├── streamlit_app.py
├── requirements.txt
├── pyproject.toml
│
├── notebooks/
│   ├── 01_part1_sanity_and_psd.ipynb
│   └── 01_part2_pipeline.ipynb
│
├── reports/
│   └── report.md
│
└── src/tractian_cm/
    ├── io/
    ├── dsp/
    ├── part1/
    ├── part2/
    └── pipeline/
```

The Streamlit application uses the same inference pipeline validated during model development.

---

# Part 1 – Carpet Detection

### Objective

Detect broadband spectral elevation (“carpet”) above 1000 Hz without rotational speed information.

### Method

1. Validate sampling consistency
2. Compute PSD using Welch’s method
3. Restrict to frequencies ≥ 1000 Hz
4. Estimate rolling-median spectral baseline (dB)
5. Detect sustained elevation above relative threshold
6. Enforce minimum bandwidth

### Rationale

* RMS alone is unreliable without speed control.
* Relative spectral elevation is more robust to operating variability.
* Rolling baseline reduces sensitivity to narrow peaks.

---

# Part 2 – Structural Looseness Diagnosis

### Physical Expectation

Structural looseness alters:

* System stiffness
* Harmonic response at 1×, 2×, 3×
* Nonlinear contact → increased crest factor

### Domain Shift Identified

Train and test sets differ in:

* Sampling rate
* Signal duration
* Spectral resolution

To prevent overfitting:

* Harmonics are anchored to rpm
* Frequency tolerance adapts to bin width
* Ratios are used instead of absolute amplitudes

---

# Benchmarking

Three approaches evaluated:

1. 2×/1× single-threshold heuristic
2. Logistic Regression
3. Random Forest (RandomizedSearchCV)

Validation protocol:

* Stratified 70/30 split
* 5-fold CV
* Threshold tuning on training folds only
* Strict leakage control

---

# Final Model

A multi-feature physics-informed heuristic combining:

* 2×/1× harmonic ratio
* 3×/1× harmonic ratio
* Crest factor
* Sigmoid-based soft scoring

Selected for deployment due to:

* Comparable F1 performance
* Full interpretability
* Deterministic behavior
* Lower computational cost
* Ease of validation in industrial settings

Random Forest remains as benchmark reference.

---

# Template Compliance

The required model interfaces were implemented using `pydantic.BaseModel`:

* `Model.predict(sample: np.ndarray) -> int`
* `LoosenessModel.predict(sample: np.ndarray, rpm: float) -> int`
* `LoosenessModel.score(sample: np.ndarray, rpm: float) -> float`

These wrap the validated inference pipeline used in deployment.

---

# Reproducing Results

## 1. Clone Repository

```
git clone https://github.com/marioandrededeus/vibration-condition-monitoring.git
cd vibration-condition-monitoring
```

## 2. Create Environment

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

Pinned versions ensure deterministic builds.

---

## 4. Run Streamlit Application

```
streamlit run streamlit_app.py
```

---

## 5. Reproduce Benchmarks

* Part 1 exploratory validation:
  `notebooks/01_part1_sanity_and_psd.ipynb`

* Part 2 modeling and evaluation:
  `notebooks/01_part2_pipeline.ipynb`

All reported metrics originate from these notebooks.

---

# Outputs Included

The repository contains:

* Model comparison metrics
* Harmonic interaction analysis
* Domain-shift investigation
* Short technical report (`reports/report.md`)

---

# Design Principles

* Physics before ML
* Investigate bias before optimizing
* Avoid leakage
* Prefer harmonic-anchored features
* Maintain training–deployment parity
* Ensure deterministic environments

---

# References

Xu et al. (2024) – Lubrication starvation and broadband vibration
Randall (2011) – Vibration-based Condition Monitoring

---

# Final Statement

This solution delivers a bias-aware, domain-shift-robust, and production-ready vibration diagnostic pipeline, balancing statistical validation with physical interpretability.