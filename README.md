# Vibration Condition Monitoring

## Frequency-Domain Carpet Detection and Structural Looseness Diagnosis

---

# Overview

This repository implements a vibration-based condition monitoring system structured as a two-stage analytical pipeline:

* **Part 1 – Broadband Carpet Detection** (frequency-domain)
* **Part 2 – Structural Looseness Diagnosis** (harmonic-based, rotation-synchronous)

The goal of this project is not merely to train a classifier, but to design a:

> **Physically grounded, bias-aware, resolution-robust, and deployable diagnostic system.**

The solution integrates:

* Digital Signal Processing (FFT / PSD / harmonic extraction)
* Physics-informed feature engineering
* Sampling-rate domain-shift mitigation
* RPM-bias investigation
* Controlled model benchmarking
* Interpretable heuristic modeling
* Production-ready Streamlit deployment

---

## 🌐 Live Application

👉 **Access the deployed web application:**
[https://vibration-monitoring.streamlit.app/](https://vibration-monitoring.streamlit.app/)

The web application enables interactive execution of:

### Part 1 – Carpet Detector

* PSD visualization
* High-frequency broadband detection (≥ 1000 Hz)
* Relative spectral elevation (dB)
* Rolling-median baseline estimation
* Industrial-style traffic-light indicator

### Part 2 – Looseness Detector

* Tri-axial vibration upload (single CSV or multi-file)
* RPM input
* Automatic harmonic extraction (1×, 2×, 3×)
* Crest factor computation
* Feature preview before inference
* Traffic-light decision system aligned with Part 1
* Structured output table (industrial-style diagnostic output)

The application demonstrates:

* Training–deployment parity
* Modular inference layer
* Reproducible feature extraction
* Clean packaging (src layout)
* Cloud-ready architecture

---

## 📄 Technical Case Report (Detailed Analysis)

A full technical report including:

* Exploratory data analysis
* RPM bias detection
* Sampling-rate domain shift mitigation
* Harmonic interaction analysis
* Benchmark comparison (heuristic vs Logistic Regression vs Random Forest)
* Model selection rationale

is available here:

👉 [https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/Tech_Case_Mario_Deus.md](https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/Tech_Case_Mario_Deus.md)

This document provides complete transparency into the analytical decisions and modeling rigor behind the deployed solution.

---

# Scientific Motivation

## Part 1 – Broadband Carpet Detection

Broadband spectral elevation is often associated with lubrication degradation and distributed contact phenomena.

Xu et al. (2024) demonstrate that lubrication starvation increases high-frequency spectral energy. However, RMS amplitude strongly depends on rotational speed, making time-domain detection unreliable without RPM metadata.

Since RPM is not provided in Part 1:

* Detection is performed in the **frequency domain**
* Analysis focuses on frequencies ≥ 1000 Hz
* Relative dB elevation is used instead of raw RMS
* A rolling-median baseline is estimated
* Carpet regions are defined by persistent broadband energy rise

This ensures robustness under missing operational variables.

---

## Part 2 – Structural Looseness Diagnosis

Structural looseness affects stiffness and introduces nonlinear mechanical behavior.

Expected vibration signatures include:

* Elevated amplitude at 1× rotational frequency
* Harmonic amplification at 2× and 3×
* Increased crest factor
* Nonlinear response characteristics

The deployed model combines:

* 2× / 1× amplitude ratio
* 3× / 1× amplitude ratio
* Crest factor
* Soft sigmoid aggregation

Unlike black-box ML models, the final solution preserves direct physical interpretability.

---

# Dataset Considerations

## Sampling-Rate Domain Shift

The dataset exhibits different sampling characteristics:

| Split | Samples | Approx. fs | Duration |
| ----- | ------- | ---------- | -------- |
| Train | 2048    | ~4 kHz     | ~0.5 s   |
| Test  | 16384   | ~8 kHz     | ~2.0 s   |

This introduces domain shift in:

* Frequency resolution (Δf)
* Nyquist frequency
* Time-window duration

Mitigation strategy:

* Frequency-anchored harmonic detection
* Resolution-aware tolerance bands
* Ratio-based features (dimensionless)
* Avoidance of raw amplitude thresholds

---

## RPM Bias Detection

Exploratory analysis revealed:

* All samples at rpm = 1595 were labeled looseness.

To validate robustness:

* Models were evaluated on the full dataset
* A restricted evaluation at rpm = 1598 was conducted

Results confirmed that looseness behavior is driven by harmonic interaction and nonlinear amplification rather than a trivial RPM threshold.

---

# Model Benchmarking

Three approaches were evaluated:

1. Single-feature physics heuristic (2×/1× threshold)
2. Logistic Regression
3. Random Forest (RandomizedSearchCV)

Validation protocol:

* Stratified 70/30 split
* 5-fold cross-validation
* Threshold tuning on training folds only
* Strict leakage prevention
* Final evaluation on unseen holdout

Although Random Forest slightly improved ROC-AUC, performance gains in F1 were marginal.

Given the objectives of interpretability, deployment simplicity, and physical coherence, the refined physics-informed heuristic was selected for production.

---

# Repository Structure

```
vibration-condition-monitoring/
│
├── streamlit_app.py
├── pages/
│   ├── 01_Carpet.py
│   └── 02_Looseness.py
│
├── src/
│   └── tractian_cm/
│       ├── __init__.py
│       │
│       ├── part1/
│       │   ├── carpet_model.py
│       │   └── ...
│       │
│       ├── part2/
│       │   ├── model.py
│       │   └── ...
│       │
│       └── app/
│           ├── ui_shared.py
│           └── pages/
│               ├── carpet.py
│               └── looseness.py
│
├── notebooks/
│   ├── 01_part1_sanity_and_psd.ipynb
│   ├── 02_part1_fft_carpet_metric.ipynb
│   └── 01_part2_pipeline.ipynb
│
├── reports/
│   ├── part1/
│   └── part2/
│
├── Tech_Case_Mario_Deus.md
├── requirements.txt
├── pyproject.toml
└── runtime.txt
```

---

# Environment Setup

## Create Virtual Environment

```bash
python -m venv tractianenv
tractianenv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

> The repository uses a `src` layout.
> The `requirements.txt` includes `-e .` to ensure package installation.

## Run Locally

```bash
streamlit run streamlit_app.py
```

---

# Deployment

The application is deployed on **Streamlit Community Cloud**:

[https://vibration-monitoring.streamlit.app/](https://vibration-monitoring.streamlit.app/)

Deployment ensures:

* Reproducible environment
* Deterministic dependency versions
* Public interactive demonstration
* Cloud-ready packaging

---

# References

Xu, X., Liao, X., Zhou, T., He, Z., & Hu, H.
“Vibration-based identification of lubrication starved bearing conditions.”
Measurement, 226:114156 (2024)

Randall, R. B.
*Vibration-based Condition Monitoring: Industrial, Aerospace and Automotive Applications.*
John Wiley & Sons (2011)

---

# Final Assessment

This project demonstrates:

* Detection of dataset bias (RPM regime)
* Domain-shift mitigation (sampling-rate differences)
* Physically grounded feature engineering
* Controlled benchmarking against ML models
* Interpretability-driven model selection
* Production-level Streamlit deployment

The result is not merely a classifier, but a:

> **Robust, interpretable, and deployable vibration diagnostic pipeline.**
