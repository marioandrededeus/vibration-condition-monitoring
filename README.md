
# Vibration Condition Monitoring – Frequency-Domain Carpet Detection and Looseness Classification

## Overview

This repository implements a vibration-based condition monitoring pipeline focused on:

- **Part 1:** Frequency-domain carpet detection  
- **Part 2:** Looseness classification using metadata and machine learning  

The approach combines physical signal modeling (DSP) and supervised learning, with emphasis on reproducibility, structured validation, and physically interpretable decisions.

Before implementing detection logic, signal consistency and spectral assumptions are validated to ensure that modeling decisions are grounded in signal processing fundamentals.

---

## Scientific Motivation

## Scientific Motivation

The broadband detection methodology adopted in this repository is conceptually inspired by prior research on lubrication-related bearing faults.

Xu et al., *“Vibration-based identification of lubrication starved bearing conditions”* (Mechanical Systems and Signal Processing), demonstrate that starved lubrication leads to increased broadband energy and elevated RMS levels. However, their study also shows that RMS and global vibration levels vary significantly with rotational speed, making purely time-domain metrics unreliable when speed is not controlled.

In other words:

- Lubrication degradation increases broadband spectral energy.
- Rotational speed variations can produce similar increases in global vibration levels.
- Without speed information, distinguishing operational effects from degradation becomes less precise if relying solely on time-domain statistics.

Because rotational speed is not provided in this dataset, the methodology prioritizes:

- Frequency-domain analysis,
- Relative noise-floor modeling in dB,
- Broadband elevation detection above 1000 Hz,
- Robust baseline estimation rather than absolute energy thresholds.

This approach transfers the conceptual insight from Xu et al., that broadband spectral behavior is more robust than raw RMS, while acknowledging the limitation imposed by missing operational variables. 

---

## Repository Structure

```

vibration-condition-monitoring/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── src/
│   └── tractian_cm/
│       ├── io/          # Data loading and validation
│       ├── dsp/         # Spectral analysis utilities
│       ├── part1/       # Carpet detection logic
│       └── part2/       # Looseness classification pipeline
│
├── notebooks/
│   ├── 01_part1_sanity_and_psd.ipynb
│   └── 02_part2_pipeline.ipynb (to be implemented)
│
└── data/                # Local datasets (not versioned)

```

---

## Part 1 – Carpet Detection Strategy

Carpet regions are modeled as:

- Broadband spectral elevation  
- Persistent increase in spectral noise floor  
- Occurrence in the frequency region above 1000 Hz  

Detection pipeline:

1. Validate sampling consistency.
2. Estimate PSD using Welch’s method.
3. Restrict analysis to frequencies ≥ 1000 Hz.
4. Estimate robust spectral baseline using rolling median in dB.
5. Detect regions exceeding baseline by a relative threshold.
6. Enforce minimum bandwidth constraints.

This approach prioritizes interpretability and physical consistency over heuristic peak detection.

---

## Part 2 – Looseness Classification

The second stage focuses on supervised classification using:

- Metadata-driven axis mapping  
- Time-domain and frequency-domain feature extraction  
- Baseline model comparison  
- Structured prediction interfaces via Pydantic  

The objective is to combine domain knowledge and statistical modeling while maintaining reproducibility.

---

## Design Principles

- Separate physical signal validation from machine learning.
- Avoid absolute thresholds when signal energy varies across samples.
- Use relative spectral elevation for broadband detection.
- Maintain modular and deployable project structure.
- Centralize spectral computation for consistency across notebooks and modules.

---

## Environment Setup

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

## References

Xu, X., Liao, X., Zhou, T., He, Z., & Hu, H.  
“Vibration-based identification of lubrication starved bearing conditions.”  
Measurement, 226:114156 (2024)  
https://doi.org/10.1016/j.measurement.2024.114156

This work provides conceptual grounding for frequency-domain broadband analysis as a robust alternative to purely time-domain metrics.