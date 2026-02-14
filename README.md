# Condition Monitoring – Frequency-domain Carpet Detection and Looseness Classification

## Overview

This repository implements a vibration-based condition monitoring pipeline focused on:

- **Part 1:** Frequency-domain carpet detection  
- **Part 2:** Looseness classification using metadata and machine learning  

The approach combines physical signal modeling (DSP) and supervised learning, with emphasis on:

- Sampling validation  
- Reproducible spectral analysis  
- Structured data validation via Pydantic schemas  
- Clean project organization  

The goal is to ensure that signal processing decisions are physically grounded before applying machine learning techniques.

---

## Repository Structure

```

tractian-condition-monitoring/
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

## Environment Setup

Create and activate a virtual environment (Windows example):

```

python -m venv tractianenv
tractianenv\Scripts\activate

```

Install the project in editable mode:

```

python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"

```

This ensures:

- Proper package import from `src/`
- Reproducible dependency management
- Consistent spectral computation between notebooks and modules

---

## Part 1 – Carpet Detection Strategy

The carpet phenomenon is modeled as:

- Broadband energy elevation
- Persistent increase in spectral noise floor
- Occurrence in the frequency region above 1000 Hz

Methodological steps:

1. Validate time axis consistency and sampling rate.
2. Estimate PSD using Welch's method.
3. Restrict analysis to frequencies ≥ 1000 Hz.
4. Estimate a robust noise floor (rolling median in dB).
5. Detect regions where PSD exceeds the baseline by a relative threshold.
6. Enforce minimum bandwidth and non-overlapping region constraints.

This approach prioritizes physical interpretability and robustness over heuristic peak detection.

---

## Part 2 – Looseness Classification

The second stage focuses on supervised classification using:

- Metadata-driven axis mapping (horizontal, vertical, axial)
- Time-domain and frequency-domain feature extraction
- Baseline model comparison
- Structured prediction interface via Pydantic models

The objective is to combine domain knowledge and statistical modeling while maintaining reproducibility.

---

## Design Principles

- Separate physical signal validation from ML modeling.
- Avoid absolute thresholds when signal energy varies.
- Use relative spectral elevation (dB above baseline) for carpet detection.
- Keep the repository deployable and modular.
- Maintain reproducibility through controlled PSD parameters.

---

## Notes

- Raw datasets are not versioned.
- External case materials and reference papers are excluded from the repository.
- All signal processing assumptions are validated in the initial notebook before modeling.

---

## Status

Current progress:

- Signal validation and PSD sanity checks implemented.
- Structured schemas and loaders implemented.
- Baseline carpet detection model under development.