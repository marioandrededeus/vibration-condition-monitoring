
<div style="text-align: center; margin-top: 120px;">

# VIBRATION-BASED CONDITION MONITORING  
## Broadband Carpet Detection and Structural Looseness Diagnosis  

<br><br>

**Technical Assessment Submission**  
Data Science – Condition Monitoring  

<br><br><br>

**Author**  
Mario André de Deus  

<br>

**Date**  
March 2026  

</div>

\newpage

# Table of Contents

1. Introduction  
2. Problem Definition  
3. Part 1 — Broadband Carpet Detection  
4. Part 2 — Structural Looseness Diagnosis  
5. Key Insights and Engineering Decisions  
6. Conclusion  
7. References  

---

# 1. Introduction

This report addresses two vibration-based diagnostic tasks:

1. Detection of broadband “carpet” regions in frequency spectra.  
2. Prediction of structural looseness using tri-axial vibration signals.  

The objective extends beyond classification performance. The solution emphasizes:

- Physical interpretability  
- Explicit bias investigation  
- Robustness to sampling-rate domain shift  
- Deterministic deployment readiness  

The methodology integrates signal processing, statistical validation, and physics-informed modeling to ensure alignment with both the challenge requirements and industrial vibration-analysis practice.

---

# 2. Problem Definition

## Part 1 Requirements

- Detect carpet regions above 1000 Hz  
- Allow multiple disjoint regions  
- Prevent overlapping intervals  
- Plot spectral regions  
- Propose severity metric  
- Identify worst sample  

## Part 2 Requirements

- Extract relevant features  
- Develop looseness classifier  
- Respect provided Pydantic template  
- Accept H/V/A inputs  
- Predict test dataset  
- (Bonus) Implement severity score  

---

# 3. Part 1 — Broadband Carpet Detection

## 3.1 Signal Integrity Validation

Before spectral analysis, waveform integrity was verified:

- Uniform time steps  
- No missing values  
- Stable acquisition length  

<p align="center">
  <img src="https://raw.githubusercontent.com/marioandrededeus/vibration-condition-monitoring/main/reports/part1/waveform_file11d8.png" width="400">
</p> 
<p align="center"><em>Figure 1 — Representative waveform segment.</em></p>



---

## 3.2 Spectral Representation Strategy

Welch PSD was initially used for exploratory diagnostics.  
However, PSD smooths fine spectral structure and may obscure peak-spacing irregularity.

Final metric uses FFT magnitude to preserve peak microstructure.
<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure2_psd_example.png?raw=true" width="400">
</p> 
<p align="center"><em>Figure 2 — PSD example (≥1000 Hz)</em></p>

---

## 3.3 Carpet Definition and Operationalization

Carpet regions consist of spectral peaks randomly close to each other,  
in contrast to regularly spaced harmonic structures.

Constraints:

- Frequency ≥ 1000 Hz  
- Multiple disjoint regions allowed  
- No overlapping intervals  

---

## 3.4 Peak Spacing Irregularity

For consecutive peaks:

$$
\Delta f_i = f_{i+1} - f_i
$$

Irregularity quantified via coefficient of variation:

$$
CV = \frac{\sigma(\Delta f)}{\mu(\Delta f)}
$$

Low CV → harmonic regularity  
High CV → carpet-like clustering  

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure3_psd_peak_spacing.png?raw=true?raw=true" width="600">
</p> 
<p align="center"><em>Figure 3 — Regular vs irregular peak spacing</em></p>

---

## 3.5 Dynamic Clustering

Peaks are grouped when:

$$
\Delta f < gap\_max\_hz
$$

Clusters define candidate carpet regions.

Properties:

- Disjoint intervals  
- No overlap  
- Respect frequency constraint (≥1000 Hz)  

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure4_clusterFFT.png?raw=true?raw=true?raw=true" width="600">
</p> 
<p align="center"><em>Figure 4 — Cluster visualization on FFT</em></p>

---

## 3.6 Carpet Severity Metric

Severity increases with:

- Peak density  
- Spacing irregularity  
- Absence of dominant resonance  

Per-sample severity = maximum cluster score.

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure5_severity_rank.png?raw=true?raw=true?raw=true?raw=true" width="600">
</p> 
<p align="center"><em>Figure 5 — Severity ranking across sample</em></p>

---

## 3.7 Robustness Analysis

Clustering tolerance tested at:

- 25 Hz  
- 35 Hz  
- 50 Hz  

Worst sample remained invariant across configurations.

---

## 3.8 Final Result

The sample:

`3186c48d-fc24-5300-910a-6d0bafdd87ea.csv`
<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure6_worst_sample.png?raw=true?raw=true?raw=true?raw=true?raw=true" width="600">
</p> 
<p align="center"><em>Figure 6 — Sample with the worst carpet symptom</em></p>

presents the strongest carpet symptom.

Result is stable under tolerance variation and consistent with spectral observations.

---

# 4. Part 2 — Structural Looseness Diagnosis

## 4.1 Physical Hypothesis

Structural looseness modifies:

- Global stiffness  
- Harmonic response  
- Nonlinear contact behavior  

Expected signatures:

- Amplified 1×, 2×, 3× harmonics  
- Increased crest factor  

---

## 4.2 Domain Shift Investigation

Sampling difference observed:

| Split | Sampling Rate | Duration |
|--------|--------------|----------|
| Train | ~4 kHz | ~0.5 s |
| Test | ~8 kHz | ~2.0 s |

Mitigation:

- Harmonics anchored to rpm  
- Resolution-aware tolerance  
- Ratio-based features  

---

## 4.3 Feature Engineering

Extracted features:

- 1×, 2×, 3× harmonic amplitudes  
- 2×/1× and 3×/1× ratios  
- Crest factor  
- Low-frequency band metrics  

![Figure 7 — Harmonic ratio separation](reports/figures/part2_harmonic_scatter.png)

---

## 4.4 Structural Bias Detection

Sensor distribution balanced.

RPM exhibited regime artifact:

- All 1595 rpm samples labeled looseness.

Model evaluation performed under:

- Full dataset  
- rpm = 1598 only  

---

## 4.5 Baseline Model — 2×/1× Threshold

The single-feature harmonic threshold achieved strong recall.

![Figure 8 — Baseline con]()

<!--stackedit_data:
eyJoaXN0b3J5IjpbNDQ3ODU5NzIwXX0=
-->