
<div align="center" style="margin-top:120px;">

# VIBRATION-BASED CONDITION MONITORING  
## Broadband Carpet Detection and Structural Looseness Diagnosis  

<br><br>

**Technical Case**  
Condition Monitoring Assessment  

<br><br>

**Author**  
Mario André de Deus  

<br>

March 2026  

</div>



# Abstract

This technical case study presents a two-stage vibration analysis framework designed to address broadband carpet detection and structural looseness diagnosis.

Rather than approaching the challenge purely as a classification task, the solution emphasizes:

- Physical interpretability  
- Bias investigation prior to modeling  
- Robustness under domain shift  
- Deterministic deployment design  

The final system integrates frequency-domain microstructure analysis, harmonic feature engineering, and physics-informed scoring to produce a production-ready diagnostic pipeline.

---

# 1. Introduction — Framing the Engineering Problem

Industrial vibration diagnostics is fundamentally an exercise in physical reasoning.

The challenge consists of two tasks:

1. Detect broadband “carpet” regions in frequency spectra.
2. Diagnose structural looseness using tri-axial vibration signals.

Although both tasks operate in the frequency domain, they reflect distinct physical mechanisms:

- Carpet behavior reflects spectral microstructure irregularity.
- Structural looseness reflects stiffness alteration and harmonic amplification.

The central engineering objective was:

> To design metrics that reflect physical phenomena rather than statistical coincidence.

---

# 2. Part 1 — Broadband Carpet Detection

---

## 2.1 Signal Integrity Validation

Before implementing any spectral metric, signal integrity was verified:

- Uniform time sampling  
- Stable acquisition duration  
- Absence of corrupted values  

This step ensures that spectral irregularities are not acquisition artifacts.

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure1_waveform.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="400">
</p> 
<p align="center"><em>Figure 1 — Representative waveform segment</em></p>

---

## 2.2 Spectral Representation Strategy

Welch PSD was initially used for exploratory diagnostics.

However, PSD smooths peak microstructure, which is central to the carpet definition.

The final detection strategy relies on FFT magnitude, which preserves exact peak positions.

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure2_psdfft_example.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="4700">
</p> 
<p align="center"><em>Figure 2 — PSD (≥1000 Hz focus)FFT - Spectral representation</em></p>

---

## 2.3 Carpet Definition and Operationalization

Carpet regions consist of spectral peaks randomly close to each other.

In contrast, harmonic systems exhibit regular peak spacing.

Operational constraints:

- Frequency ≥ 1000 Hz  
- Multiple disjoint regions allowed  
- No overlapping intervals  

---

## 2.4 Peak Spacing Irregularity

For consecutive FFT peaks:

$$
\Delta f_i = f_{i+1} - f_i
$$

Regular harmonic structures produce near-constant spacing.

Carpet structures produce high variability.

Irregularity is quantified via:

$$
CV = \frac{\sigma(\Delta f)}{\mu(\Delta f)}
$$

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure3_psd_peak_spacing.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 3 — Regular vs Irregular Peak Spacing Comparison</em></p>

---

## 2.5 Dynamic Clustering and Region Extraction

Rather than predefining static frequency bands, a dynamic clustering rule was implemented.

Peaks are grouped when:

$$
\Delta f < gap\_max\_hz
$$

This ensures:

- Disjoint regions  
- No overlap  
- Physical interpretability  

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure4_clusterFFT.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 4 — Clustered Regions Overlaid on FFT</em></p>

---

## 2.6 Carpet Severity Metric

Severity increases with:

- Peak density  
- Spacing irregularity  
- Absence of dominant single resonance  

Per-sample severity is defined as the maximum cluster score.

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure5_severity_rank.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 5 — Carpet Severity Ranking Across Samples</em></p>

---

## 2.7 Robustness Analysis

Clustering tolerance (`gap_max_hz`) was evaluated at:

- 25 Hz  
- 35 Hz  
- 50 Hz  

The worst-case sample remained invariant across parameter variation.

This indicates ranking stability.

---

## 2.8 Final Result

The sample:

`3186c48d-fc24-5300-910a-6d0bafdd87ea.csv`
<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part1/figure6_worst_sample.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 6 —  Sensitivity Analysis Across Clustering Tolerances</em></p>

consistently presents the strongest carpet symptom.

The result is stable and consistent with spectral inspection.

---

# 3. Part 2 — Structural Looseness Diagnosis

---

## 3.1 Structural Looseness — Physical Framing and Modeling Strategy

Structural looseness alters system stiffness.

Expected vibration signatures include:

- Amplified 1× rotational frequency  
- Enhanced 2× and 3× harmonics  
- Increased crest factor due to nonlinear contact  

Unlike carpet behavior, looseness modifies harmonic structure rather than broadband high-frequency energy.

---

## 3.2 Sampling-Rate Domain Shift

A structural difference was observed between train and test splits:

| Split | Sampling Rate | Duration |
|-------|--------------|----------|
| Train | ~4 kHz | ~0.5 s |
| Test | ~8 kHz | ~2.0 s |

Implications:

- Different spectral resolution  
- Different Nyquist frequency  
- Different time window length  

Mitigation:

- Harmonics anchored to rpm  
- Resolution-aware tolerances  
- Ratio-based features  

---

## 3.3 Feature Engineering Anchored in Physics

Extracted features:

- 1×, 2×, 3× harmonic amplitudes  
- 2×/1× and 3×/1× ratios  
- Crest factor  
- Low-frequency band metrics  

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part2/figure7_harmonics.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 7 —  Harmonic amplification</em></p>

---

## 3.4 Structural Bias — Sensor and RPM

Sensor distribution was balanced.
<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part2/figure8_sensor_bias.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 8 —  Sensor bias</em></p>

However, rpm revealed structural correlation:

- All 1595 rpm samples labeled looseness.
- 1598 rpm contained both classes.

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part2/figure9_rpm_bias.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="400">
</p> 
<p align="center"><em>Figure 9 —  RPM bias</em></p>
To prevent regime-based shortcut learning, evaluation was performed in two scenarios:

- Full dataset  
- rpm = 1598 only  

---

## 3.5 Baseline — Single Harmonic Threshold

A simple 2×/1× threshold achieved strong recall.

This confirms harmonic amplification as the dominant looseness signature.

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part2/figure10_baseline_treshold.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 10 —  Baseline treshold</em></p>

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part2/figure11_baseline_cm.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 11 —  Baseline confusion matrices</em></p>

---

## 3.6 Logistic Regression and Random Forest Benchmarking

Supervised models were evaluated using:

- Stratified splits  
- Cross-validation  
- Holdout validation  

Performance gains over baseline were moderate.

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part2/figure12_reglog_cm.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 12 —  Logistic regression confusion matrices</em></p>

---

## 3.7 Physics-Informed Heuristic v2

A soft-scoring model combining:

- 2×/1× harmonic amplification  
- 3×/1× nonlinear amplification  
- Crest factor  

was implemented.

Severity score normalized to [0,1].

This approach achieved performance comparable to Random Forest while offering:

- Interpretability  
- Lower complexity  
- Deterministic inference  

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part2/figure13_models_comparison.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 13 —  Models comparison</em></p>

---

## 3.8 Final Inference Strategy

The deployed heuristic was tuned using the rpm=1598 subset to reduce regime artifact risk.

This ensures:

- Physical feature reliance  
- Reduced shortcut learning  
- Better operational robustness  

<p align="center">
  <img src="https://github.com/marioandrededeus/vibration-condition-monitoring/blob/main/reports/part2/figure13_models_comparison.png?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true?raw=true" width="700">
</p> 
<p align="center"><em>Figure 14 —  Models comparison</em></p>

---

# 4. Engineering Insights

Key insights from the investigation:

1. Spectral structure is more informative than broadband energy for carpet detection.
2. Dataset bias must be investigated before model selection.
3. Harmonic amplification is the dominant looseness indicator.
4. Marginal ROC gains do not justify black-box deployment.
5. Physics-informed heuristics can match ML performance with greater transparency.

---

# 5. Conclusion

The final solution integrates:

- FFT-based microstructure analysis  
- Dynamic spectral clustering  
- Harmonic feature engineering  
- Bias-aware modeling  
- Physics-informed scoring  

Part 1 delivers a deterministic and robust carpet detection metric.

Part 2 delivers an interpretable looseness classifier grounded in physical vibration behavior.

The outcome is not merely predictive — it is structurally diagnostic and deployment-ready.

---

# References

Xu, X., Liao, X., Zhou, T., He, Z., & Hu, H. (2024).  
Vibration-based identification of lubrication starved bearing conditions. *Measurement*.

Randall, R. B. (2011).  
Vibration-based Condition Monitoring: Industrial, Aerospace and Automotive Applications. Wiley.

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1NTY4NTUyMDcsLTg5ODE4NjE2OSwtNT
QzODM5NjU0LDc0MzIxODc2OV19
-->