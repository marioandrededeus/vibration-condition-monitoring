# Part 2 — Structural Looseness Prediction

## Goal
Build a model to detect **structural looseness** from raw vibration signals and run it on `test_data`, following the provided template (`Wave` + `LoosenessModel.predict` and optional `.score`).

## Inputs (from the case)
- `data/`: CSV signals with columns:
  - `X-Axis` = time (s)
  - `Ch1 Y-Axis`, `Ch2 Y-Axis`, `Ch3 Y-Axis` = acceleration (g) for axes X, Y, Z
- `part_3_metadata.csv`: labeled subset metadata
  - `condition` (healthy / structural_looseness)
  - `sensor_id`
  - `rpm`
  - `orientation`: mapping from `axisX/axisY/axisZ` to `horizontal/vertical/axial`
- `test_data/`: CSV signals with columns `t, x, y, z`
- `test_metadata.csv`: `rpm`, `asset`, and `orientation` for each `sample_id`

## Project phases
### Phase 1 — Data preparation
Deliverables:
- Metadata parsing and validation (`orientation` parsing + constraints)
- Unified **samples index** with:
  - split: `train_labeled`, `train_unlabeled`, `test`
  - filepaths + metadata fields
- Sanity checks (metadata-level; signal-level checks in next commit)

### Phase 2 — EDA (univariate + bivariate)
- Class balance (healthy vs looseness)
- Distributions and potential confounders (e.g., rpm, sensor_id)
- Visualization of candidate features (time/frequency)
- Insightful plots supporting feature selection

### Phase 3 — Modeling + validation
- Baseline heuristics / physics-inspired rules
- Classic ML (LogReg, RandomForest) with/without balancing
- Optionally deep models (TensorFlow/Keras) if justified by data volume
- Strong validation: Group CV (by sensor_id), confusion matrix, ROC AUC, PR AUC, threshold calibration
- Implement `LoosenessModel` template and integrate into the Part 1 webapp

## Non-negotiable constraints
- `LoosenessModel` **must receive waves already mapped to**:
  - horizontal (`wave_hor`), vertical (`wave_ver`), axial (`wave_axi`)
- Schema normalization happens **before** the model (data vs test_data column differences).
- One single feature pipeline reused by EDA, training, and webapp.
