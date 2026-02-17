## Sampling Frequency and Window Length Mismatch

Sanity checks revealed structural differences between training and test sets:

| Split | Samples | fs (Hz) | Duration (s) | Δf (Hz) |
|-------|---------|---------|--------------|---------|
| Train | 2048    | ~4 kHz  | ~0.5 s       | ~1.95   |
| Test  | 16384   | ~8 kHz  | ~2.0 s       | ~0.48   |

This indicates a domain shift in:

- Sampling frequency
- Spectral resolution
- Window duration
- Nyquist limit

### Implications

High-frequency broadband features are not directly comparable across splits.

### Mitigation

Feature engineering prioritizes:

- Rotation-synchronous harmonics (rpm-based)
- Low-frequency band ratios
- Relative amplitude metrics
- Resolution-aware harmonic detection

This ensures cross-dataset robustness.
