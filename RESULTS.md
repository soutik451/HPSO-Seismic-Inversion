# Results on the synthetic data set

All numbers come from `python make_synthetic_data.py`, `python 01_synthetic_test.py` and `python 02_section_inversion.py`; every figure and table is in `results/`. CC is the Pearson correlation with the true impedance. "0–80 Hz" means the true impedance filtered to the seismic band (0–80 Hz).

## Synthetic data set (`synthetic_data/`)

* **Section:** 201 samples × 101 traces, 2 ms sampling, 1700–2100 ms. The well is at trace 50 (CDP 1051).
* **Impedance model:** blocky layered sequence with a compaction trend, a low-impedance (porous) interval and fine-scale log detail.
* **Structure:** an anticline, a regional dip, thickening of layers, and about 3 % lateral property change.
* **Seismic:** 25 Hz Ricker wavelet rotated by 20°, plus 15 % Gaussian noise.
* **Well-tied trace:** the noise-free trace at the well plus an independent 10 % noise realisation.

## 1. Single-trace noise test (`results/synthetic_test/`)

Setup:
* **True model:** the well impedance.
* **Wavelet:** Ricker, 30 Hz.
* **Initial model:** the 10 Hz low-pass of the log.
* **Repeats:** 5 noise realisations and PSO seeds per noise level.

Results (mean ± std):

| noise | initial model CC | PSO CC | HPSO CC | HPSO CC (0–80 Hz) |
|---|---|---|---|---|
| 0 % | 0.78 | 0.71 ± 0.07 | **0.98 ± 0.00** | 1.00 |
| 10 % | 0.78 | 0.73 ± 0.09 | **0.97 ± 0.00** | 0.99 |
| 20 % | 0.78 | 0.72 ± 0.05 | **0.95 ± 0.01** | 0.97 |
| 30 % | 0.78 | 0.74 ± 0.06 | **0.93 ± 0.01** | 0.95 |

HPSO stays accurate as noise increases. Stand-alone PSO (300 iterations) lowers the seismic misfit but does not converge to an accurate model in 201 dimensions; the quasi-Newton step after PSO is what gives the accuracy.

## 2. Section inversion (`results/section_inversion/`)

**Wavelet:** Ricker 25 Hz with 20° phase, recovering the true wavelet. Its well tie is CC = 0.995.

**λ:** noise level from the tie 0.009, so the discrepancy principle gives λ = 3.1.

### At the well

| result | CC | R² | RMSE | CC (0–80 Hz) | R² (0–80 Hz) |
|---|---|---|---|---|---|
| Initial model (10 Hz) | 0.784 | 0.613 | 487 | 0.807 | 0.650 |
| After PSO only | 0.740 | 0.431 | 590 | 0.764 | 0.463 |
| **HPSO, section trace (case A)** | **0.959** | **0.919** | **223** | **0.985** | **0.968** |
| HPSO, well-tied trace (case B) | 0.956 | 0.912 | 231 | 0.982 | 0.964 |

### Whole section vs the true model

| result | CC | R² | mean trace CC | lowest trace CC |
|---|---|---|---|---|
| Initial model | 0.704 | 0.494 | 0.704 | 0.628 |
| After PSO | 0.702 | 0.374 | 0.707 | 0.569 |
| **HPSO** | **0.923** | **0.853** | **0.924** | **0.887** |

### Misfit

Mean over the 101 traces:

| stage | normalised misfit |
|---|---|
| initial model | 0.99 |
| after PSO | 0.15 |
| after QNM | 0.021 |

The synthetic-to-observed CC is 0.99.

### Figures

| file | content |
|---|---|
| `fig00_lambda_discrepancy.png` | λ selection |
| `fig01_well_tie_check.png`, `fig02_wavelet.png` | well tie, wavelet, spectra |
| `fig03_structure_and_initial_model.png` | DTW-tracked horizons and the structural 10 Hz initial model |
| `fig04_well_comparison_logs.png`, `fig05_well_crossplots.png` | true vs inverted impedance at the well |
| `fig06_inverted_impedance_section.png`, `fig06b_..._1800-2050ms.png` | 2-D inverted section with the well log inserted |
| `fig07_observed_synthetic_residual.png` | observed, synthetic and residual sections |
| `fig08_initial_pso_hpso_sections.png` | initial model, after PSO, after QNM |
| `fig09_misfit_per_trace.png` | misfit and synthetic CC per trace |
| `fig10_convergence_caseA.png`, `fig11_convergence_caseB.png` | HPSO convergence |
| `fig13_true_vs_inverted_section.png`, `fig14_true_vs_inverted_traces.png` | inverted vs true model over the whole section |

Numerical results:
* `well_comparison_metrics.csv`
* `section_vs_true_model_metrics.csv`
* `well_logs_true_vs_inverted.csv`
* `misfit_per_trace.csv`
* `summary.json`
* `inverted_impedance_section.npy` / `.txt`
* `initial_model_section.npy`
* `synthetic_section.npy`
