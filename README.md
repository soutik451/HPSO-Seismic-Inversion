# HPSO-Seismic-Inversion

Post-stack acoustic impedance inversion with **Hybrid Particle Swarm Optimization (HPSO)**: a global Particle Swarm Optimization (PSO) search followed by a local quasi-Newton (BFGS-type) refinement, constrained by a structurally consistent low-frequency initial model.

This is the impedance-inversion part of the workflow in *"Integrated Quantitative Reservoir Characterization using Hybrid Particle Swarm Optimization and Gated Linear Unit Residual Network for Acoustic Impedance Inversion and Petrophysical Property Prediction"* (Bhakta et al., submitted to Computers & Geosciences). The method follows Kant et al. (2024a, Phys. Chem. Earth 136, 103754, https://doi.org/10.1016/j.pce.2024.103754).

**Data:** the field data used in the paper are confidential. This repository uses a fully **synthetic** data set with the same format (201 samples × 101 traces, 2 ms, one well at the middle trace). It is created by `make_synthetic_data.py`, and the results in `results/` come from it.

## Method

1. **Forward model.** Convolutional model *s = W r(m)* with *m = ln Z* and *r_i = tanh((m_i − m_{i−1})/2)*, which equals *(Z_i − Z_{i−1})/(Z_i + Z_{i−1})*.
2. **Objective.**

   J(m) = ‖w ⊙ (d − W r(m))‖² / ‖w ⊙ d‖² + λ · mean(((m − m₀)/δ)²)

   * The first term is the normalised seismic misfit.
   * The second term is a soft constraint to the initial model m₀.
   * The search space is m₀ ± δ, with δ = 0.45 in ln Z.
   * w is a small taper on the data weights over the first and last 10 samples.
3. **Wavelet.**
   * Candidates are a statistical wavelet (smoothed average amplitude spectrum of the section) and Ricker wavelets of 15–50 Hz.
   * At the well-tied trace, each candidate gets a constant phase and bulk shift.
   * The candidate with the best well tie is kept and scaled by least squares.
4. **Low-frequency initial model (structurally constrained).**
   * The well impedance is low-pass filtered at 10 Hz (zero-phase Butterworth).
   * The section's structure is tracked trace-to-trace with dynamic time warping, starting at the well.
   * The filtered log is interpolated along the tracked horizons.
5. **HPSO, trace by trace.**
   * PSO: 50 particles, 100 iterations, inertia weight 0.9 → 0.4, c1 = c2 = 1.5. Particles start as band-limited perturbations of m₀.
   * The best PSO model is then refined by the quasi-Newton method (L-BFGS-B, analytic gradient, same bounds).
6. **λ by the discrepancy principle.** λ is chosen so the final misfit on the well-tied trace equals the noise level estimated from the well tie (1 − CC_tie²). The synthetic noise test uses a fixed λ = 0.5.

## Files

| file | purpose |
|---|---|
| `config.py` | data paths, column names, geometry and all inversion settings |
| `make_synthetic_data.py` | creates the synthetic data set in `synthetic_data/` |
| `01_synthetic_test.py` | single-trace noise test: well log, Ricker wavelet, 0 / 10 / 20 / 30 % Gaussian noise, PSO vs HPSO, 5 repeats each |
| `02_section_inversion.py` | wavelet and well tie, structural initial model, 2-D HPSO inversion, comparison with the true impedance at the well and over the whole section, all figures |
| `hpso/forward.py` | forward model, objective, analytic gradient |
| `hpso/optimizers.py` | PSO, quasi-Newton, HPSO, λ selection |
| `hpso/lfm.py` | low-pass filter, DTW structure tracking, structural initial model |
| `hpso/wavelet.py` | Ricker and statistical wavelets, phase rotation, scaling, tie scan |
| `hpso/plotting.py`, `hpso/metrics.py`, `hpso/data_io.py` | helpers |
| `quick_test.py` | quick test of the whole workflow (about 1 minute) |
| `synthetic_data/` | synthetic section, well file and true impedance section |
| `results/` | results of the full run (see `RESULTS.md`) |

## Installation

```
pip install -r requirements.txt
```

Python 3.10 was used. Only NumPy, SciPy, pandas, Matplotlib and openpyxl are needed; no GPU is required.

## Quick test

```
python quick_test.py
```

This recreates the synthetic data, then runs both scripts with fewer repeats; the outputs go to `results_quick/`. It checks that all outputs exist and that HPSO improves on the initial model, then prints `QUICK TEST PASSED`.

## Full run (reproduces `results/`)

```
python make_synthetic_data.py
python 01_synthetic_test.py
python 02_section_inversion.py
```

Total run time is about 2 minutes on a laptop CPU.

## Your own data

In `config.py`, set `DATA_DIR`, `SEISMIC_FILE`, `WELL_FILE`, the column names, `T0_MS`, `DT`, `WELL_TRACE` and `CDP_FIRST`.

* **Seismic:** a text matrix (n_samples rows × n_traces columns) or a `.npy` file.
* **Well file:** CSV or Excel containing the true impedance and, optionally, a well-tied seismic trace.
* **True section:** `TRUE_SECTION_FILE` exists only for synthetic data. If the file is not there, that comparison is skipped.

## License

MIT
