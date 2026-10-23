"""
Creates the synthetic data set used in this repository (fully synthetic, no field data).

  synthetic_data/synthetic_seismic.txt                 (201 samples x 101 traces) post-stack section, 2 ms, 1700-2100 ms
  synthetic_data/synthetic_well.csv                    true impedance + well-tied seismic trace at the well (trace 50)
  synthetic_data/synthetic_true_impedance_section.npy  true impedance of the whole section (for evaluation only)

Model
* impedance in "relative geologic time": blocky carbonate/shale-like layers (2-30 samples thick) with a
  compaction trend, a porous low-impedance interval and fine-scale (high-frequency) log detail
* structure: an anticline + regional dip + layer thickening with depth, and small lateral property changes
* seismic: reflectivity convolved with a 25 Hz Ricker wavelet rotated by 20 degrees, plus 15 % Gaussian
  noise; the well-tied trace has an independent 10 % noise realisation

    python make_synthetic_data.py
"""
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from hpso import ricker, rotate_phase, impedance_to_reflectivity

rng = np.random.default_rng(2026)
NS, NTR, DT, T0 = 201, 101, 0.002, 1700.0
WELL = 50
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_data")
os.makedirs(OUT, exist_ok=True)

# ---------------- layered model along relative geologic time (finer than the seismic sampling)
N_RGT = 420
ln_z = np.zeros(N_RGT)
i = 0
while i < N_RGT:
    th = int(rng.integers(2, 30))
    ln_z[i:i + th] = rng.normal(0.0, 0.10)
    i += th
trend = np.linspace(9.10, 9.48, N_RGT)                     # about 9000 -> 13100 (m/s)(g/cc)
reservoir = -0.16 * np.exp(-((np.arange(N_RGT) - 230) / 12.0) ** 4)   # porous low-impedance interval
fine = gaussian_filter1d(rng.standard_normal(N_RGT), 0.7) * 0.05      # high-frequency log detail
ln_z_rgt = trend + ln_z + reservoir + fine

# lateral property change (smooth along layers)
lat = gaussian_filter(rng.standard_normal((N_RGT, NTR)), sigma=(2, 15))
lat *= 0.03 / lat.std()                                       # ~3 % lateral change

# ---------------- structure: sample t of trace j -> relative geologic time
x = np.arange(NTR)
t = np.arange(NS)
anticline = 14.0 * np.exp(-((x - 72) / 18.0) ** 2)
dip = -0.12 * (x - WELL)
Z = np.zeros((NS, NTR))
for j in range(NTR):
    stretch = 1.0 + 0.12 * (j - WELL) / NTR                 # layers thicken towards one side
    rgt = 110 + (t - NS / 2) * stretch + anticline[j] - anticline[WELL] + dip[j]
    Z[:, j] = np.exp(np.interp(rgt, np.arange(N_RGT), ln_z_rgt + lat[:, j] - lat[:, WELL]))

# ---------------- seismic
w = 3.0e4 * rotate_phase(ricker(25.0, DT, 61), 20.0)
clean = np.column_stack([np.convolve(impedance_to_reflectivity(Z[:, j]), w, "same") for j in range(NTR)])
S = clean + 0.15 * clean.std() * rng.standard_normal(clean.shape)
s_well = clean[:, WELL] + 0.10 * clean.std() * rng.standard_normal(NS)

np.savetxt(os.path.join(OUT, "synthetic_seismic.txt"), S, fmt="%.4f")
pd.DataFrame({"Time_ms": T0 + 2 * t, "True_Impedance": Z[:, WELL], "Well_Tied_Seismic": s_well}
             ).to_csv(os.path.join(OUT, "synthetic_well.csv"), index=False, float_format="%.4f")
np.save(os.path.join(OUT, "synthetic_true_impedance_section.npy"), Z)
print(f"synthetic data written to {OUT}: section {S.shape}, well trace {WELL}, "
      f"AI {Z.min():.0f}-{Z.max():.0f}")
