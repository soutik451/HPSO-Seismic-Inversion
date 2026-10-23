"""
Synthetic noise test (Section 4.1 of the paper).

The acoustic impedance of the well is the true model. A synthetic trace is made by convolving its
reflectivity with a Ricker wavelet, Gaussian noise of 0, 10, 20 and 30 % is added, and the trace is
inverted with stand-alone PSO and with HPSO (PSO + quasi-Newton), both starting from the 10 Hz
low-frequency model. Every noise level is repeated with N_REPEATS noise realisations / PSO seeds.

    python 01_synthetic_test.py                        # -> results/synthetic_test
    python 01_synthetic_test.py --quick --repeats 2    # quick test -> results_quick/
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
import hpso as H
from hpso.plotting import save

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true", help="write to results_quick/ (quick test)")
p.add_argument("--repeats", type=int, default=cfg.N_REPEATS)
args = p.parse_args()

out = os.path.join(cfg.RESULTS_DIR + ("_quick" if args.quick else ""), "synthetic_test")
os.makedirs(out, exist_ok=True)

from hpso.data_io import get_paths, load_well
_, well_path = get_paths(cfg)
well = load_well(well_path, cfg.N_SAMPLES)
z_true = well[cfg.WELL_IMP_COL].to_numpy(float)
n = len(z_true)
t = cfg.T0_MS + np.arange(n) * cfg.DT * 1000

w = H.ricker(cfg.RICKER_FREQ, cfg.DT, cfg.WAVELET_LEN)
W = H.conv_matrix(w, n)
d_clean = W @ H.impedance_to_reflectivity(z_true)
m0 = np.log(H.lowpass(z_true, cfg.LFM_CUTOFF_HZ, cfg.DT))
lb, ub = m0 - cfg.BOUND, m0 + cfg.BOUND
z_band = np.exp(H.lowpass(np.log(z_true), cfg.EVAL_BAND_HZ, cfg.DT))

rows, examples = [], {}
t0 = time.time()
for nl in cfg.NOISE_LEVELS:
    for rep in range(args.repeats):
        rng = np.random.default_rng(1000 + rep)
        d = d_clean + nl * d_clean.std() * rng.standard_normal(n)
        obj = H.Objective(d, W, m0, lam=cfg.LAMBDA, delta=cfg.BOUND)

        pso_kw = dict(cfg.PSO); pso_kw["n_iter"] = cfg.PSO_ONLY_ITER
        x_pso, _, _ = H.pso(obj, lb, ub, seed=cfg.SEED + rep, x0=m0, **pso_kw)
        x_h, _, hist = H.hpso(obj, lb, ub, pso_kw=cfg.PSO, qnm_iter=cfg.QNM_ITER, seed=cfg.SEED + rep)

        for name, x in [("Initial model", m0), ("PSO", x_pso), ("HPSO", x_h)]:
            s_full = H.scores(z_true, np.exp(x))
            s_band = H.scores(z_band, np.exp(x))
            rows.append({"noise_%": int(nl * 100), "repeat": rep, "method": name,
                         "CC": s_full["CC"], "R2": s_full["R2"], "RMSE": s_full["RMSE"],
                         "CC_band": s_band["CC"], "R2_band": s_band["R2"],
                         "seismic_misfit": float(obj.misfit(x))})
        if rep == 0:
            examples[nl] = dict(d=d, pso=np.exp(x_pso), hpso=np.exp(x_h), hist=hist)
    print(f"  noise {int(nl*100):2d} %  done  [{time.time()-t0:.0f} s]")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "synthetic_test_all_runs.csv"), index=False)
summ = df.groupby(["noise_%", "method"])[["CC", "R2", "RMSE", "CC_band", "seismic_misfit"]].agg(["mean", "std"])
summ.round(4).to_csv(os.path.join(out, "synthetic_test_summary.csv"))
print(summ.round(3).to_string())

# ---------- figure: comparison for each noise level (like Fig. 6) ----------
fig, axs = plt.subplots(len(cfg.NOISE_LEVELS), 1, figsize=(8, 2.6 * len(cfg.NOISE_LEVELS)), sharex=True)
for ax, nl in zip(np.atleast_1d(axs), cfg.NOISE_LEVELS):
    e = examples[nl]
    sel = df[(df["noise_%"] == int(nl * 100))]
    cc_p = sel[sel.method == "PSO"]["CC"]; cc_h = sel[sel.method == "HPSO"]["CC"]
    ax.plot(t, z_true, "k", lw=1.4, label="True impedance")
    ax.plot(t, np.exp(m0), color="0.6", ls="--", lw=1.0, label="Initial model (10 Hz)")
    ax.plot(t, e["pso"], "b", lw=1.0, label=f"PSO (CC = {cc_p.mean():.2f} ± {cc_p.std():.2f})")
    ax.plot(t, e["hpso"], "r", lw=1.0, label=f"HPSO (CC = {cc_h.mean():.2f} ± {cc_h.std():.2f})")
    ax.set_ylabel("AI (m/s·g/cc)")
    ax.set_title(f"{int(nl*100)} % Gaussian noise", loc="left")
    ax.legend(loc="upper left", ncol=2, frameon=False)
np.atleast_1d(axs)[-1].set_xlabel("Two-way time (ms)")
fig.tight_layout()
save(fig, os.path.join(out, "fig_synthetic_noise_test.png"))

# ---------- figure: CC vs noise ----------
fig, ax = plt.subplots(figsize=(5, 3.6))
for name, c in [("PSO", "b"), ("HPSO", "r"), ("Initial model", "0.5")]:
    g = df[df.method == name].groupby("noise_%")["CC"]
    ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, color=c, marker="o", capsize=3, label=name)
ax.set_xlabel("Noise (%)"); ax.set_ylabel("CC with true impedance")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, os.path.join(out, "fig_synthetic_cc_vs_noise.png"))

# ---------- figure: synthetic seismograms ----------
fig, axs = plt.subplots(1, len(cfg.NOISE_LEVELS), figsize=(9, 5), sharey=True)
for ax, nl in zip(axs, cfg.NOISE_LEVELS):
    ax.plot(examples[nl]["d"], t, "k", lw=0.8)
    ax.fill_betweenx(t, 0, examples[nl]["d"], where=examples[nl]["d"] > 0, color="k", alpha=0.6)
    ax.set_title(f"{int(nl*100)} %"); ax.set_xticks([])
axs[0].invert_yaxis(); axs[0].set_ylabel("Two-way time (ms)")
fig.suptitle(f"Synthetic seismograms (Ricker {cfg.RICKER_FREQ:.0f} Hz)")
fig.tight_layout()
save(fig, os.path.join(out, "fig_synthetic_seismograms.png"))

json.dump({"wavelet": f"Ricker {cfg.RICKER_FREQ} Hz", "lambda": cfg.LAMBDA, "bound": cfg.BOUND,
           "pso": cfg.PSO, "qnm_iter": cfg.QNM_ITER, "pso_only_iter": cfg.PSO_ONLY_ITER,
           "repeats": args.repeats},
          open(os.path.join(out, "settings.json"), "w"), indent=2)
print("results in", out)
