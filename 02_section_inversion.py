"""
HPSO post-stack acoustic impedance inversion of a 2-D seismic section (Section 4.2 of the paper),
here run on the synthetic data set in synthetic_data/ (the field data of the paper are confidential).

Case A  2-D section (SEISMIC_FILE): statistical zero-phase wavelet, structurally constrained 10 Hz
        initial model, trace-by-trace HPSO inversion, comparison with the true impedance at the well.
Case B  well-tied seismic trace at the well (WELL_SEIS_COL): wavelet phase / shift / scale from the
        well tie, HPSO inversion, comparison with the true impedance.

    python 02_section_inversion.py           # data in config.py -> results/
    python 02_section_inversion.py --quick   # same, results to results_quick/ (used by quick_test.py)
"""
import os, json, time, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
import hpso as H
from hpso.data_io import get_paths, load_section, load_well
from hpso.plotting import save, section, IMP_CMAP

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true", help="write to results_quick/ (quick test)")
p.add_argument("--well-trace", type=int, default=None, help="column index of the well (default: config)")
p.add_argument("--cdp-first", type=int, default=None, help="CDP number of the first trace (default: config)")
args = p.parse_args()
if args.well_trace is not None:
    cfg.WELL_TRACE = args.well_trace
if args.cdp_first is not None:
    cfg.CDP_FIRST = args.cdp_first

OUT = os.path.join(cfg.RESULTS_DIR + ("_quick" if args.quick else ""), "section_inversion")
os.makedirs(OUT, exist_ok=True)
seis_path, well_path = get_paths(cfg)

# ------------------------------------------------------------------ data
S = load_section(seis_path)[: cfg.N_SAMPLES]
well = load_well(well_path, cfg.N_SAMPLES)
z_true = well[cfg.WELL_IMP_COL].to_numpy(float)
ns, ntr = S.shape
jw = cfg.WELL_TRACE
t = cfg.T0_MS + np.arange(ns) * cfg.DT * 1000
cdp = cfg.CDP_FIRST + np.arange(ntr)
r_true = H.impedance_to_reflectivity(z_true)
z_band = np.exp(H.lowpass(np.log(z_true), cfg.EVAL_BAND_HZ, cfg.DT))
print(f"section {S.shape}, well trace index {jw} (CDP {cdp[jw]}), {t[0]:.0f}-{t[-1]:.0f} ms")
report = {"section_shape": list(S.shape), "well_cdp": int(cdp[jw])}

# ------------------------------------------------------------------ wavelet
# candidates: statistical (average spectrum of the section, zero phase) and Ricker wavelets.
# With a well-tied trace, every candidate gets a constant phase + bulk shift from the tie and the one
# with the best tie is kept, scaled by least squares. Without it: statistical, zero phase, RMS scale.
ws, fr, amp = H.statistical_wavelet(S, cfg.DT, cfg.WAVELET_LEN, smooth_hz=cfg.WAVELET_SMOOTH_HZ)
peak_f = fr[np.argmax(amp)]
has_tied = cfg.WELL_SEIS_COL in well.columns
if has_tied:
    s_well = well[cfg.WELL_SEIS_COL].to_numpy(float)
    cands_w = {"statistical": ws}
    cands_w.update({f"Ricker {f} Hz": H.ricker(f, cfg.DT, cfg.WAVELET_LEN) for f in cfg.RICKER_SCAN_HZ})
    ties = {k: H.tie_scan(v, r_true, s_well) for k, v in cands_w.items()}
    wname = max(ties, key=lambda k: ties[k][0])
    tie_cc, ph, sh = ties[wname]
    w_unit = np.roll(H.rotate_phase(cands_w[wname], ph), sh)
    wavelet = w_unit * H.scale_wavelet_ls(w_unit, r_true, s_well)
    wdesc = f"{wname}, phase {ph} deg, shift {sh} samples, least-squares scale (well tie CC {tie_cc:.3f})"
    report["wavelet_candidates_tie_CC"] = {k: round(v[0], 4) for k, v in ties.items()}
else:
    wavelet = ws * H.scale_wavelet_rms(ws, r_true, S)
    wdesc = "statistical, zero phase, RMS scale"
    tie_cc = None
print("wavelet:", wdesc)
report["wavelet"] = wdesc
W = H.conv_matrix(wavelet, ns)

# ------------------------------------------------------------------ well-tie check
syn_true = W @ r_true
cc_sec = float(np.corrcoef(syn_true, S[:, jw])[0, 1])
scan_sec = H.tie_scan(wavelet, r_true, S[:, jw], phases=[0], shifts=range(-40, 41))
report["tie_section_trace"] = {"CC": cc_sec, "best_CC_shift_pm80ms": scan_sec[0], "best_shift_samples": scan_sec[2]}
print(f"tie, section trace {jw}: CC = {cc_sec:.2f} (best within +/-80 ms: {scan_sec[0]:.2f} at {scan_sec[2]} samples)")
if has_tied:
    cc_tied = float(np.corrcoef(syn_true, s_well)[0, 1])
    report["tie_well_tied_trace"] = {"CC": cc_tied}
    print(f"tie, well-tied trace: CC = {cc_tied:.2f}")

fig, axs = plt.subplots(1, 4 if has_tied else 3, figsize=(10, 6.5), sharey=True)
axs[0].plot(z_true, t, "k", lw=1); axs[0].set_title("True AI"); axs[0].set_xlabel("AI (m/s·g/cc)")
axs[1].plot(r_true, t, "k", lw=0.8); axs[1].set_title("Reflectivity")
axs[2].plot(syn_true, t, "r", lw=1, label="synthetic (true log)")
axs[2].plot(S[:, jw], t, "k", lw=1, label=f"section trace (CDP {cdp[jw]})")
axs[2].set_title(f"Section trace at the well\nCC = {cc_sec:.2f}")
axs[2].legend(loc="lower right", fontsize=7)
if has_tied:
    axs[3].plot(syn_true, t, "r", lw=1, label="synthetic (true log)")
    axs[3].plot(s_well, t, "k", lw=1, label="well-tied trace")
    axs[3].set_title(f"Well-tied trace\nCC = {cc_tied:.2f}")
    axs[3].legend(loc="lower right", fontsize=7)
for a in axs[2:]:
    a.set_xlabel("Amplitude")
axs[0].invert_yaxis(); axs[0].set_ylabel("Two-way time (ms)")
fig.suptitle("Well tie check")
fig.tight_layout()
save(fig, os.path.join(OUT, "fig01_well_tie_check.png"))

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
tw = (np.arange(cfg.WAVELET_LEN) - cfg.WAVELET_LEN // 2) * cfg.DT * 1000
axs[0].plot(tw, wavelet / np.abs(wavelet).max(), "k"); axs[0].axhline(0, color="0.7", lw=0.5)
axs[0].set_xlabel("Time (ms)"); axs[0].set_title("Wavelet: " + wdesc.split(",")[0], fontsize=9)
fw, aw = H.wavelet.amplitude_spectrum(wavelet, cfg.DT)
axs[1].plot(fr, amp / amp.max(), color="0.5", label="section (smoothed)")
axs[1].plot(fw, aw / aw.max(), "k", label="wavelet")
axs[1].set_xlim(0, 150); axs[1].legend(fontsize=8)
axs[1].set_xlabel("Frequency (Hz)"); axs[1].set_ylabel("Normalised amplitude")
axs[1].set_title(f"Amplitude spectra (section peak {peak_f:.0f} Hz)")
fig.tight_layout()
save(fig, os.path.join(OUT, "fig02_wavelet.png"))

# ------------------------------------------------------------------ structural low-frequency model
Z0, tau, z_low = H.structural_lfm(z_true, S, jw, cfg.DT, fc=cfg.LFM_CUTOFF_HZ,
                                  max_shift=cfg.DTW_MAX_SHIFT, smooth_traces=cfg.LFM_LATERAL_SMOOTH)
np.save(os.path.join(OUT, "initial_model_section.npy"), Z0)

fig, axs = plt.subplots(2, 1, figsize=(9, 8.5))
section(axs[0], S, t, cdp, title="Seismic section with tracked horizons (DTW)", cbar_label="Amplitude",
        well_cdp=cdp[jw])
axs[0].contour(cdp, t, tau, levels=np.arange(10, ns - 10, 12), colors="k", linewidths=0.6)
vmin, vmax = np.percentile(z_true, [2, 98])
section(axs[1], Z0, t, cdp, cmap=IMP_CMAP, vmin=vmin, vmax=vmax,
        title=f"Structurally constrained initial model ({cfg.LFM_CUTOFF_HZ:.0f} Hz)",
        cbar_label="AI (m/s·g/cc)", well_cdp=cdp[jw])
fig.tight_layout()
save(fig, os.path.join(OUT, "fig03_structure_and_initial_model.png"))

# ------------------------------------------------------------------ data weights + lambda
wts = np.ones(ns)
if cfg.EDGE_TAPER > 0:
    ramp = np.sin(np.linspace(0, np.pi / 2, cfg.EDGE_TAPER)) ** 2
    wts[:cfg.EDGE_TAPER], wts[-cfg.EDGE_TAPER:] = ramp, ramp[::-1]

if has_tied:
    noise = 1.0 - tie_cc ** 2           # part of the trace not explained by the log
    m0B = np.log(z_low)
    lam, scan = H.choose_lambda(lambda l: H.Objective(s_well, W, m0B, lam=l, delta=cfg.BOUND, weights=wts),
                                m0B - cfg.BOUND, m0B + cfg.BOUND, noise, cfg.LAMBDA_GRID,
                                pso_kw=cfg.PSO, qnm_iter=cfg.QNM_ITER, seed=cfg.SEED)
    report["lambda"] = {"rule": "discrepancy principle on the well-tied trace", "noise_level": noise,
                        "lambda": lam, "scan_lambda_misfit": scan}
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sl, sm = np.array(scan).T
    ax.semilogx(sl, sm, "ko-"); ax.axhline(noise, color="r", ls="--", label=f"noise level 1-CC² = {noise:.2f}")
    ax.axvline(lam, color="b", ls=":", label=f"selected λ = {lam:.1f}")
    ax.set_xlabel("λ (weight of the initial-model constraint)"); ax.set_ylabel("final normalised misfit")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig00_lambda_discrepancy.png"))
else:
    lam = cfg.LAMBDA
    report["lambda"] = {"rule": "config LAMBDA", "lambda": lam}
print(f"lambda = {lam:.2f}")

# ------------------------------------------------------------------ CASE A: trace-by-trace HPSO
Zinv = np.zeros_like(S); Syn = np.zeros_like(S); Zpso = np.zeros_like(S)
mis = np.zeros((ntr, 3))
t0 = time.time()
for j in range(ntr):
    m0 = np.log(Z0[:, j])
    obj = H.Objective(S[:, j], W, m0, lam=lam, delta=cfg.BOUND, weights=wts)
    x, _, hist = H.hpso(obj, m0 - cfg.BOUND, m0 + cfg.BOUND, pso_kw=cfg.PSO,
                        qnm_iter=cfg.QNM_ITER, seed=cfg.SEED + j)
    Zinv[:, j] = np.exp(x); Zpso[:, j] = np.exp(hist["m_pso"]); Syn[:, j] = W @ H.reflectivity(x)
    mis[j] = hist["misfit_initial"], hist["misfit_after_pso"], hist["misfit_final"]
    if j == jw:
        hist_well = hist
    if (j + 1) % 20 == 0:
        print(f"  trace {j+1}/{ntr}  [{time.time()-t0:.0f} s]")
print(f"inversion of {ntr} traces: {time.time()-t0:.0f} s")
np.save(os.path.join(OUT, "inverted_impedance_section.npy"), Zinv)
np.save(os.path.join(OUT, "synthetic_section.npy"), Syn)
np.savetxt(os.path.join(OUT, "inverted_impedance_section.txt"), Zinv, fmt="%.2f")
pd.DataFrame(mis, columns=["misfit_initial", "misfit_after_PSO", "misfit_final"],
             index=pd.Index(cdp, name="CDP")).to_csv(os.path.join(OUT, "misfit_per_trace.csv"))

cc_syn = np.array([np.corrcoef(S[:, j], Syn[:, j])[0, 1] for j in range(ntr)])
report["caseA"] = {
    "mean_misfit_initial": float(mis[:, 0].mean()), "mean_misfit_after_pso": float(mis[:, 1].mean()),
    "mean_misfit_final": float(mis[:, 2].mean()),
    "synthetic_vs_observed_CC_mean": float(cc_syn.mean()),
    "synthetic_vs_observed_CC_well": float(cc_syn[jw]),
}

# ------------------------------------------------------------------ CASE B: well-tied trace
if has_tied:
    objB = H.Objective(s_well, W, m0B, lam=lam, delta=cfg.BOUND, weights=wts)
    xB, _, histB = H.hpso(objB, m0B - cfg.BOUND, m0B + cfg.BOUND, pso_kw=cfg.PSO,
                          qnm_iter=cfg.QNM_ITER, seed=cfg.SEED)
    zB = np.exp(xB)
    synB = W @ H.reflectivity(xB)
    report["caseB"] = {"misfit_initial": histB["misfit_initial"], "misfit_after_pso": histB["misfit_after_pso"],
                       "misfit_final": histB["misfit_final"],
                       "synthetic_vs_observed_CC": float(np.corrcoef(s_well, synB)[0, 1])}

# ------------------------------------------------------------------ comparison at the well
cands = {"Initial model (10 Hz)": z_low, "HPSO - section trace (case A)": Zinv[:, jw],
         "PSO only stage - section trace": Zpso[:, jw]}
if has_tied:
    cands["HPSO - well-tied trace (case B)"] = zB
for name, col in cfg.COMPARE_COLS.items():
    if col in well:
        cands[name] = well[col].to_numpy(float)
rows = []
for name, z in cands.items():
    a, b = H.scores(z_true, z), H.scores(z_band, z)
    rows.append({"result": name, **{k: round(v, 4) for k, v in a.items()},
                 "CC_0-80Hz": round(b["CC"], 4), "R2_0-80Hz": round(b["R2"], 4)})
comp = pd.DataFrame(rows).set_index("result")
comp.to_csv(os.path.join(OUT, "well_comparison_metrics.csv"))
print(comp.to_string())
report["well_comparison"] = comp.to_dict(orient="index")

pd.DataFrame({"time_ms": t, "true_AI": z_true, "true_AI_0-80Hz": z_band, "initial_model": z_low,
              "HPSO_caseA": Zinv[:, jw], **({"HPSO_caseB": zB} if has_tied else {})}
             ).to_csv(os.path.join(OUT, "well_logs_true_vs_inverted.csv"), index=False)

# figure: logs at the well
ncol = 3 if has_tied else 2
fig, axs = plt.subplots(1, ncol, figsize=(3.4 * ncol + 1, 7), sharey=True)
panels = [("Case A: section trace", Zinv[:, jw])] + ([("Case B: well-tied trace", zB)] if has_tied else [])
for ax, (title, z) in zip(axs, panels):
    sc = H.scores(z_true, z)
    ax.plot(z_true, t, "k", lw=1.2, label="True AI")
    ax.plot(z_low, t, color="0.55", ls="--", lw=1, label="Initial model")
    ax.plot(z, t, "r", lw=1.1, label="HPSO")
    ax.set_title(f"{title}\nCC = {sc['CC']:.2f},  R² = {sc['R2']:.2f}")
    ax.set_xlabel("AI (m/s·g/cc)"); ax.legend(loc="lower left", fontsize=7.5)
ax = axs[-1]
for name, z, c in [("Case A", Zinv[:, jw], "tab:blue")] + ([("Case B", zB, "tab:red")] if has_tied else []):
    ax.plot(z_band, t, "k", lw=1.2) if name == "Case A" else None
    ax.plot(z, t, color=c, lw=1, label=f"{name} (CC = {H.scores(z_band, z)['CC']:.2f})")
ax.set_title("True AI filtered 0-80 Hz"); ax.set_xlabel("AI (m/s·g/cc)"); ax.legend(loc="lower left", fontsize=7.5)
axs[0].invert_yaxis(); axs[0].set_ylabel("Two-way time (ms)")
fig.tight_layout()
save(fig, os.path.join(OUT, "fig04_well_comparison_logs.png"))

# figure: cross-plots
fig, axs = plt.subplots(1, ncol - 1 if has_tied else 1, figsize=(4.4 * (ncol - 1), 4.2), squeeze=False)
for ax, (title, z) in zip(axs[0], panels):
    sc = H.scores(z_true, z)
    ax.scatter(z_true, z, s=10, c=t, cmap="viridis")
    lo, hi = min(z_true.min(), z.min()), max(z_true.max(), z.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("True AI"); ax.set_ylabel("Inverted AI")
    ax.set_title(f"{title}\nCC = {sc['CC']:.2f}, R² = {sc['R2']:.2f}")
    ax.set_aspect("equal")
fig.tight_layout()
save(fig, os.path.join(OUT, "fig05_well_crossplots.png"))

# ------------------------------------------------------------------ section figures
vmin, vmax = np.percentile(Zinv, [2, 98])
fig, ax = plt.subplots(figsize=(10, 5.2))
section(ax, Zinv, t, cdp, cmap=IMP_CMAP, vmin=vmin, vmax=vmax, title="HPSO inverted acoustic impedance",
        cbar_label="AI (m/s·g/cc)")
# insert the true well log as a coloured strip at the well
strip = np.repeat(z_true[:, None], 3, axis=1)
ax.imshow(strip, aspect="auto", cmap=IMP_CMAP, vmin=vmin, vmax=vmax,
          extent=[cdp[jw] - 1.5, cdp[jw] + 1.5, t[-1], t[0]], interpolation="nearest")
ax.axvline(cdp[jw] - 1.5, color="k", lw=0.6); ax.axvline(cdp[jw] + 1.5, color="k", lw=0.6)
ax.text(cdp[jw], t[0] + 6, "well", ha="center", va="top", fontsize=8)
ax.set_xlim(cdp[0] - 0.5, cdp[-1] + 0.5); ax.set_ylim(t[-1], t[0])
fig.tight_layout()
save(fig, os.path.join(OUT, "fig06_inverted_impedance_section.png"))
ax.set_ylim(min(2050, t[-1]), max(1800, t[0]))       # window shown in the paper (1800-2050 ms)
ax.texts[0].set_position((cdp[jw], max(1800, t[0]) + 3))
fig.savefig(os.path.join(OUT, "fig06b_inverted_impedance_section_1800-2050ms.png"))
plt.close(fig)

fig, axs = plt.subplots(3, 1, figsize=(9, 12))
vmx = np.percentile(np.abs(S), 99)
section(axs[0], S, t, cdp, vmin=-vmx, vmax=vmx, title="Observed seismic", cbar_label="Amplitude", well_cdp=cdp[jw])
section(axs[1], Syn, t, cdp, vmin=-vmx, vmax=vmx, title=f"Synthetic from the inverted model (mean CC = {cc_syn.mean():.2f})",
        cbar_label="Amplitude", well_cdp=cdp[jw])
section(axs[2], S - Syn, t, cdp, vmin=-vmx, vmax=vmx, title="Residual (observed - synthetic)", cbar_label="Amplitude",
        well_cdp=cdp[jw])
fig.tight_layout()
save(fig, os.path.join(OUT, "fig07_observed_synthetic_residual.png"))

fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
vmin, vmax = np.percentile(Zinv, [2, 98])
section(axs[0], Z0, t, cdp, cmap=IMP_CMAP, vmin=vmin, vmax=vmax, title="Initial model", cbar_label="AI", well_cdp=cdp[jw])
section(axs[1], Zpso, t, cdp, cmap=IMP_CMAP, vmin=vmin, vmax=vmax, title="After PSO (global stage)", cbar_label="AI", well_cdp=cdp[jw])
section(axs[2], Zinv, t, cdp, cmap=IMP_CMAP, vmin=vmin, vmax=vmax, title="After QNM (HPSO result)", cbar_label="AI", well_cdp=cdp[jw])
for a in axs[1:]:
    a.set_ylabel("")
fig.tight_layout()
save(fig, os.path.join(OUT, "fig08_initial_pso_hpso_sections.png"))

# trace-wise misfit / CC
fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axs[0].plot(cdp, mis[:, 0], color="0.5", label="initial model")
axs[0].plot(cdp, mis[:, 1], "b", label="after PSO")
axs[0].plot(cdp, mis[:, 2], "r", label="after QNM (HPSO)")
axs[0].set_ylabel("Normalised misfit"); axs[0].set_yscale("log"); axs[0].legend(fontsize=8)
axs[1].plot(cdp, cc_syn, "k"); axs[1].set_ylabel("CC synthetic vs observed"); axs[1].set_xlabel("CDP")
for a in axs:
    a.axvline(cdp[jw], color="k", ls="--", lw=0.7)
axs[1].ticklabel_format(axis="x", style="plain", useOffset=False)
fig.tight_layout()
save(fig, os.path.join(OUT, "fig09_misfit_per_trace.png"))

# ------------------------------------------------------------------ convergence at the well (like Fig. 9)
def convergence_plot(hist, title, path):
    P = np.array(hist["pso"]["particles"])          # (iter, particles)
    q = np.array(hist["qnm"]["misfit"])
    n1 = P.shape[0]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    it1 = np.arange(1, n1 + 1)
    for k in range(P.shape[1]):
        ax.plot(it1, P[:, k], color="k", lw=0.3, alpha=0.35)
    ax.plot(it1, P.mean(axis=1), "r", lw=2.0, label="mean misfit of the swarm")
    ax.plot(it1, hist["pso"]["gbest"], color="orange", lw=1.5, label="global best (PSO)")
    it2 = n1 + np.arange(1, len(q) + 1)
    ax.plot(np.r_[n1, it2], np.r_[hist["misfit_after_pso"], q], "r", lw=2.0, ls="-")
    ax.axvline(n1, color="b", ls="--", lw=1.2, label="PSO -> QNM")
    ax.set_yscale("log"); ax.set_xlabel("Iteration"); ax.set_ylabel("Normalised misfit")
    ax.set_title(f"{title}\ninitial {hist['misfit_initial']:.2f} -> PSO {hist['misfit_after_pso']:.2f}"
                 f" -> QNM {hist['misfit_final']:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, path)

convergence_plot(hist_well, "HPSO convergence, section trace at the well (case A)", os.path.join(OUT, "fig10_convergence_caseA.png"))
if has_tied:
    convergence_plot(histB, "HPSO convergence, well-tied trace (case B)", os.path.join(OUT, "fig11_convergence_caseB.png"))

# ------------------------------------------------------------------ comparison with other results at the well
others = {k: v for k, v in cands.items() if k in cfg.COMPARE_COLS}
if others:
    fig, axs = plt.subplots(1, len(others) + 1, figsize=(3.2 * (len(others) + 1), 7), sharey=True)
    for ax, (name, z) in zip(axs, [("This code, case A", Zinv[:, jw])] + list(others.items())):
        ax.plot(z_true, t, "k", lw=1.1, label="True")
        ax.plot(z, t, "r", lw=1)
        ax.set_title(f"{name}\nCC = {H.scores(z_true, z)['CC']:.2f}", fontsize=9)
        ax.set_xlabel("AI")
    axs[0].invert_yaxis(); axs[0].set_ylabel("Two-way time (ms)")
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig12_comparison_other_results.png"))

# ------------------------------------------------------------------ whole section vs true model (synthetic data only)
true_path = os.path.join(cfg.DATA_DIR, getattr(cfg, "TRUE_SECTION_FILE", "") or "_none_")
if os.path.exists(true_path):
    Zt = load_section(true_path)[:ns]
    sec_rows = []
    for name, Z in [("Initial model", Z0), ("After PSO", Zpso), ("HPSO", Zinv)]:
        a = H.scores(Zt.ravel(), Z.ravel())
        cc_tr = np.array([np.corrcoef(Zt[:, j], Z[:, j])[0, 1] for j in range(ntr)])
        sec_rows.append({"result": name, "CC_section": round(a["CC"], 4), "R2_section": round(a["R2"], 4),
                         "RMSE_section": round(a["RMSE"], 1), "mean_trace_CC": round(float(cc_tr.mean()), 4),
                         "min_trace_CC": round(float(cc_tr.min()), 4)})
    sec_df = pd.DataFrame(sec_rows).set_index("result")
    sec_df.to_csv(os.path.join(OUT, "section_vs_true_model_metrics.csv"))
    print(sec_df.to_string())
    report["section_vs_true_model"] = sec_df.to_dict(orient="index")

    vmin, vmax = np.percentile(Zt, [2, 98])
    fig, axs = plt.subplots(3, 1, figsize=(9, 12))
    section(axs[0], Zt, t, cdp, cmap=IMP_CMAP, vmin=vmin, vmax=vmax, title="True impedance model",
            cbar_label="AI (m/s·g/cc)", well_cdp=cdp[jw])
    section(axs[1], Zinv, t, cdp, cmap=IMP_CMAP, vmin=vmin, vmax=vmax,
            title=f"HPSO inverted impedance (CC with true model = {sec_df.loc['HPSO', 'CC_section']:.2f})",
            cbar_label="AI (m/s·g/cc)", well_cdp=cdp[jw])
    err = 100 * (Zinv - Zt) / Zt
    section(axs[2], err, t, cdp, cmap="RdBu_r", vmin=-20, vmax=20, title="Error (inverted - true) / true",
            cbar_label="%", well_cdp=cdp[jw])
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig13_true_vs_inverted_section.png"))

    picks = [0, ntr // 4, jw, 3 * ntr // 4, ntr - 1]
    fig, axs = plt.subplots(1, len(picks), figsize=(2.6 * len(picks), 6.5), sharey=True)
    for ax, j in zip(axs, picks):
        ax.plot(Zt[:, j], t, "k", lw=1.0, label="true")
        ax.plot(Z0[:, j], t, color="0.55", ls="--", lw=0.9, label="initial")
        ax.plot(Zinv[:, j], t, "r", lw=1.0, label="HPSO")
        ax.set_title(f"CDP {cdp[j]}\nCC = {np.corrcoef(Zt[:, j], Zinv[:, j])[0, 1]:.2f}", fontsize=9)
        ax.set_xlabel("AI")
    axs[0].invert_yaxis(); axs[0].set_ylabel("Two-way time (ms)"); axs[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig14_true_vs_inverted_traces.png"))

json.dump(report, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)
print("results in", OUT)
