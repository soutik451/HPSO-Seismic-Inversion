"""
Low-frequency initial model (LFM).

1. the well impedance is low-pass filtered (10 Hz, zero-phase Butterworth)
2. the structure of the section is tracked from the seismic itself: every trace is aligned to
   its neighbour with dynamic time warping (DTW, smooth shifts with a strain limit), starting
   at the well and going outwards. This gives, for every trace and time, the equivalent time
   at the well (a horizon-consistent "relative geologic time" mapping)
3. the filtered well log is interpolated along this mapping, so the LFM follows the reflectors
"""
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def lowpass(x, fc, dt, order=4):
    b, a = butter(order, fc / (0.5 / dt))
    return filtfilt(b, a, x, padtype="odd", padlen=min(3 * max(len(a), len(b)) * 10, len(x) - 1))


def dtw_shifts(f, g, max_shift=6, strain=1, win=11):
    """
    shifts u(t) (samples) such that g(t + u(t)) ~ f(t), via dynamic time warping
    (Hale, 2013) with |u(t) - u(t-1)| <= 1 every `strain` samples. Alignment errors are
    smoothed over `win` samples to make the tracking stable.
    """
    n = len(f)
    lags = np.arange(-max_shift, max_shift + 1)
    e = np.full((n, len(lags)), 1e6)
    fz = (f - f.mean()) / (f.std() + 1e-12)
    gz = (g - g.mean()) / (g.std() + 1e-12)
    for k, l in enumerate(lags):
        idx = np.arange(n) + l
        ok = (idx >= 0) & (idx < n)
        e[ok, k] = (fz[ok] - gz[idx[ok]]) ** 2
    e = uniform_filter1d(e, size=win, axis=0, mode="nearest")

    # accumulate (forward) with the shift change limited to 1 sample
    D = e.copy()
    for i in range(1, n):
        prev = D[i - 1]
        m = prev.copy()
        m[1:] = np.minimum(m[1:], prev[:-1])
        m[:-1] = np.minimum(m[:-1], prev[1:])
        D[i] += m
    # backtrack
    u = np.zeros(n, dtype=int)
    k = int(np.argmin(D[-1]))
    u[-1] = k
    for i in range(n - 2, -1, -1):
        cand = [c for c in (k - 1, k, k + 1) if 0 <= c < len(lags)]
        k = min(cand, key=lambda c: D[i, c])
        u[i] = k
    return gaussian_filter1d(lags[u].astype(float), 2.0)


def structural_mapping(section, well_idx, max_shift=6):
    """
    returns tau (n_samples, n_traces): for each trace and sample the corresponding sample
    index (fractional) at the well trace.
    """
    ns, ntr = section.shape
    t = np.arange(ns, dtype=float)
    tau = np.zeros((ns, ntr))
    tau[:, well_idx] = t
    for direction in (1, -1):
        j_prev = well_idx
        j = well_idx + direction
        while 0 <= j < ntr:
            # sample t of trace j corresponds to sample t + u(t) of trace j_prev
            u = dtw_shifts(section[:, j], section[:, j_prev], max_shift=max_shift)
            tau[:, j] = np.interp(t + u, t, tau[:, j_prev])
            j_prev, j = j, j + direction
    return tau


def structural_lfm(z_well, section, well_idx, dt, fc=10.0, max_shift=6, smooth_traces=3):
    """10 Hz well impedance propagated along the tracked structure -> (n_samples, n_traces)"""
    z_low = lowpass(np.log(z_well), fc, dt)
    tau = structural_mapping(section, well_idx, max_shift=max_shift)
    ns, ntr = section.shape
    t = np.arange(ns, dtype=float)
    m0 = np.zeros((ns, ntr))
    for j in range(ntr):
        m0[:, j] = np.interp(tau[:, j], t, z_low)
    if smooth_traces > 1:
        m0 = uniform_filter1d(m0, size=smooth_traces, axis=1, mode="nearest")
        m0[:, well_idx] = z_low            # keep the well trace exact
    # keep the model low-frequency after the mapping
    m0 = np.column_stack([lowpass(m0[:, j], fc, dt) for j in range(ntr)])
    return np.exp(m0), tau, np.exp(z_low)
