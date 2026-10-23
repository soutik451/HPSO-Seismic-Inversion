"""Wavelets: Ricker, statistical (from the seismic amplitude spectrum), phase rotation and scaling."""
import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import tukey


def ricker(f, dt, n=81):
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def rotate_phase(w, deg):
    """constant phase rotation of a wavelet (degrees)"""
    return np.real(hilbert(w) * np.exp(-1j * np.deg2rad(deg)))


def statistical_wavelet(section, dt, n=61, taper_frac=0.2, smooth_hz=3.0):
    """
    Zero-phase wavelet whose amplitude spectrum is the average amplitude spectrum of
    the (tapered) traces of the section, smoothed over `smooth_hz` (short traces give a
    spiky spectrum that reflects the reflectivity, not the wavelet). section: (n_samples, n_traces).
    """
    from scipy.ndimage import gaussian_filter1d
    ns = section.shape[0]
    tap = tukey(ns, taper_frac)[:, None]
    nfft = 1024
    A = np.abs(np.fft.rfft((section - section.mean(0)) * tap, n=nfft, axis=0)).mean(axis=1)
    A = gaussian_filter1d(A, smooth_hz * nfft * dt)
    w = np.fft.irfft(A, n=nfft)
    w = np.roll(w, n // 2)[:n] * tukey(n, 0.5)
    w /= np.max(np.abs(w))
    freqs = np.fft.rfftfreq(nfft, dt)
    return w, freqs, A


def amplitude_spectrum(x, dt, nfft=1024):
    f = np.fft.rfftfreq(nfft, dt)
    return f, np.abs(np.fft.rfft(x, n=nfft))


def scale_wavelet_ls(w, r, trace):
    """least-squares scale so that conv(r, a*w) fits the trace (needs a tied well)"""
    s = np.convolve(r, w, "same")
    return float(np.dot(s, trace) / (np.dot(s, s) + 1e-12))


def scale_wavelet_rms(w, r, section):
    """
    scale from RMS amplitudes only: rms(seismic) / rms(conv(r_well, w)).
    Uses the well reflectivity statistics, not its alignment with the seismic.
    """
    s = np.convolve(r, w, "same")
    return float(np.sqrt(np.mean(section ** 2)) / (np.sqrt(np.mean(s ** 2)) + 1e-12))


def tie_scan(w, r, trace, phases=range(-180, 180, 5), shifts=range(-10, 11), edge=15):
    """scan constant phase and bulk shift (samples) maximising correlation of synthetic and trace"""
    best = (-2, 0, 0)
    for ph in phases:
        s = np.convolve(r, rotate_phase(w, ph), "same")
        for sh in shifts:
            c = np.corrcoef(np.roll(s, sh)[edge:-edge], trace[edge:-edge])[0, 1]
            if c > best[0]:
                best = (c, ph, sh)
    return best
