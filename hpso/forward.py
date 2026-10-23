"""
Convolutional forward model for post-stack acoustic impedance inversion.

The model is parameterised by m = ln(Z) at every time sample. The normal-incidence
reflection coefficient between samples i-1 and i is

    r_i = (Z_i - Z_{i-1}) / (Z_i + Z_{i-1}) = tanh((m_i - m_{i-1}) / 2)

and the synthetic trace is s = W r, with W the (centred, 'same' size) convolution
matrix of the wavelet.
"""
import numpy as np


def conv_matrix(wavelet, n):
    """n x n matrix W such that W @ r == np.convolve(r, wavelet, 'same') (odd-length wavelet)."""
    w = np.asarray(wavelet, dtype=np.float64)
    if len(w) % 2 == 0:
        w = np.append(w, 0.0)
    h = len(w) // 2
    W = np.zeros((n, n))
    for i in range(n):
        for k in range(len(w)):
            j = i - (k - h)
            if 0 <= j < n:
                W[i, j] = w[k]
    return W


def reflectivity(m):
    """m: (..., n) log-impedance -> r: (..., n), r[..., 0] = 0."""
    r = np.zeros_like(m)
    r[..., 1:] = np.tanh(0.5 * (m[..., 1:] - m[..., :-1]))
    return r


def synthetic(m, W):
    return reflectivity(m) @ W.T


def impedance_to_reflectivity(z):
    return reflectivity(np.log(np.asarray(z, dtype=np.float64)))


class Objective:
    """
    J(m) = ||d - W r(m)||^2 / ||d||^2  +  lam * mean(((m - m0) / delta)^2)

    first term  : normalised seismic misfit (the error plotted in the convergence figure)
    second term : soft constraint to the low-frequency initial model m0
    """

    def __init__(self, d, W, m0, lam=0.02, delta=0.45, weights=None):
        self.d = np.asarray(d, dtype=np.float64)
        self.W = W
        self.m0 = np.asarray(m0, dtype=np.float64)
        self.lam = lam
        self.delta = delta
        self.n = len(self.d)
        # data weights (e.g. a taper at the window edges where the wavelet is truncated)
        self.wt = np.ones(self.n) if weights is None else np.asarray(weights, dtype=np.float64)
        self.dnorm = float(np.sum(self.wt * self.d ** 2)) + 1e-12

    def misfit(self, m):
        """normalised data misfit, works for a single model (n,) or a swarm (p, n)"""
        res = synthetic(m, self.W) - self.d
        return np.sum(self.wt * res ** 2, axis=-1) / self.dnorm

    def prior(self, m):
        return np.mean(((m - self.m0) / self.delta) ** 2, axis=-1)

    def __call__(self, m):
        return self.misfit(m) + self.lam * self.prior(m)

    def grad(self, m):
        """analytic gradient of J for a single model (n,)"""
        dm = 0.5 * (m[1:] - m[:-1])
        r = np.zeros(self.n)
        r[1:] = np.tanh(dm)
        res = self.W @ r - self.d
        g_r = 2.0 * (self.W.T @ (self.wt * res)) / self.dnorm
        sech2 = 1.0 - r[1:] ** 2
        g = np.zeros(self.n)
        g[1:] += 0.5 * sech2 * g_r[1:]
        g[:-1] -= 0.5 * sech2 * g_r[1:]
        g += self.lam * 2.0 * (m - self.m0) / (self.delta ** 2 * self.n)
        return g

    def value_and_grad(self, m):
        return float(self(m)), self.grad(m)
