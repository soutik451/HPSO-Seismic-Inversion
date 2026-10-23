"""
Particle Swarm Optimization (global search), Quasi-Newton (BFGS-type, local refinement)
and the hybrid PSO -> QNM scheme (HPSO).

PSO update (Kennedy and Eberhart, 1995, with linearly decreasing inertia weight):
    v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x)
    x <- x + v
r1 and r2 are drawn once per particle and iteration, and the swarm is initialised with
band-limited random perturbations of the low-frequency model, so the particles stay inside
the seismic bandwidth (the seismic data do not constrain frequencies above it).

The best PSO model is then passed to the quasi-Newton method (L-BFGS-B: limited-memory BFGS
approximation of the inverse Hessian, with the same bounds as the PSO search space).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt


def _bandlimited_noise(rng, shape, band):
    b, a = butter(4, band)
    x = filtfilt(b, a, rng.standard_normal(shape), axis=-1)
    return x / (x.std() + 1e-12)


def pso(obj, lb, ub, n_particles=50, n_iter=100, w_start=0.9, w_end=0.4, c1=1.5, c2=1.5,
        vmax_frac=0.2, init_std=0.15, band=0.32, seed=0, x0=None, track=True):
    """
    band     : cut-off of the initial perturbations as a fraction of Nyquist (0.32 = 80 Hz at 2 ms)
    init_std : std of the initial perturbations (in ln Z)
    """
    rng = np.random.default_rng(seed)
    n = len(lb)
    span = ub - lb
    centre = x0 if x0 is not None else 0.5 * (lb + ub)
    x = np.clip(centre + init_std * _bandlimited_noise(rng, (n_particles, n), band), lb, ub)
    x[0] = np.clip(centre, lb, ub)           # one particle starts at the initial model
    v = 0.1 * init_std * _bandlimited_noise(rng, (n_particles, n), band)
    vmax = vmax_frac * span

    f = obj(x)
    pbest, pbest_f = x.copy(), f.copy()
    g = int(np.argmin(f))
    gbest, gbest_f = x[g].copy(), float(f[g])

    hist = {"gbest": [], "mean": [], "particles": []}
    for it in range(n_iter):
        w = w_start - (w_start - w_end) * it / max(n_iter - 1, 1)
        r1, r2 = rng.random((n_particles, 1)), rng.random((n_particles, 1))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -vmax, vmax)
        x = x + v
        out = (x < lb) | (x > ub)
        x = np.clip(x, lb, ub)
        v[out] *= -0.5                       # damped bounce at the search-space walls

        f = obj(x)
        better = f < pbest_f
        pbest[better], pbest_f[better] = x[better], f[better]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])
        if track:
            mis = obj.misfit(x)
            hist["particles"].append(mis)
            hist["mean"].append(float(np.mean(mis)))
            hist["gbest"].append(float(obj.misfit(gbest)))
    return gbest, gbest_f, hist


def qnm(obj, x0, lb, ub, max_iter=200, track=True):
    hist = {"misfit": []}

    def cb(xk):
        if track:
            hist["misfit"].append(float(obj.misfit(xk)))

    res = minimize(obj.value_and_grad, x0, jac=True, method="L-BFGS-B",
                   bounds=list(zip(lb, ub)), callback=cb,
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-9})
    return res.x, float(res.fun), hist


def choose_lambda(make_obj, lb, ub, target, grid, pso_kw=None, qnm_iter=200, seed=0):
    """
    discrepancy principle: the weight of the initial-model constraint is chosen so that the
    final normalised misfit equals the noise level `target` (data should not be fitted
    better than their noise). make_obj(lam) -> Objective. Returns lam and the scan table.
    """
    rows = []
    for lam in grid:
        obj = make_obj(lam)
        _, _, h = hpso(obj, lb, ub, pso_kw=pso_kw, qnm_iter=qnm_iter, seed=seed)
        rows.append((lam, h["misfit_final"]))
    lams, mis = np.array(rows).T
    # misfit grows with lambda: interpolate log(lambda) at the target misfit
    if target <= mis.min():
        lam = lams[np.argmin(mis)]
    elif target >= mis.max():
        lam = lams[np.argmax(mis)]
    else:
        order = np.argsort(mis)
        lam = float(np.exp(np.interp(target, mis[order], np.log(lams[order]))))
    return lam, rows


def hpso(obj, lb, ub, pso_kw=None, qnm_iter=200, seed=0):
    """global PSO search followed by quasi-Newton refinement of the best PSO model"""
    pso_kw = dict(pso_kw or {})
    x_pso, f_pso, h1 = pso(obj, lb, ub, seed=seed, x0=obj.m0, **pso_kw)
    x, f, h2 = qnm(obj, x_pso, lb, ub, max_iter=qnm_iter)
    hist = {"pso": h1, "qnm": h2, "misfit_initial": float(obj.misfit(obj.m0)),
            "misfit_after_pso": float(obj.misfit(x_pso)), "misfit_final": float(obj.misfit(x)),
            "m_pso": x_pso}
    return x, f, hist
