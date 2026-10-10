"""Equivalent Robin coefficient alpha(omega) for a porous wall.

We look for a complex alpha such that the air-only problem with the Robin
condition  eta_0 du/dn + alpha u = 0  on the wall reproduces, as closely as
possible, the full air + porous-wall transmission problem. The mismatch is
measured mode by mode in an H^1-type norm (weights A, B) and summed over the
Fourier modes of the source:

    e(alpha, omega) = sum_k e_k(alpha, omega)

All functions are vectorised over the mode array k.
"""
import numpy as np
from scipy.optimize import minimize

from .materials import C_0, Material

ETA_0 = 1.0
XI_0 = 1.0 / C_0**2


def lambda0(k, omega):
    """Decay rate in air: real for evanescent modes, imaginary for propagating ones."""
    d = k**2 - XI_0 / ETA_0 * omega**2
    return np.where(d >= 0, np.sqrt(np.abs(d)) + 0j, 1j * np.sqrt(np.abs(d)))


def lambda1(k, omega, mat: Material):
    """Complex decay rate inside the porous wall (Re > 0, Im < 0)."""
    t = k**2 - mat.xi1 / mat.eta1 * omega**2
    r = np.sqrt(t**2 + (mat.a / mat.eta1 * omega) ** 2)
    return np.sqrt((t + r) / 2) - 1j * np.sqrt((-t + r) / 2)


def _f(x, lam0, L):
    return (lam0 * ETA_0 - x) * np.exp(-lam0 * L) + (lam0 * ETA_0 + x) * np.exp(lam0 * L)


def mode_errors(alpha, omega, k, gk, mat: Material, L, A=1.0, B=1.0):
    """e_k(alpha, omega) for every mode. Real-valued up to round-off."""
    l0 = lambda0(k, omega)
    l1e = lambda1(k, omega, mat) * mat.eta1
    f1, fa = _f(l1e, l0, L), _f(alpha, l0, L)
    chi = gk * ((l0 * ETA_0 - l1e) / f1 - (l0 * ETA_0 - alpha) / fa)
    gam = gk * ((l0 * ETA_0 + l1e) / f1 - (l0 * ETA_0 + alpha) / fa)

    c2, g2 = np.abs(chi) ** 2, np.abs(gam) ** 2
    em, ep = np.exp(-2 * l0 * L), np.exp(2 * l0 * L)
    cross = chi * np.conj(gam)
    w = A + B * k**2

    evanescent = k**2 >= XI_0 / ETA_0 * omega**2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e_ev = (w * ((c2 * (1 - em) + g2 * (ep - 1)) / (2 * l0) + 2 * L * np.real(cross))
                + B * l0 / 2 * (c2 * (1 - em) + g2 * (ep - 1))
                - 2 * B * l0**2 * L * np.real(cross))
        osc = np.imag(cross * (1 - em))
        e_pr = (w * (L * (c2 + g2) + 1j / l0 * osc)
                + B * L * np.abs(l0) ** 2 * (c2 + g2)
                + 1j * B * l0 * osc)
    return np.real(np.where(evanescent, e_ev, e_pr))


def normalised_error(alpha, omega, k, gk, mat, L, **w):
    """e(alpha, omega) / sum |g_k|^2.

    The objective is quadratic in the source, so dividing by its energy makes
    the optimisation independent of the source amplitude. Without this, a weak
    source (e.g. far from the centre of a narrow-band spectrum) gives an
    objective so small that the optimiser stops at its initial guess.
    """
    return mode_errors(alpha, omega, k, gk, mat, L, **w).sum() / np.sum(np.abs(gk) ** 2)


def optimal_alpha(omega, k, gk, mat, L, starts=((1.0, -1.0), (20.0, -20.0), (100.0, -100.0)), **w):
    """Minimise the normalised error over alpha in C from several starting points."""
    best = None
    for s in starts:
        res = minimize(lambda z: normalised_error(z[0] + 1j * z[1], omega, k, gk, mat, L, **w),
                       np.asarray(s, float), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    return complex(best.x[0], best.x[1]), best.fun


def sweep(omegas, k, gk, mat, L, **w):
    """alpha*(omega) and the residual error over a frequency grid (warm-started)."""
    alphas, errs = np.empty(len(omegas), complex), np.empty(len(omegas))
    prev = None
    for i, om in enumerate(omegas):
        starts = ((1.0, -1.0), (20.0, -20.0), (100.0, -100.0))
        if prev is not None:
            starts = ((prev.real, prev.imag),) + starts
        alphas[i], errs[i] = optimal_alpha(om, k, gk, mat, L, starts=starts, **w)
        prev = alphas[i]
    return alphas, errs
