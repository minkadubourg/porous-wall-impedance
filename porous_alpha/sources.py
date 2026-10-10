"""Incident sources g(y) on the left boundary and their Fourier coefficients.

The domain is periodic in y with period 2L, so the source is expanded on the
modes k_n = n pi / L. Because the objective is quadratic in g, only the
*relative* weights |g_k|^2 matter for the optimal alpha: any frequency envelope
g(y, omega) = s(omega) g(y) rescales the objective and leaves alpha unchanged.
"""
import numpy as np
from scipy.integrate import quad


def uniform(y):
    """Plane wave: constant over the whole aperture. Excites only k = 0."""
    return 1.0


def gaussian_beam(width=1.0):
    """Spatially localised source, g(y) = exp(-y^2 / width^2)."""
    return lambda y: np.exp(-(y / width) ** 2)


def fourier_coefficients(g, L, n_modes):
    """g_k = 1/(2L) * int_{-L}^{L} g(y) exp(-i k y) dy for k = n pi / L, |n| <= n_modes."""
    n = np.arange(-n_modes, n_modes + 1)
    k = n * np.pi / L
    gk = np.empty(k.shape, dtype=complex)
    for i, kk in enumerate(k):
        re, _ = quad(lambda y: g(y) * np.cos(kk * y), -L, L, limit=200)
        im, _ = quad(lambda y: -g(y) * np.sin(kk * y), -L, L, limit=200)
        gk[i] = (re + 1j * im) / (2 * L)
    return k, gk


PROFILES = {"uniform": uniform, "gaussian": gaussian_beam(1.0)}
