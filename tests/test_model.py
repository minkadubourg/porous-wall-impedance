import numpy as np
import pytest

from porous_alpha import MATERIALS, PROFILES, fourier_coefficients, normalised_error, optimal_alpha
from porous_alpha.model import lambda1

L = 2.0
K_U, GK_U = fourier_coefficients(PROFILES["uniform"], L, 5)
K_G, GK_G = fourier_coefficients(PROFILES["gaussian"], L, 20)


@pytest.mark.parametrize("key", list(MATERIALS))
@pytest.mark.parametrize("omega", [300.0, 5_000.0, 25_000.0])
def test_plane_wave_optimum_is_closed_form(key, omega):
    mat = MATERIALS[key]
    alpha, err = optimal_alpha(omega, K_U, GK_U, mat, L)
    exact = complex(mat.eta1 * lambda1(np.array(0.0), omega, mat))
    assert abs(alpha - exact) / abs(exact) < 1e-5
    assert err < 1e-10


@pytest.mark.parametrize("scale", [1.0, 1e-4, 1e-10])
def test_optimum_independent_of_source_amplitude(scale):
    mat = MATERIALS["isorel"]
    ref, _ = optimal_alpha(3_000.0, K_G, GK_G, mat, L)
    scaled, _ = optimal_alpha(3_000.0, K_G, scale * GK_G, mat, L)
    assert abs(scaled - ref) / abs(ref) < 1e-4


def test_physical_sign_and_positive_residual():
    mat = MATERIALS["melamine"]
    for omega in [500.0, 8_000.0, 20_000.0]:
        alpha, err = optimal_alpha(omega, K_G, GK_G, mat, L)
        assert alpha.real > 0 and alpha.imag < 0
        assert err > 0
        assert normalised_error(alpha, omega, K_G, GK_G, mat, L) == pytest.approx(err)
