"""Compute every sweep used in the README and store it in results/sweeps.npz."""
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from porous_alpha import MATERIALS, PROFILES, fourier_coefficients, mode_errors, sweep  # noqa: E402

L = 2.0          # half-aperture and air-layer depth (m)
N_MODES = 20     # Fourier modes kept on each side of k = 0
OMEGAS = np.linspace(100, 30_000, 400)  # angular frequency (rad/s)
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "sweeps.npz")


def unnormalised_sweep(omegas, k, gk_of_omega, mat):
    """Baseline: minimise the raw |e| with L-BFGS-B from (1, 1), without normalisation."""
    out = []
    for om in omegas:
        gk = gk_of_omega(om)
        f = lambda z: abs(mode_errors(z[0] + 1j * z[1], om, k, gk, mat, L).sum())
        out.append(complex(*minimize(f, [1.0, 1.0], method="L-BFGS-B").x))
    return np.array(out)


def main():
    data = {"omegas": OMEGAS, "L": L}
    k_u, gk_u = fourier_coefficients(PROFILES["uniform"], L, N_MODES)
    k_g, gk_g = fourier_coefficients(PROFILES["gaussian"], L, N_MODES)

    for key, mat in MATERIALS.items():
        print("material", mat.name)
        a, e = sweep(OMEGAS, k_u, gk_u, mat, L)
        data[f"uniform_{key}_alpha"], data[f"uniform_{key}_err"] = a, e
        a, e = sweep(OMEGAS, k_g, gk_g, mat, L)
        data[f"gaussian_{key}_alpha"], data[f"gaussian_{key}_err"] = a, e

    # Amplitude artefact: narrow-band source centred on omega0 = 15000 rad/s
    print("narrow-band artefact")
    omega0, spread = 15_000.0, 1_000.0
    envelope = lambda om: np.exp(-((om - omega0) ** 2) / (2 * spread**2))
    mat = MATERIALS["isorel"]
    data["narrowband_naive_alpha"] = unnormalised_sweep(OMEGAS, k_u, lambda om: gk_u * envelope(om), mat)
    data["narrowband_envelope"] = envelope(OMEGAS)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **data)
    print("saved", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
