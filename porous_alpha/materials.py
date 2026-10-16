"""Porous materials described by a rigid-frame equivalent-fluid model.

Each material is characterised by three macroscopic parameters:
porosity phi, flow resistivity sigma (N.s.m^-4) and tortuosity alpha_h.
From these we derive the coefficients of the damped wave equation inside the wall

    div(eta_1 grad u) + omega^2 xi_1 (1 + i a / (xi_1 omega)) u = 0
"""
from dataclasses import dataclass

RHO_0 = 1.2        # air density (kg/m^3)
C_0 = 340.0        # speed of sound in air (m/s)
GAMMA_P = 7.0 / 5  # ratio of specific heats for air


@dataclass(frozen=True)
class Material:
    name: str
    phi: float      # porosity
    sigma: float    # flow resistivity
    alpha_h: float  # tortuosity

    @property
    def eta1(self) -> float:
        return self.phi / self.alpha_h

    @property
    def xi1(self) -> float:
        return self.phi * GAMMA_P / C_0**2

    @property
    def a(self) -> float:
        return self.sigma * self.phi**2 * GAMMA_P / (C_0**2 * RHO_0 * self.alpha_h)


MELAMINE = Material("Melamine foam", phi=0.99, sigma=14_000.0, alpha_h=1.02)
ISOREL = Material("ISOREL", phi=0.70, sigma=142_300.0, alpha_h=1.15)
BIRCH_LT = Material("Birch LT", phi=0.529, sigma=151_429.0, alpha_h=1.37)

MATERIALS = {"melamine": MELAMINE, "isorel": ISOREL, "birch": BIRCH_LT}
