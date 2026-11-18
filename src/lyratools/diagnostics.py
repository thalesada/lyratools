"""Simulation diagnostics."""
import numpy as np


def grav_timestep(acceleration, softening, a=1.0, psi=0.012):
    """AREPO gravitational timestep sqrt(2 psi a eps / (2.8 |acc|)) in code units."""
    acc = np.linalg.norm(np.asarray(acceleration, float), axis=1)
    return np.sqrt(2 * psi * a * np.asarray(softening, float) / 2.8 / acc)
