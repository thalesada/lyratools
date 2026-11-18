"""Analytic profiles, empirical relations and simple physical estimates."""
import numpy as np

from . import config
from .constants import G, BOLTZMANN, PROTONMASS, MSUN, G_PC_KMS2_PER_MSUN, PC_PER_KMS_TO_MYR


# ---------------- profiles
def powerlaw(x, alpha, C):
    return C * x ** (-alpha)


def sersic(R, I_0, n, R_e):
    """Sersic profile with central intensity I_0, index n and half-light radius R_e."""
    b_n = 1.9992 * n - 0.3271
    return I_0 * np.exp(-b_n * ((R / R_e) ** (1 / n) - 1))


def nfw(r, rho_s, r_s):
    return rho_s / (r / r_s * (1 + r / r_s) ** 2)


def einasto(r, alpha, rho_s, r_s):
    return rho_s * np.exp(-2 / alpha * ((r / r_s) ** alpha - 1))


def schechter(M, Mc=2e6, C=1e8, beta=-2):
    return C * M ** beta * np.exp(-M / Mc)


# ---------------- Kennicutt-Schmidt
def ks_relation_figgs(sigma_gas):
    """Sigma_SFR [Msun/yr/kpc^2] for Sigma_gas [Msun/pc^2] (FIGGS fit)."""
    return 10 ** (-4.36 + np.log10(sigma_gas) * 1.5)


def ks_relation_things(sigma_gas):
    return 10 ** (-4.35 + np.log10(sigma_gas) * 1.43)


# ---------------- abundance matching (masses in Msun)
def guo_abundance_matching(mhalo):
    """Stellar mass from halo mass, Guo et al. (2010) eq. 3."""
    m = np.asarray(mhalo, float) / 1e10
    c, M0, alpha, beta, gamma = 0.129, 10 ** 1.4, -0.926, 0.261, -2.44
    return 1e10 * c * m * ((m / M0) ** alpha + (m / M0) ** beta) ** gamma


def moster_abundance_matching(mhalo, a, scatter=False):
    """
    Stellar mass from halo mass at scale factor a, Moster et al. (2013) eq. 2.
    With scatter=True returns (lower, upper) using the 1-sigma parameter ranges.
    """
    m = np.asarray(mhalo, float) / 1e10
    p = dict(m0=11.59, m1=1.195, n0=0.0351, n1=-0.0247, b0=1.376, b1=-0.826, g0=0.608, g1=0.329)
    e = dict(m0=0.236, m1=0.353, n0=0.0058, n1=0.0069, b0=0.153, b1=0.225, g0=0.059, g1=0.173)

    def evaluate(q):
        mm = 10.0 ** (q["m0"] + (1 - a) * q["m1"]) * 1e-10
        n = q["n0"] + (1 - a) * q["n1"]
        b = q["b0"] + (1 - a) * q["b1"]
        g = q["g0"] + (1 - a) * q["g1"]
        return 1e10 * m * 2.0 * n / ((m / mm) ** (-b) + (m / mm) ** g)

    if not scatter:
        return evaluate(p)
    lo = {k: v - e[k] for k, v in p.items()}
    hi = {k: v + e[k] for k, v in p.items()}
    return evaluate(lo), evaluate(hi)


def behroozi_abundance_matching(log_mhalo):
    """Stellar mass [Msun] from log10 halo mass: low-mass linear fit to Behroozi et al. (2013)."""
    beh = np.genfromtxt(config.data_path("behroozi"), unpack=True)
    p = np.poly1d(np.polyfit(beh[0][:8], beh[1][:8], 1))
    return 10 ** p(log_mhalo) * 10 ** np.asarray(log_mhalo)


# ---------------- simple estimates
def jeans_temperature(n, m_jeans, mu=0.62):
    """Temperature [K] at which number density n [cm^-3] has Jeans mass m_jeans [Msun]."""
    F = 4 * mu * PROTONMASS * G / (np.pi * BOLTZMANN)
    return F * (m_jeans * MSUN) ** (2 / 3.0) * (n * PROTONMASS) ** (1 / 3.0)


def relaxation_time(m_cluster, r_half_pc, mean_mass=0.547, gamma=0.02):
    """
    Two-body half-mass relaxation time [Gyr] (Spitzer 1987) for a cluster of mass
    m_cluster [Msun] and half-mass radius r_half_pc [pc]; mean stellar mass in Msun,
    Coulomb-logarithm factor gamma (Giersz & Heggie 1996).
    """
    N = m_cluster / mean_mass
    t = 0.138 * np.sqrt(m_cluster) * r_half_pc ** 1.5 / (mean_mass * np.sqrt(G_PC_KMS2_PER_MSUN) * np.log(gamma * N))
    return t * PC_PER_KMS_TO_MYR / 1e3


def evaporation_factor(m_init, mean_age_gyr, r_half_pc, m_half, **schechter_kwargs):
    """
    Survival weight of a star cluster under evaporation: Schechter(M + Delta) / Schechter(M),
    with mass loss Delta = mu_ev * age and mu_ev = 1100 rho_h^0.5 (McLaughlin & Fall 2008),
    rho_h = m_half / (4 pi r_half^3) [Msun/pc^3]. Returns (weight, Delta [Msun]).
    """
    rho_h = m_half / (4 * np.pi * r_half_pc ** 3)
    delta = 1100 * rho_h ** 0.5 * mean_age_gyr
    return schechter(m_init + delta, **schechter_kwargs) / schechter(m_init, **schechter_kwargs), delta


def rho_crit(z, cosmology):
    """Critical density [g/cm^3] at redshift z."""
    return cosmology.critical_density(z).to("g/cm**3").value


def rho_mean_matter(z, cosmology):
    """Mean cosmic matter density [g/cm^3] at redshift z."""
    return cosmology.Om(z) * rho_crit(z, cosmology)


def mag_at_sb(r_pc, sb):
    """Absolute magnitude of a disc of radius r_pc [pc] with surface brightness sb [mag/arcsec^2]."""
    return sb - 21.572 - 2.5 * np.log10(np.pi * np.asarray(r_pc) ** 2)
