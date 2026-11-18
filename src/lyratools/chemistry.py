"""Abundance ratios, ion fractions, self-shielding, cooling and dynamical times."""
from functools import lru_cache

import h5py
import numpy as np
from scipy.interpolate import interpn

from . import config
from .constants import (BOLTZMANN, ELEMENTS, GAMMA_MINUS1, SOLAR_TABLE_ROWS, HYDROGEN_MASSFRAC, PROTONMASS, G, YR, MSUN, KPC)


@lru_cache(maxsize=None)
def _solar(path):
    return np.genfromtxt(path, skip_header=1, unpack=True)[1]


def abundance_ratio(metals, element1, element2, floor=-10.0):
    """
    [X/Y] relative to solar (Asplund et al. 2009) from an (N, 10) array of LYRA
    element mass fractions (GFM_Metals). Cells/stars with no Y get `floor`.
    """
    if element1 not in ELEMENTS or element2 not in ELEMENTS:
        raise KeyError(f"elements available: {list(ELEMENTS)}")
    solar = _solar(config.data_path("solar"))
    metals = np.asarray(metals, dtype=float)
    x, y = metals[:, ELEMENTS[element1]], metals[:, ELEMENTS[element2]]
    solar_ratio = solar[SOLAR_TABLE_ROWS[element1]] / solar[SOLAR_TABLE_ROWS[element2]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ab = np.log10(x / y / solar_ratio)
    return np.where(y == 0, floor, ab)


def ion_fraction(logT, logZ, lognH, redshift=0.0, element="Mg", ion=4, table=None):
    """
    log10 ion fraction from the Ploeckinger & Schaye (2020) tables, given log T [K],
    log Z/Zsun and log nH [cm^-3]. Out-of-range values give -50.
    """
    table = table or config.data_path("ion_table")
    anames = ["00hydrogen", "01helium", "02carbon", "03nitrogen", "04oxygen", "05neon",
              "06magnesium", "07silicon", "08sulphur", "09calcium", "10iron"]
    with h5py.File(table, "r") as plo:
        names = np.array(plo["ElementNamesShort"][()], dtype=str)
        lookup = dict(zip(names, anames))
        if element not in lookup:
            raise KeyError(f"element {element} not in table ({list(names)})")
        points = [plo["TableBins/RedshiftBins"][()], plo["TableBins/TemperatureBins"][()],
                  plo["TableBins/MetallicityBins"][()], plo["TableBins/DensityBins"][()]]
        data = plo["Tdep/IonFractions/" + lookup[element]][:, :, :, :, ion - 1]
    logT = np.asarray(logT, dtype=float)
    xi = np.column_stack([np.full(logT.shape, redshift), logT, np.asarray(logZ, float), np.asarray(lognH, float)])
    return interpn(points, data, xi, bounds_error=False, fill_value=-50)


@lru_cache(maxsize=None)
def _rahmati(path):
    return np.genfromtxt(path, unpack=True)


def self_shielding_factor(nH, redshift):
    """
    Fraction of the UV background that is shielded (1 - photoionisation suppression factor)
    from the Rahmati et al. (2013) fits, for nH [cm^-3] at the given redshift.
    """
    rah = _rahmati(config.data_path("rahmati"))
    alpha1, alpha2, beta, xi, n0, f = (np.interp(redshift, rah[0], rah[i]) for i in range(1, 7))
    nH = np.asarray(nH, dtype=float)
    A = (1 - f) * np.power(1 + np.power(nH / 10 ** n0, beta), alpha1)
    B = f * np.power(1 + nH / 10 ** n0, alpha2)
    return 1 - (A + B)


def cooling_time(u, rho, cooling_rate, X=HYDROGEN_MASSFRAC):
    """
    Cooling time [Myr] per cell. u: specific internal energy [cm^2/s^2],
    rho: density [g/cm^3], cooling_rate: GFM_CoolingRate = Lambda_net / nH^2 [erg cm^3 / s]
    (negative = cooling). Cells that are being heated get 0.
    """
    u, rho, lam = (np.asarray(v, dtype=float) for v in (u, rho, cooling_rate))
    nH = X * rho / PROTONMASS
    with np.errstate(divide="ignore", invalid="ignore"):
        t = u / (-(nH ** 2 / rho) * lam)
    return np.where(lam >= 0, 0.0, t / (1e6 * YR))


def cooling_time_within(mass, u, rho, cooling_rate, X=HYDROGEN_MASSFRAC, chunk=1 << 22):
    """
    Cooling time [Myr] of a gas reservoir: total thermal energy / total net cooling rate.
    mass [Msun], u [cm^2/s^2], rho [g/cm^3], cooling_rate [erg cm^3/s]; X scalar or per cell.
    Pass only the selected cells. Sums accumulate in float64, chunk by chunk.
    """
    mass, u, rho, lam = (np.asarray(v) for v in (mass, u, rho, cooling_rate))
    X = np.asarray(X)
    e_th = edot = 0.0
    for a in range(0, len(mass), chunk):
        sl = slice(a, a + chunk)
        m_g = mass[sl].astype(np.float64) * MSUN
        r = rho[sl].astype(np.float64)
        nH = (X[sl] if X.ndim else X).astype(np.float64) * r / PROTONMASS
        e_th += np.sum(m_g * u[sl], dtype=np.float64)
        edot += np.sum(nH ** 2 * lam[sl].astype(np.float64) * m_g / r, dtype=np.float64)   # erg/s, negative = cooling
    return e_th / -edot / (1e6 * YR) if edot < 0 else np.inf


def dynamical_time(mass_msun, radius_kpc):
    """Dynamical time 1/sqrt(4 pi G rho) [Myr] of a mass [Msun] enclosed within radius_kpc."""
    rho = mass_msun * MSUN / (4.0 / 3.0 * np.pi * (radius_kpc * KPC) ** 3)
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt(4 * np.pi * G * rho) / (1e6 * YR)


def dynamical_time_within(r_kpc, mass_msun, radius_kpc):
    """Dynamical time 1/sqrt(4 pi G rho) [Myr] of all matter within radius_kpc."""
    m = np.sum(np.asarray(mass_msun)[np.asarray(r_kpc) < radius_kpc], dtype=np.float64)
    return dynamical_time(m, radius_kpc)


def temperature(u, electron_abundance, X=HYDROGEN_MASSFRAC):
    """Gas temperature [K] from specific internal energy u [cm^2/s^2] and electron abundance (per H atom)."""
    X = np.asarray(X, np.float64)
    mu = 4.0 / (1.0 + 3.0 * X + 4.0 * X * np.asarray(electron_abundance, np.float64))
    return GAMMA_MINUS1 * np.asarray(u, np.float64) / BOLTZMANN * mu * PROTONMASS


# ------------------------------------------------------------------ abundances by number
def twelve_log_oh_metals(metals):
    """12 + log10(O/H) by number per cell/star from an (N, 10) GFM_Metals array (NaN without O or H)."""
    from .timeseries.main import oh_per_particle
    metals = np.asarray(metals, dtype=float)
    return oh_per_particle(metals[:, ELEMENTS["O"]], metals[:, ELEMENTS["H"]])


def twelve_log_oh_reservoir(metals, mass, sel=None):
    """
    12 + log10(O/H) by number of a reservoir: ratio of the summed O and H masses (the
    definition of the time-series code, lyratools.timeseries.main.oh_from_sums).
    metals: (N, 10) GFM_Metals, mass: (N,) any mass unit; sel: optional mask.
    """
    from .timeseries.main import oh_from_sums
    metals = np.asarray(metals)
    return oh_from_sums(metals[:, ELEMENTS["O"]], metals[:, ELEMENTS["H"]], np.asarray(mass), sel)


def abundance_ratio_reservoir(metals, mass, element1, element2, sel=None):
    """[X/Y] (Asplund et al. 2009) of a reservoir from the summed element masses; NaN if empty."""
    metals, mass = np.asarray(metals), np.asarray(mass)
    if sel is not None:
        metals, mass = metals[sel], mass[sel]
    sx = np.sum(metals[:, ELEMENTS[element1]] * mass, dtype=np.float64)
    sy = np.sum(metals[:, ELEMENTS[element2]] * mass, dtype=np.float64)
    if not (sx > 0 and sy > 0):
        return np.nan
    solar = _solar(config.data_path("solar"))
    return float(np.log10(sx / sy / (solar[SOLAR_TABLE_ROWS[element1]] / solar[SOLAR_TABLE_ROWS[element2]])))
