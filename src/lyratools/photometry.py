"""
Stellar photometry from PARSEC isochrones, for LYRA's individually resolved stars
(initial mass above `mass_limit`) and unresolved populations (particles of exactly
`mass_limit`, integrated over a truncated Kroupa IMF).

    mags = magnitudes(Z, age_yr, m_init, filters="optical")      # dict band -> array
    L_V  = luminosity(mags["Vmag"], zeropoint=4.8)               # Lsun (or Jy with JWST zeropoints)
"""
import glob
from functools import lru_cache

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline, interpn

from . import config

PARSEC_SOLAR = 0.0152
FILL_MAG = 35.0


# ----------------------------------------------------------------- isochrone tables
def read_parsec(pattern):
    """Read all PARSEC isochrone files matching `pattern` into one structured array."""
    files = np.sort(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"no PARSEC files match {pattern}")
    data, mags = None, None
    for f in files:
        names = np.genfromtxt(f, comments="@", skip_header=13, max_rows=1, dtype=str)
        if names[28] != "mbolmag":
            raise ValueError(f"cannot find magnitude columns in {f}")
        mags = list(names[29:])
        temp = np.genfromtxt(f, usecols=np.concatenate([[1, 2, 3], np.arange(28, 28 + len(mags))]),
                             names="MH, logAge, Mini, " + ", ".join(mags))
        data = temp if data is None else np.hstack([data, temp])
    logZ_bins, first = np.unique(data["MH"], return_index=True)
    mh_idx = np.append(first, len(data))
    age_idx = np.array([], dtype=int)
    for i in range(len(logZ_bins)):
        logAge_bins, d = np.unique(data["logAge"][mh_idx[i]:mh_idx[i + 1]], return_index=True)
        age_idx = np.append(age_idx, d + mh_idx[i])
    idx = np.append(age_idx, len(data))
    return data, logZ_bins, logAge_bins, mags, idx


@lru_cache(maxsize=4)
def parsec_cube(pattern, mass_limit=4.0, len_mass=200):
    """Isochrone magnitudes re-gridded onto a regular (log Z, log age, log mass) cube."""
    data, logZ, logAge, mags, idx = read_parsec(pattern)
    mmin = max(mass_limit, np.unique(data["Mini"]).min())
    mmax = min(100.0, np.unique(data["Mini"]).max())
    logM = np.linspace(np.log10(mmin), np.log10(mmax), len_mass)
    cube = np.zeros((len(logZ), len(logAge), len(logM), len(mags)))
    for i in range(len(logZ)):
        for j in range(len(logAge)):
            m = i * len(logAge) + j
            i1, i2 = idx[m], idx[m + 1]
            Ms, first = np.unique(np.log10(data["Mini"][i1:i2]), return_index=True)
            for q, mag in enumerate(mags):
                cube[i, j, :, q] = CubicSpline(Ms, data[mag][i1:i2][first], extrapolate=False)(logM)
    return cube, logZ, logAge, logM, mags


# ----------------------------------------------------------------- truncated Kroupa IMF
def _imf_norms(mass_limit, m_min=0.01):
    n0 = 0.08 ** (1 - 0.3) / 0.08 ** (1 - 1.3)
    n1 = 0.5 ** (1 - 1.3) / 0.5 ** (1 - 2.3) * n0
    c0 = 1 / (2 - 0.3) * (0.08 ** (2 - 0.3) - m_min ** (2 - 0.3))
    c1 = n0 / (2 - 1.3) * (0.5 ** (2 - 1.3) - 0.08 ** (2 - 1.3)) + c0
    return n0, n1, c0, c1, m_min


def _mass_integral_below(m, mass_limit):
    n0, n1, c0, c1, m_min = _imf_norms(mass_limit)
    if m < 0.08:
        return 1 / (2 - 0.3) * (m ** (2 - 0.3) - m_min ** (2 - 0.3))
    if m < 0.5:
        return n0 / (2 - 1.3) * (m ** (2 - 1.3) - 0.08 ** (2 - 1.3)) + c0
    return n1 / (2 - 2.3) * (m ** (2 - 2.3) - 0.5 ** (2 - 2.3)) + c1


def imf_truncated(M, mass_limit=4.0, population_mass=None):
    """
    Kroupa IMF (slopes 0.3, 1.3, 2.3) truncated at mass_limit, normalised so that the
    population has total mass `population_mass` (default: mass_limit, LYRA's unresolved particle mass).
    """
    population_mass = mass_limit if population_mass is None else population_mass
    n0, n1, *_ = _imf_norms(mass_limit)
    M = np.asarray(M, float)
    v = np.where(M <= 0.08, M ** -0.3, np.where(M <= 0.5, n0 * M ** -1.3, n1 * M ** -2.3))
    return v * population_mass / _mass_integral_below(mass_limit, mass_limit)


@lru_cache(maxsize=4)
def integrated_cube(pattern, mass_limit=4.0):
    """Integrated magnitudes of an unresolved population on a (log Z, log age) grid."""
    data, logZ, logAge, mags, idx = read_parsec(pattern)
    cube = np.zeros((len(logZ), len(logAge), len(mags)))
    for i in range(len(logZ)):
        for j in range(len(logAge)):
            m = i * len(logAge) + j
            i1, i2 = idx[m], idx[m + 1]
            Ms, first = np.unique(data["Mini"][i1:i2], return_index=True)
            tr = Ms <= mass_limit
            for q, mag in enumerate(mags):
                L = 10 ** (-0.4 * data[mag][i1:i2][first][tr][:-1])
                cube[i, j, q] = -2.5 * np.log10(simpson(L * imf_truncated(Ms[tr][:-1], mass_limit), x=Ms[tr][:-1]))
    # pad: very low metallicity and oldest ages take the nearest table values
    new = np.zeros((cube.shape[0] + 1, cube.shape[1] + 1, cube.shape[2]))
    new[1:, :-1] = cube
    new[1:, -1] = cube[:, -1]
    new[0, :-1] = cube[0]
    return new, np.append([-10], logZ), np.append(logAge, 10.14), mags


def _pattern(filters, data_pattern):
    if data_pattern is not None:
        return data_pattern
    return config.data_path({"optical": "parsec_optical", "JWST": "parsec_jwst"}[filters])


def magnitudes(Z, age_yr, m_init, filters="optical", mass_limit=4.0, len_mass=200, data_pattern=None,
               clip_to_table=False):
    """
    Absolute magnitude of every star in each band of the isochrone set.

    Z:       metal mass fraction
    age_yr:  stellar age [yr]
    m_init:  initial mass [Msun]; particles with m_init == mass_limit are unresolved populations
    filters: 'optical' (U..K) or 'JWST' (NIRCam), or give data_pattern (glob of isochrone files)
    clip_to_table: if True, resolved stars with metallicity or age outside the isochrone grid
             take the values at the nearest grid edge instead of the fill value. Useful for very
             metal-poor stars below the lowest PARSEC [M/H].
    Returns {band: magnitudes}; stars outside the tables get 35.
    """
    pattern = _pattern(filters, data_pattern)
    cube, logZ, logAge, logM, mags = parsec_cube(pattern, mass_limit, len_mass)
    icube, logZ_i, logAge_i, _ = integrated_cube(pattern, mass_limit)
    with np.errstate(divide="ignore", invalid="ignore"):
        feh = np.maximum(np.log10(np.asarray(Z, float) / PARSEC_SOLAR), -10.0)
        la = np.log10(np.asarray(age_yr, float))
        lm = np.log10(np.asarray(m_init, float))
    unresolved = np.isclose(m_init, mass_limit)
    if clip_to_table:
        feh = np.clip(feh, logZ[0], logZ[-1])
        la = np.clip(la, logAge[0], logAge[-1])
    out = np.full((len(feh), len(mags)), FILL_MAG)
    if (~unresolved).any():
        out[~unresolved] = interpn([logZ, logAge, logM], cube, np.column_stack([feh, la, lm])[~unresolved],
                                   bounds_error=False, fill_value=FILL_MAG)
    if unresolved.any():
        out[unresolved] = interpn([logZ_i, logAge_i], icube, np.column_stack([feh, la])[unresolved],
                                  bounds_error=False, fill_value=FILL_MAG)
    out = np.where(np.isfinite(out), out, FILL_MAG)
    return {mag: out[:, i] for i, mag in enumerate(mags)}


# ----------------------------------------------------------------- luminosities
def jwst_zeropoint(band, path=None):
    """(mean zeropoint, Vega flux [Jy]) of a NIRCam band, e.g. 'F200W'."""
    path = path or config.data_path("jwst_zeropoints")
    filt = np.genfromtxt(path, skip_header=6, dtype=str, unpack=True, delimiter="|")
    names = np.array([n.strip()[6:] for n in filt[1]])
    f = np.genfromtxt(path, skip_header=6, delimiter="|", usecols=[4, 5], unpack=True)
    sel = names == band
    if not sel.any():
        raise KeyError(f"band {band} not in {np.unique(names)}")
    return float(np.mean(f[0][sel])), float(np.mean(f[1][sel]))


def luminosity(mag, zeropoint=4.8, reference=1.0):
    """
    reference * 10^(-0.4 (mag - zeropoint)): Lsun for the V band with zeropoint 4.8,
    or Jy using jwst_zeropoint(band) as (zeropoint, reference). Fill values give NaN.
    """
    mag = np.asarray(mag, float)
    L = reference * 10 ** (-0.4 * (mag - zeropoint))
    return np.where(mag >= FILL_MAG, np.nan, L)


def total_magnitude(mag, mask=None):
    """Combined magnitude of a set of stars (fill values ignored)."""
    mag = np.asarray(mag, float)
    sel = (mag < FILL_MAG) & (True if mask is None else mask)
    if not np.any(sel):
        return np.nan
    return -2.5 * np.log10(np.sum(10 ** (-0.4 * mag[sel])))


def add_magnitude_fields(ds, filters="optical", mass_limit=4.0, len_mass=200, data_pattern=None, clip_to_table=False):
    """
    Register lazy per-star magnitude fields '<band>' (e.g. 'Vmag') on ds.data['PartType4'].
    Requires GFM_Metallicity, GFM_InitialMass and the lyratools 'Age' field (lyratools.add_fields).
    """
    from .units import magnitude as _mag, require_scida
    require_scida("photometry.add_magnitude_fields")
    import dask.array as da
    pattern = _pattern(filters, data_pattern)
    _, _, _, _, mags = parsec_cube(pattern, mass_limit, len_mass)

    def compute_block(Z, age_myr, m, band):
        return magnitudes(Z, age_myr * 1e6, m, mass_limit=mass_limit, len_mass=len_mass, data_pattern=pattern,
                          clip_to_table=clip_to_table)[band]

    for band in mags:
        def field(arrs, snap=None, band=band, **kwargs):
            Z = _mag(arrs["GFM_Metallicity"])
            age = _mag(arrs["Age"].to("Myr"))
            m = _mag(arrs["GFM_InitialMass"].to("Msun"))
            return da.map_blocks(compute_block, Z, age, m, band, dtype=float)
        ds.data["PartType4"].register_field(name=band, description=f"absolute {band} ({filters})")(field)
    return mags
