"""
Supernova and AGB energetics and yields from the Sukhbold et al. (2016) and
Karakas tables used by LYRA, plus helpers for LYRA's SNTime convention.

Masses are initial stellar masses in Msun; times in Myr.
"""
from functools import lru_cache

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import config
from .constants import SN_MIN_MASS, SNTIME_NONE, SNTIME_UNIT_MYR, ZSOLAR


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _sukhbold(path):
    data = np.genfromtxt(path, unpack=True, skip_header=1)
    # rows: Minitial, ESN, 0, H, He, Li, ... (element masses for winds and SN ejecta)
    t = dict(Minitial=data[0], ESN=data[1])
    t["MSN"] = np.sum(data[3:86], axis=0)
    t["Mwinds"] = np.sum(data[87:], axis=0)
    t["Meject"] = t["Mwinds"] + t["MSN"]
    t["Zwinds"] = np.sum(data[5:86], axis=0)
    t["ZSN"] = np.sum(data[89:], axis=0)
    t["Metal"] = t["Zwinds"] + t["ZSN"]
    return t


def sukhbold_table():
    return _sukhbold(config.data_path("sukhbold"))


@lru_cache(maxsize=None)
def _karakas(path):
    kar = np.genfromtxt(path, unpack=True, skip_header=1)
    # 30 masses x 6 metallicities; rows 123: wind yields per element (A = 1 is hydrogen)
    masses = kar[1][:30]
    metallicities = np.unique(kar[0] / ZSOLAR)
    mej = np.reshape(np.sum(kar[123:], axis=0), (6, 30))
    zej = np.reshape(np.sum(kar[125:], axis=0), (6, 30))
    grid = (metallicities, masses)
    return dict(mass_return=RegularGridInterpolator(grid, mej),
                metal_return=RegularGridInterpolator(grid, zej),
                metal_lim=(metallicities[0], metallicities[-1]),
                mass_lim=(masses[0], masses[-1]))


def karakas_table():
    return _karakas(config.data_path("karakas"))


# --------------------------------------------------------------------------
# SN energies and yields
# --------------------------------------------------------------------------
def sn_energy(m_init):
    """SN energy [erg] for initial mass m_init [Msun]; zero below 8 Msun."""
    t = sukhbold_table()
    m = np.asarray(m_init, dtype=float)
    e = np.interp(m, t["Minitial"], t["ESN"])
    return np.where(m < SN_MIN_MASS, 0.0, e)


def _nearest_scaled(m, table_m, table_val):
    m = np.atleast_1d(np.asarray(m, dtype=float))
    idx = np.clip(np.searchsorted(table_m, m), 1, len(table_m) - 1)
    lower_closer = (table_m[idx] - m) > (m - table_m[idx - 1])
    idx = np.where(lower_closer, idx - 1, idx)
    return table_val[idx] * m / table_m[idx]


def sn_mass_return(m_init, mode="all"):
    """Mass [Msun] returned by a massive star: mode 'all', 'winds' or 'SN'."""
    t = sukhbold_table()
    key = {"all": "Meject", "winds": "Mwinds", "SN": "MSN"}[mode]
    return _nearest_scaled(m_init, t["Minitial"], t[key])


def sn_metal_return(m_init, mode="all"):
    """Metal mass [Msun] returned by a massive star: mode 'all', 'winds' or 'SN'."""
    t = sukhbold_table()
    key = {"all": "Metal", "winds": "Zwinds", "SN": "ZSN"}[mode]
    return _nearest_scaled(m_init, t["Minitial"], t[key])


def agb_returns(m_init, Z):
    """
    Mass and metal mass [Msun] returned by AGB stars of initial mass m_init [Msun]
    and metal mass fraction Z, interpolated in the Karakas tables (clipped to the table range).
    """
    k = karakas_table()
    zz = np.clip(np.asarray(Z, dtype=float) / ZSOLAR, *k["metal_lim"])
    mm = np.clip(np.asarray(m_init, dtype=float), *k["mass_lim"])
    pts = np.column_stack([np.atleast_1d(zz), np.atleast_1d(mm)])
    return k["mass_return"](pts), k["metal_return"](pts)


# --------------------------------------------------------------------------
# LYRA SNTime convention
# --------------------------------------------------------------------------
def sn_state(sntime):
    """0 = never a SN, 1 = SN still to come, 2 = SN has exploded."""
    s = np.asarray(sntime, dtype=float)
    return np.where(s == SNTIME_NONE, 0, np.where(s > 0, 1, 2))


def sntime_convention(sntime, tform_code, time_code, rtol=1e-5):
    """
    Which SNTime convention a snapshot uses:
      'lifetime_gyr' (cosmological runs): -1 never, +lifetime [Gyr] alive, -lifetime [Gyr] dead;
      'absolute' (isolated LYRA runs): -1 never, +time of death [code time] alive (> Time),
                 -time of death [code time] dead (>= the formation time).
    'absolute' only if every alive star has SNTime >= Time and every dead star -SNTime >= its formation
    time (both in code time units, relative tolerance rtol); otherwise, and with no alive or dead stars,
    'lifetime_gyr'. Only meaningful for isolated runs, whose formation times are code times.
    """
    s = np.asarray(sntime, dtype=float)
    t = np.asarray(tform_code, dtype=float)
    alive, dead = s > 0, (s < 0) & (s != SNTIME_NONE)
    if not (alive.any() or dead.any()):
        return "lifetime_gyr"
    ok_alive = bool(np.all(s[alive] >= time_code * (1 - rtol))) if alive.any() else True
    ok_dead = bool(np.all(-s[dead] >= np.broadcast_to(t, s.shape)[dead] * (1 - rtol))) if dead.any() else True
    return "absolute" if ok_alive and ok_dead else "lifetime_gyr"


def sntime_as_lifetime(sntime, tform_code, myr_per_code, unit_myr=SNTIME_UNIT_MYR):
    """
    Absolute-convention SNTime (isolated runs, code times of death) converted to the lifetime convention
    (+-lifetime in units of unit_myr, Gyr by default; -1 kept), so that sn_state, death_time, died_within
    and sn_energies apply unchanged. Lifetimes are floored at a tiny positive value to keep the sign.
    """
    s = np.asarray(sntime, dtype=float)
    life = np.maximum((np.abs(s) - np.asarray(tform_code, dtype=float)) * myr_per_code / unit_myr, 1e-12)
    out = np.where(s > 0, life, -life)
    return np.where((s == SNTIME_NONE) | ~np.isfinite(s) | (s == 0), s, out)


def death_time(tform_myr, sntime, unit_myr=SNTIME_UNIT_MYR):
    """Absolute cosmic time [Myr] at which a star died (SN or end of life), NaN if still alive."""
    s = np.asarray(sntime, dtype=float)
    dead = (s < 0) & (s != SNTIME_NONE)
    return np.where(dead, np.asarray(tform_myr) + np.abs(s) * unit_myr, np.nan)


def died_within(tform_myr, sntime, m_init, t_now, dt, massive=True, unit_myr=SNTIME_UNIT_MYR):
    """
    Stars that died in the last dt [Myr] before t_now [Myr]: SN progenitors (massive=True,
    m_init > 8 Msun) or lower-mass stars ending as AGB (massive=False).
    """
    td = death_time(tform_myr, sntime, unit_myr)
    mass_sel = np.asarray(m_init) > SN_MIN_MASS if massive else np.asarray(m_init) < SN_MIN_MASS
    with np.errstate(invalid="ignore"):
        return mass_sel & (td > t_now - dt) & (td <= t_now)


def dead_massive(m_init, m_now):
    """Massive stars that have lost mass, i.e. already returned their ejecta."""
    return (np.asarray(m_init) > SN_MIN_MASS) & (np.asarray(m_init) > np.asarray(m_now))


def dead_agb(m_init, m_now):
    return (np.asarray(m_init) < SN_MIN_MASS) & (np.asarray(m_init) > np.asarray(m_now))


def sn_energies(m_init, sntime, failed_energy=1e49):
    """Energy [erg] injected by each exploded SN; failed SNe (zero table energy) get `failed_energy`."""
    m = np.asarray(m_init, dtype=float)
    exploded = (sn_state(sntime) == 2) & (m > SN_MIN_MASS)
    e = np.zeros(m.shape)
    if exploded.any():
        ee = sn_energy(m[exploded])
        e[exploded] = np.where(ee == 0, failed_energy, ee)
    return e
