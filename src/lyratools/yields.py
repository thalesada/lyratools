"""
Post-processing correction of LYRA's carbon, nitrogen and oxygen abundances for
metallicity-dependent stellar yields.

LYRA returns massive-star ejecta from the solar-metallicity Sukhbold et al. (2016) models
for every star, and AGB ejecta from Karakas (2010) (metallicity-dependent). Two problems follow:

  * carbon: ~90% of massive-star carbon comes from the strong, carbon-rich winds of solar-
    metallicity stars above ~40 Msun; metal-poor stars have far weaker winds;
  * nitrogen: the Karakas models produce large amounts of primary nitrogen by hot-bottom
    burning in 4-8 Msun AGB stars; other AGB models (e.g. FRUITY) produce far less.

The correction uses that individual elements are passive tracers and that oxygen traces the
amount of massive-star ejecta each gas cell (or star, through its birth gas) received:

    u        = O_massive / Y_O^sim                 "massive-star populations' worth" of ejecta
    X_mass   = u * Y_X^sim                         massive-star part of element X (as simulated)
    X_AGB    = X - X_mass                          AGB part (residual)
    X_corr   = u * Y_X(Z_eff) + X_AGB * R_X(Z, t)

with Y_X the IMF-averaged massive-star yields (simulated: solar; corrected: at the metallicity
Z_eff of the polluting stars) and R_X the ratio of IMF-averaged AGB yields, comparison set / Karakas.

Metallicity-dependent massive-star yields are built from LYRA's own Sukhbold table:
    y_X(m, Z) = [y_X^SN - (1 - z) X_X,birth M_ej^SN] + z^alpha y_X^wind,       z = Z/Zsun
i.e. the solar birth composition carried by the supernova ejecta scales with Z, the wind
ejecta scale with the mass-loss rate (Mdot ~ Z^alpha, alpha ~ 0.7-0.85), and, optionally, the
net supernova nitrogen (secondary in non-rotating models) scales with Z. Limitation: weaker
winds also change core masses, supernova yields and explodability; this is not modelled.

Typical use:
    corr = YieldCorrection.from_files()               # LYRA tables via lyratools.set_data_dir
    out  = corr.correct(X, z_eff=..., agb_age_myr=...)   # X: dict of C, N, O (and H) mass fractions
    log_NO = number_ratio(out["N"], out["O"], "N", "O")

All masses in Msun, times in Myr, metallicities Z/Zsun unless stated (Karakas/FRUITY grids use absolute Z).
"""
import os
from functools import lru_cache

import numpy as np

from . import config

ATOMIC_MASS = {"H": 1.008, "He": 4.0026, "C": 12.011, "N": 14.007, "O": 15.999}
# Asplund et al. (2009) present-day solar mass fractions
SOLAR_MASS_FRACTION = {"H": 0.7381, "C": 2.39e-3, "N": 6.99e-4, "O": 5.78e-3}
Z_SUN_ABS = 0.0134
ELEMENTS = ("C", "N", "O")
FRUITY_DEFAULT = os.path.join(os.path.dirname(__file__), "data", "fruity_agb_CNO.txt")


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------
def number_ratio(x_a, x_b, el_a, el_b):
    """log10 of the number ratio a/b from mass fractions (e.g. log N/O as observers quote it)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10((np.asarray(x_a, float) / ATOMIC_MASS[el_a]) / (np.asarray(x_b, float) / ATOMIC_MASS[el_b]))


def twelve_log_oh(x_o, x_h):
    """12 + log(O/H) from oxygen and hydrogen mass fractions."""
    return 12 + number_ratio(x_o, x_h, "O", "H")


def oxygen_metallicity(x_o):
    """Metallicity in solar units from oxygen alone (immune to the C and N excess)."""
    return np.asarray(x_o, float) / SOLAR_MASS_FRACTION["O"]


def kroupa_imf(m):
    """Kroupa (2001) IMF dN/dm (unnormalised)."""
    m = np.asarray(m, float)
    return np.where(m < 0.08, (m / 0.08) ** -0.3 * (0.08 / 0.5) ** -1.3,
                    np.where(m < 0.5, (m / 0.5) ** -1.3, (m / 0.5) ** -2.3))


def imf_normalisation(imf=kroupa_imf, m_min=0.08, m_max=100.0):
    """Mass of stars formed per unit IMF normalisation (to express yields per Msun formed)."""
    m = np.geomspace(m_min, m_max, 4000)
    return np.trapezoid(m * imf(m), m)


def lifetime_myr(m, z_abs=Z_SUN_ABS):
    """Stellar lifetime [Myr], Raiteri, Villata & Navarro (1996) fit (0.6-120 Msun, Z = 7e-5-0.03)."""
    lz = np.log10(np.clip(z_abs, 7e-5, 0.03))
    a0 = 10.13 + 0.07547 * lz - 0.008084 * lz ** 2
    a1 = -4.424 - 0.7939 * lz - 0.1187 * lz ** 2
    a2 = 1.262 + 0.3385 * lz + 0.05417 * lz ** 2
    lm = np.log10(np.asarray(m, float))
    return 10 ** (a0 + a1 * lm + a2 * lm ** 2) / 1e6


def _imf_weights(m, imf, m_lo, m_hi):
    """Trapezoid weights of a tabulated mass grid for integrating f(m) * imf(m) dm over [m_lo, m_hi]."""
    m = np.asarray(m, float)
    edges = np.concatenate([[m[0]], 0.5 * (m[1:] + m[:-1]), [m[-1]]])
    lo, hi = np.clip(edges[:-1], m_lo, m_hi), np.clip(edges[1:], m_lo, m_hi)
    return imf(m) * np.maximum(hi - lo, 0.0)


# --------------------------------------------------------------------------------------------
# massive stars: LYRA's Sukhbold et al. (2016) table, made metallicity dependent
# --------------------------------------------------------------------------------------------
class MassiveStarYields:
    """
    Sukhbold et al. (2016) ejecta as stored in LYRA's StellarEvolution_Sukhbold2015.txt:
    per mass, a supernova block and a wind block of element masses (hydrogen first).
    """

    def __init__(self, masses, e_sn, sn, wind, birth):
        self.m, self.e_sn = masses, e_sn
        self.sn, self.wind = sn, wind                 # dict element -> array over masses
        self.mej_sn = sn["_total"]
        self.birth = birth                            # birth mass fractions of the (solar) models

    @classmethod
    def from_file(cls, path=None):
        path = path or config.data_path("sukhbold")
        d = np.loadtxt(path, skiprows=1)
        snb, wb = d[:, 3:86], d[:, 87:170]
        idx = {"H": 0, "He": 1, "C": 5, "N": 6, "O": 7}
        sn = {k: snb[:, i] for k, i in idx.items()}
        wind = {k: wb[:, i] for k, i in idx.items()}
        sn["_total"], wind["_total"] = snb.sum(1), wb.sum(1)
        # birth composition: oxygen from the wind of the lowest-mass model (unaffected by
        # dredge-up), C and N from solar ratios relative to O
        x_o = wb[0, 7] / wb[0].sum()
        birth = {"O": x_o, "C": x_o * SOLAR_MASS_FRACTION["C"] / SOLAR_MASS_FRACTION["O"],
                 "N": x_o * SOLAR_MASS_FRACTION["N"] / SOLAR_MASS_FRACTION["O"]}
        return cls(d[:, 0], d[:, 1], sn, wind, birth)

    def per_star(self, element, z=1.0, alpha=0.8, secondary_n=True):
        """
        Ejected mass of `element` per star for metallicity z = Z/Zsun.
        z = 1 reproduces the table (what LYRA uses) for any alpha.
        """
        z = float(z)
        sn = self.sn[element] - (1.0 - z) * self.birth[element] * self.mej_sn
        if element == "N" and secondary_n:
            sn = z * self.sn["N"]                     # birth N and secondary N both scale with Z
        y = np.maximum(sn, 0.0) + z ** alpha * self.wind[element]
        return y

    def imf_yield(self, element, z=1.0, alpha=0.8, secondary_n=True, imf=kroupa_imf, m_range=(8.0, 100.0),
                  m_imf=(0.08, 100.0)):
        """IMF-averaged ejected mass of `element` per Msun of stars formed."""
        w = _imf_weights(self.m, imf, *m_range)
        return np.sum(w * self.per_star(element, z, alpha, secondary_n)) / imf_normalisation(imf, *m_imf)


# --------------------------------------------------------------------------------------------
# massive stars: Limongi & Chieffi (2018) + Boccioli & Roberti (2026), in yields_lc18.py (later release);
# imported at the end of this module, LC18MassiveYields is None when the module is not installed.
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# AGB stars: Karakas (LYRA) and a comparison grid (FRUITY by default)
# --------------------------------------------------------------------------------------------
class AGBYields:
    """Grid of AGB ejecta (total ejected element masses) over initial mass and absolute metallicity."""

    def __init__(self, z_grid, m_grid, table, name):
        self.z, self.m, self.table, self.name = z_grid, m_grid, table, name   # table[el]: (nZ, nM)

    @classmethod
    def from_karakas(cls, path=None):
        """LYRA's StellarEvolution_Karakas.txt (30 masses x 6 metallicities, ejecta from column 123 on)."""
        path = path or config.data_path("karakas")
        d = np.loadtxt(path, skiprows=1)
        z, m = np.unique(d[:, 0]), np.unique(d[:, 1])
        col = {"H": 123, "C": 128, "N": 129, "O": 130}
        tab = {}
        for el, c in col.items():
            t = np.zeros((len(z), len(m)))
            for row in d:
                t[np.searchsorted(z, row[0]), np.searchsorted(m, row[1])] = row[c]
            tab[el] = t
        return cls(z, m, tab, "Karakas")

    @classmethod
    def from_fruity(cls, path=None):
        """FRUITY (Cristallo et al. 2011, 2015) table shipped with lyratools (columns Z M Mej H He C N O)."""
        d = np.loadtxt(path or FRUITY_DEFAULT)
        z, m = np.unique(d[:, 0]), np.unique(d[:, 1])
        col = {"H": 3, "C": 5, "N": 6, "O": 7}
        tab = {}
        for el, c in col.items():
            t = np.full((len(z), len(m)), np.nan)
            for row in d:
                t[np.searchsorted(z, row[0]), np.searchsorted(m, row[1])] = row[c]
            tab[el] = t
        return cls(z, m, tab, "FRUITY")

    def per_star(self, element, z_abs):
        """Ejecta per star on the mass grid, linearly interpolated in log Z (clamped to the grid)."""
        lz = np.log10(np.clip(z_abs, self.z[0], self.z[-1]))
        lgrid = np.log10(self.z)
        i = int(np.clip(np.searchsorted(lgrid, lz) - 1, 0, len(lgrid) - 2))
        f = (lz - lgrid[i]) / (lgrid[i + 1] - lgrid[i])
        return (1 - f) * self.table[element][i] + f * self.table[element][i + 1]


class AGBYieldRatio:
    """
    IMF-averaged AGB ejecta of a comparison grid relative to Karakas (LYRA), for stars that have
    died by a given age. high_mass sets the treatment of masses above the comparison grid
    (FRUITY stops at 6 Msun): 'karakas' keeps Karakas there; 'scaled' scales the comparison
    grid's most massive model by ejected hydrogen.
    """

    def __init__(self, sim, alt, imf=kroupa_imf, m_range=(0.75, 8.0), high_mass="karakas"):
        self.sim, self.alt, self.imf, self.m_range, self.high_mass = sim, alt, imf, m_range, high_mass

    def _grid(self, which, element, z_abs):
        """Mass grid and per-star ejecta: Karakas ('sim'), or the comparison grid completed with
        Karakas below its lowest mass and, above its highest mass, Karakas or scaled ('alt')."""
        sim = self.sim
        if which == "sim":
            return sim.m, sim.per_star(element, z_abs)
        alt = self.alt
        m_a, y_a = alt.m, alt.per_star(element, z_abs)
        y_s = sim.per_star(element, z_abs)
        lo, hi = sim.m < m_a[0], sim.m > m_a[-1]
        if self.high_mass == "karakas":
            y_hi = y_s[hi]
        else:   # scale the most massive comparison model by the ejected hydrogen of the Karakas models
            h = sim.per_star("H", z_abs)
            y_hi = y_a[-1] * h[hi] / h[np.argmin(abs(sim.m - m_a[-1]))]
        m = np.concatenate([sim.m[lo], m_a, sim.m[hi]])
        y = np.concatenate([y_s[lo], y_a, y_hi])
        return m, y

    def _integrate(self, which, element, z_abs, age_myr):
        m, y = self._grid(which, element, z_abs)
        w = _imf_weights(m, self.imf, *self.m_range)
        if age_myr is not None:
            w = w * (lifetime_myr(m, z_abs) <= age_myr)
        return np.nansum(w * y)

    def ratio(self, element, z_abs, age_myr=None):
        """Y_alt / Y_Karakas for `element` at absolute metallicity z_abs (all AGB masses if age is None)."""
        y_sim = self._integrate("sim", element, z_abs, age_myr)
        y_alt = self._integrate("alt", element, z_abs, age_myr)
        return y_alt / y_sim if y_sim > 0 else 1.0

    def imf_yield(self, element, z_abs, age_myr=None, which="sim", m_imf=(0.08, 100.0)):
        """IMF-averaged AGB ejecta per Msun formed, 'sim' (Karakas) or 'alt' (comparison grid)."""
        return self._integrate(which, element, z_abs, age_myr) / imf_normalisation(self.imf, *m_imf)


# --------------------------------------------------------------------------------------------
# the correction
# --------------------------------------------------------------------------------------------
class YieldCorrection:
    """
    Correct simulated C, N, O mass fractions (gas cells or stellar birth abundances).

    alpha        wind mass-loss metallicity exponent (Mdot ~ Z^alpha); 0 keeps solar winds
    secondary_n  scale the massive-star supernova nitrogen with Z (non-rotating models)
    massive_range, agb_range: mass ranges [Msun] treated as massive / AGB stars
    high_mass    treatment of AGB masses above the comparison grid ('karakas' or 'scaled')
    high_mass_table  corrected massive-star yields: 'sukhbold' (default; LYRA's Sukhbold table with
                 the metallicity scalings above) or 'LC18' (LC18MassiveYields: LC18 + BR26 tables
                 mixed with the Prantzos et al. 2018 rotation weights; alpha and secondary_n are
                 then not used). The simulated yields are always LYRA's Sukhbold table.
    lc18         LC18MassiveYields instance (default: read from the data files when needed)
    """

    def __init__(self, massive, agb_sim, agb_alt, alpha=0.8, secondary_n=True, imf=kroupa_imf,
                 massive_range=(8.0, 100.0), agb_range=(0.75, 8.0), high_mass="karakas",
                 high_mass_table="sukhbold", lc18=None):
        self.massive, self.alpha, self.secondary_n = massive, alpha, secondary_n
        self.imf, self.massive_range = imf, massive_range
        self.agb = AGBYieldRatio(agb_sim, agb_alt, imf, agb_range, high_mass)
        table = str(high_mass_table).lower()
        if table not in ("sukhbold", "lc18"):
            raise ValueError(f"high_mass_table must be 'sukhbold' or 'LC18', not {high_mass_table!r}")
        self.high_mass_table = table
        if table == "lc18" and lc18 is None and LC18MassiveYields is None:
            raise ValueError("high_mass_table='LC18' needs lyratools.yields_lc18, which is not part of this release")
        self.lc18 = (lc18 or LC18MassiveYields.from_files()) if table == "lc18" else lc18

    @classmethod
    def from_files(cls, sukhbold=None, karakas=None, fruity=None, lc18_tables=None, **kwargs):
        """lc18_tables: the three LC18 BR26 table paths (v = 0, 150, 300), read if high_mass_table='LC18'
        (default: the 'lc18_br26_v*' files of the data directory)."""
        if (str(kwargs.get("high_mass_table", "sukhbold")).lower() == "lc18" and kwargs.get("lc18") is None
                and LC18MassiveYields is not None):
            kwargs["lc18"] = LC18MassiveYields.from_files(lc18_tables)
        return cls(MassiveStarYields.from_file(sukhbold), AGBYields.from_karakas(karakas),
                   AGBYields.from_fruity(fruity), **kwargs)

    # ---- IMF-averaged yields
    @lru_cache(maxsize=4096)
    def _massive(self, element, z, alpha, secondary_n):
        return self.massive.imf_yield(element, z, alpha, secondary_n, self.imf, self.massive_range)

    @lru_cache(maxsize=4096)
    def _massive_lc18(self, element, z):
        return self.lc18.imf_yield(element, z, self.imf, self.massive_range)

    def sim_massive_yield(self, element):
        """IMF-averaged massive-star yield per Msun formed as simulated (LYRA's solar Sukhbold table)."""
        return self._massive(element, 1.0, self.alpha, self.secondary_n)

    def massive_yield(self, element, z=1.0):
        """IMF-averaged corrected massive-star yield per Msun formed at z = Z/Zsun (oxygen based).
        With the default Sukhbold table, z = 1 is what LYRA uses (= sim_massive_yield)."""
        z = round(float(z), 6)
        if self.high_mass_table == "lc18":
            return self._massive_lc18(element, z)
        return self._massive(element, z, self.alpha, self.secondary_n)

    def population_ratios(self, z, age_myr, corrected=True):
        """
        log C/O and log N/O (number) of the ejecta of a single population of metallicity z (solar
        units) after age_myr: simulated (LYRA yields) or corrected.
        """
        z_abs = z * Z_SUN_ABS
        out = {}
        for el in ELEMENTS:
            m = self.massive_yield(el, z) if corrected else self.sim_massive_yield(el)
            a = self.agb.imf_yield(el, z_abs, age_myr, "alt" if corrected else "sim")
            out[el] = m + a
        return {"C/O": number_ratio(out["C"], out["O"], "C", "O"), "N/O": number_ratio(out["N"], out["O"], "N", "O")}

    # ---- per cell / per star
    def correct(self, X, z_eff=None, z_agb=None, agb_age_myr=None, o_agb_share=None):
        """
        Correct mass fractions X = {'C':, 'N':, 'O':} (arrays or scalars; 'H' optional, passed through).

        z_eff       metallicity (Z/Zsun) of the massive stars that enriched each cell; default: the
                    cell's own oxygen-based metallicity (an upper bound, giving the most secondary
                    N and wind C, i.e. upper limits on N/O and C/O)
        z_agb       metallicity of the AGB polluters (default z_eff)
        agb_age_myr age of the polluting population(s); sets which AGB masses have contributed
                    (None: all)
        o_agb_share fraction of each cell's oxygen from AGB stars (default: from the population
                    model at z_eff and agb_age_myr; small)

        Returns a dict with corrected 'C', 'N', 'O' (and 'H' if given) plus diagnostics:
        'agb_share_C', 'agb_share_N' (simulated AGB fractions) and 'negative_C', 'negative_N'
        (cells whose simulated element is below the massive-star expectation: decomposition fails).
        """
        C, N, O = (np.asarray(X[k], float) for k in ELEMENTS)
        z_eff = oxygen_metallicity(O) if z_eff is None else np.broadcast_to(np.asarray(z_eff, float), O.shape)
        z_agb = z_eff if z_agb is None else np.broadcast_to(np.asarray(z_agb, float), O.shape)

        # evaluate the IMF-averaged yields on a grid in log z and interpolate (fast for many cells)
        lz_grid = np.linspace(-4.0, 0.5, 46)
        zg = 10 ** lz_grid
        lze = np.log10(np.clip(z_eff, zg[0], zg[-1]))
        lza = np.log10(np.clip(z_agb, zg[0], zg[-1]))
        ysim = {el: self.sim_massive_yield(el) for el in ELEMENTS}
        ycor = {el: np.interp(lze, lz_grid, [self.massive_yield(el, z) for z in zg]) for el in ELEMENTS}
        ratio = {el: np.interp(lza, lz_grid, [self.agb.ratio(el, z * Z_SUN_ABS, agb_age_myr) for z in zg]) for el in ELEMENTS}
        if o_agb_share is None:
            o_agb = np.array([self.agb.imf_yield("O", z * Z_SUN_ABS, agb_age_myr, "sim") for z in zg])
            share = o_agb / (o_agb + ysim["O"])
            o_agb_share = np.interp(lza, lz_grid, share)

        O_massive = O * (1.0 - o_agb_share)
        u = O_massive / ysim["O"]
        out = {"O": u * ycor["O"] + O * o_agb_share * ratio["O"]}
        for el, x in (("C", C), ("N", N)):
            x_mass = u * ysim[el]
            x_agb = x - x_mass
            out[f"negative_{el}"] = x_agb < 0
            out[f"agb_share_{el}"] = np.clip(x_agb / np.where(x > 0, x, np.inf), 0, 1)
            out[el] = u * ycor[el] + np.maximum(x_agb, 0.0) * ratio[el]
        if "H" in X:
            out["H"] = np.asarray(X["H"], float)
        return out


# --------------------------------------------------------------------------------------------
# metallicity of the polluting massive stars
# --------------------------------------------------------------------------------------------
def polluter_metallicity(cell_pos, star_pos, star_z, star_m_init, star_t_death, t_now, window_myr=50.0,
                         radius=0.3, massive_yields=None, alpha=0.8, min_mass=8.0, default=None):
    """
    Oxygen-weighted birth metallicity (Z/Zsun, oxygen based) of the massive stars that died within
    `window_myr` before t_now and within `radius` of each cell (positions and radius in the same
    unit, e.g. kpc). Weights: each star's oxygen ejecta (from LYRA's Sukhbold table at its own
    metallicity if massive_yields is given, else its initial mass). Cells without such stars get
    `default` (e.g. the cell's own metallicity), or NaN.
    """
    from scipy.spatial import cKDTree
    star_z = np.asarray(star_z, float)
    sel = (np.asarray(star_m_init) >= min_mass) & np.isfinite(star_t_death) \
        & (np.asarray(star_t_death) <= t_now) & (np.asarray(star_t_death) > t_now - window_myr)
    if not sel.any():
        return np.full(len(cell_pos), np.nan) if default is None else np.asarray(default, float)
    sp, sz, sm = np.asarray(star_pos)[sel], star_z[sel], np.asarray(star_m_init)[sel]
    if massive_yields is not None:
        w = np.array([np.interp(m, massive_yields.m, massive_yields.per_star("O", z, alpha)) for m, z in zip(sm, sz)])
    else:
        w = sm
    tree = cKDTree(sp)
    neigh = tree.query_ball_point(np.asarray(cell_pos), r=radius)
    out = np.full(len(cell_pos), np.nan)
    for i, idx in enumerate(neigh):
        if idx:
            out[i] = np.average(sz[idx], weights=w[idx])
    if default is not None:
        out = np.where(np.isfinite(out), out, np.asarray(default, float))
    return out


def plateau_diagnostic(x_o, x_n, x_c=None, x_h=None, bins=None):
    """
    Median log N/O (and log C/O) in bins of 12+log(O/H) (or of log Z_O if x_h is None).
    In gas enriched only by massive stars, log N/O and log C/O should sit on a plateau; a rise
    with metallicity marks AGB and secondary contributions.
    """
    x = twelve_log_oh(x_o, x_h) if x_h is not None else np.log10(oxygen_metallicity(x_o))
    bins = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 12) if bins is None else bins
    idx = np.digitize(x, bins) - 1
    no = number_ratio(x_n, x_o, "N", "O")
    co = number_ratio(x_c, x_o, "C", "O") if x_c is not None else None
    rows = []
    for i in range(len(bins) - 1):
        s = idx == i
        if s.sum() < 5:
            continue
        r = {"x": 0.5 * (bins[i] + bins[i + 1]), "n": int(s.sum()), "log_NO": np.nanmedian(no[s])}
        if co is not None:
            r["log_CO"] = np.nanmedian(co[s])
        rows.append(r)
    return rows


# --------------------------------------------------------------------------------------------
# convenience for lyratools particle sets (GFM_Metals columns: H, He, C, N, O, ...)
# --------------------------------------------------------------------------------------------
def correct_particles(parts, corr, z_eff=None, agb_age_myr=None, metals_field="GFM_Metals", **kwargs):
    """
    Apply the correction to a lyratools Particles set (gas or stars) using its GFM_Metals field.
    Returns the corrected dict plus 'log_CO', 'log_NO', '12logOH' (corrected) and the same with
    suffix '_sim' (uncorrected), all as number ratios.
    """
    from .constants import ELEMENTS as COL
    from .units import as_array
    met = as_array(parts[metals_field])
    X = {el: met[:, COL[el]] for el in ("H", "C", "N", "O")}
    out = corr.correct(X, z_eff=z_eff, agb_age_myr=agb_age_myr, **kwargs)
    out["log_CO"] = number_ratio(out["C"], out["O"], "C", "O")
    out["log_NO"] = number_ratio(out["N"], out["O"], "N", "O")
    out["12logOH"] = twelve_log_oh(out["O"], X["H"])
    out["log_CO_sim"] = number_ratio(X["C"], X["O"], "C", "O")
    out["log_NO_sim"] = number_ratio(X["N"], X["O"], "N", "O")
    out["12logOH_sim"] = twelve_log_oh(X["O"], X["H"])
    return out


# later-release option (yields_lc18.py); imported last because it uses the helpers above
try:
    from .yields_lc18 import LC18_VELOCITIES, PRANTZOS18_WEIGHTS, LC18MassiveYields
except ModuleNotFoundError as _e:          # module left out of this release; any other error is real
    if _e.name != __name__ + "_lc18":
        raise
    LC18_VELOCITIES, PRANTZOS18_WEIGHTS, LC18MassiveYields = (), None, None
