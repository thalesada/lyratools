"""
Dark matter halo structure and core formation.

Profiles and fits (radii in kpc, densities in Msun/kpc^3, masses in Msun, velocities in km/s):
    radial_profile           density, enclosed mass, counts in logarithmic shells
    power_radius             numerical convergence radius (Power et al. 2003)
    local_slope, inner_slope logarithmic density slope (local, or fitted between two radii)
    fit_profile              NFW, generalised NFW (free inner slope), coreNFW (Read et al. 2016), Burkert
    dicintio14_shape         (alpha, beta, gamma) predicted from M*/Mhalo (Di Cintio et al. 2014)
Core energetics:
    potential_energy, core_formation_energy   energy needed to turn the fitted cusp into the core
Kinematics and heating:
    velocity_profiles        sigma_r, sigma_t, anisotropy beta(r)
    spherical_potential      Phi(r) from all matter, spherical approximation
    potential_on_grid        the same from per-type histograms (no particle copies; for large haloes)
    apocentres               orbit apocentres of particles in that potential
    match_ids                indices of common particle IDs between snapshots
    find_ids                 positions of a small set of IDs in a very large ID array (chunked)
"""
import numpy as np
from scipy.optimize import curve_fit

G_KPC = 4.30091e-6          # kpc (km/s)^2 / Msun


# --------------------------------------------------------------------------------------------- profiles
def radial_profile(r, mass, bins):
    """Shell density [Msun/kpc^3], enclosed mass at the outer bin edges, particle counts, bin centres (geometric)."""
    r, mass = np.asarray(r, float), np.asarray(mass, float)
    bins = np.asarray(bins, float)
    m_shell = np.histogram(r, bins=bins, weights=mass)[0]
    n_shell = np.histogram(r, bins=bins)[0]
    vol = 4.0 / 3.0 * np.pi * np.diff(bins ** 3)
    m_enc = np.cumsum(m_shell) + mass[r < bins[0]].sum()
    return dict(r=np.sqrt(bins[1:] * bins[:-1]), rho=m_shell / vol, m_enc=m_enc, r_edge=bins[1:], n=n_shell)


def power_radius(r, mass, rho_crit, kappa=0.6, r_min=1e-3, r_max=None, nbins=200):
    """
    Convergence radius of Power et al. (2003): smallest radius beyond which the two-body relaxation
    time exceeds kappa times the circular orbit time at R200,
        kappa(r) = sqrt(200)/8 * N(<r)/ln N(<r) * (mean density(<r) / rho_crit)^(-1/2).
    rho_crit in Msun/kpc^3 at the snapshot redshift; mass may be a scalar for equal-mass particles.
    Returns the radius in kpc.
    """
    r = np.asarray(r)
    r_max = float(r.max()) if r_max is None else r_max
    grid = np.geomspace(r_min, r_max, nbins)
    rs = np.sort(r)
    N = np.searchsorted(rs, grid)
    if np.ndim(mass) == 0:                    # equal-mass particles (dark matter): no index sort needed
        M = N * float(mass)
    else:
        menc = np.cumsum(np.asarray(mass, float)[np.argsort(r)])
        M = np.where(N > 0, menc[np.maximum(N - 1, 0)], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_bar = M / (4.0 / 3.0 * np.pi * grid ** 3)
        k = np.sqrt(200.0) / 8.0 * N / np.log(N) * np.sqrt(rho_crit / rho_bar)
    ok = np.isfinite(k) & (N > 1) & (k >= kappa)
    bad = np.where(~ok)[0]
    if len(bad) == 0:
        return grid[0]
    if bad[-1] == len(grid) - 1:
        return np.nan
    return grid[bad[-1] + 1]


def local_slope(r, rho):
    """d log rho / d log r (central differences)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.gradient(np.log10(rho), np.log10(r))


def inner_slope(r, rho, r1, r2, n=None):
    """
    Power-law slope of rho(r) fitted between r1 and r2 (e.g. 0.01-0.02 R200, as in Di Cintio et
    al. 2014); weighted by particle counts n if given. Returns (slope, error) or (nan, nan).
    """
    r, rho = np.asarray(r), np.asarray(rho)
    s = (r >= r1) & (r <= r2) & (rho > 0)
    if s.sum() < 3:
        return np.nan, np.nan
    w = None if n is None else np.sqrt(np.maximum(np.asarray(n)[s], 1))
    p, cov = np.polyfit(np.log10(r[s]), np.log10(rho[s]), 1, w=w, cov="unscaled" if w is not None else True)
    return p[0], np.sqrt(cov[0, 0])


# --------------------------------------------------------------------------------------------- models
def nfw(r, rho_s, r_s):
    x = r / r_s
    return rho_s / (x * (1 + x) ** 2)


def gnfw(r, rho_s, r_s, gamma):
    """Generalised NFW (Zhao alpha=1, beta=3) with free inner slope gamma."""
    x = r / r_s
    return rho_s / (x ** gamma * (1 + x) ** (3 - gamma))


def burkert(r, rho_0, r_0):
    return rho_0 * r_0 ** 3 / ((r + r_0) * (r ** 2 + r_0 ** 2))


def nfw_mass(r, rho_s, r_s):
    x = r / r_s
    return 4 * np.pi * rho_s * r_s ** 3 * (np.log(1 + x) - x / (1 + x))


def core_nfw(r, rho_s, r_s, r_c, n):
    """
    coreNFW density (Read, Agertz & Collins 2016): M(<r) = M_NFW(<r) f^n with f = tanh(r/r_c);
    n = 0 is NFW, n = 1 a full core inside r_c.
    """
    f = np.tanh(r / r_c)
    return f ** n * nfw(r, rho_s, r_s) + n * f ** (n - 1) * (1 - f ** 2) / (4 * np.pi * r ** 2 * r_c) * nfw_mass(r, rho_s, r_s)


def zhao(r, rho_s, r_s, alpha, beta, gamma):
    """alpha-beta-gamma profile, as used by Di Cintio et al. (2014)."""
    x = r / r_s
    return rho_s / (x ** gamma * (1 + x ** alpha) ** ((beta - gamma) / alpha))


def dicintio14_shape(mstar_over_mhalo):
    """
    (alpha, beta, gamma) of the Di Cintio et al. (2014) profile as a function of X = log10(M*/Mhalo),
    calibrated for -4.1 < X < -1.3 (z = 0, NIHAO-precursor MaGICC haloes). Coefficients as given in
    their eq. 3; check against the paper before quoting.
    """
    X = np.log10(mstar_over_mhalo)
    alpha = 2.94 - np.log10((10 ** (X + 2.33)) ** -1.08 + (10 ** (X + 2.33)) ** 2.29)
    beta = 4.23 + 1.34 * X + 0.26 * X ** 2
    gamma = -0.06 + np.log10((10 ** (X + 2.56)) ** -0.68 + 10 ** (X + 2.56))
    return alpha, beta, gamma


MODELS = {
    "nfw": (nfw, ["rho_s", "r_s"]),
    "gnfw": (gnfw, ["rho_s", "r_s", "gamma"]),
    "core_nfw": (core_nfw, ["rho_s", "r_s", "r_c", "n"]),
    "burkert": (burkert, ["rho_0", "r_0"]),
}


def fit_profile(r, rho, model, r_min, r_max, n=None, p0=None, bounds=None):
    """
    Fit a density model in log space between r_min and r_max (Poisson weights if counts n given).
    Returns dict(params, errors, chi2_per_dof, model). For 'core_nfw', fixing the NFW part from an
    outer NFW fit is often more stable: pass bounds that pin rho_s and r_s.
    """
    func, names = MODELS[model]
    r, rho = np.asarray(r, float), np.asarray(rho, float)
    s = (r >= r_min) & (r <= r_max) & (rho > 0)
    if s.sum() <= len(names):
        return dict(params=dict.fromkeys(names, np.nan), errors=dict.fromkeys(names, np.nan), chi2_per_dof=np.nan, model=model)
    rs_guess = np.sqrt(r_min * r_max)
    rho_guess = np.interp(rs_guess, r[s], rho[s])
    defaults = {"nfw": ([rho_guess * 4, rs_guess], ([0, 1e-4], [np.inf, np.inf])),
                "gnfw": ([rho_guess * 4, rs_guess, 1.0], ([0, 1e-4, -0.5], [np.inf, np.inf, 2.5])),
                "core_nfw": ([rho_guess * 4, rs_guess, 0.2 * rs_guess, 0.5], ([0, 1e-4, 1e-4, 0.0], [np.inf, np.inf, np.inf, 1.0])),
                "burkert": ([rho_guess * 4, rs_guess], ([0, 1e-4], [np.inf, np.inf]))}
    p0 = p0 if p0 is not None else defaults[model][0]
    bounds = bounds if bounds is not None else defaults[model][1]
    sigma = None if n is None else 1.0 / np.sqrt(np.maximum(np.asarray(n, float)[s], 1))
    logf = lambda rr, *p: np.log10(np.maximum(func(rr, *p), 1e-300))
    starts = [p0]
    if model == "core_nfw":   # the core radius and strength are poorly constrained from a single start
        starts += [[p0[0], p0[1], fc * p0[1], nn] for fc in (0.05, 0.2, 0.5, 1.0) for nn in (0.3, 0.7, 1.0)]
    best = None
    for start in starts:
        start = np.clip(start, np.asarray(bounds[0], float) + 1e-12, np.asarray(bounds[1], float) - 1e-12)
        try:
            popt, pcov = curve_fit(logf, r[s], np.log10(rho[s]), p0=start, sigma=sigma, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        res = np.log10(rho[s]) - logf(r[s], *popt)
        chi2 = np.sum((res / (sigma if sigma is not None else 1.0)) ** 2) / max(s.sum() - len(names), 1)
        if best is None or chi2 < best[2]:
            best = (popt, np.sqrt(np.abs(np.diag(pcov))), chi2)
    if best is None:
        best = (np.full(len(names), np.nan), np.full(len(names), np.nan), np.nan)
    popt, err, chi2 = best
    return dict(params=dict(zip(names, popt)), errors=dict(zip(names, err)), chi2_per_dof=chi2, model=model)


def evaluate(fit, r):
    """Density of a fit_profile result at radii r."""
    func, names = MODELS[fit["model"]]
    return func(np.asarray(r, float), *[fit["params"][k] for k in names])


# --------------------------------------------------------------------------------------------- energetics
def potential_energy(r_grid, rho):
    """Self potential energy W = -4 pi G int rho M(<r) r dr of a spherical density on a radial grid [erg-like: Msun (km/s)^2]."""
    r_grid, rho = np.asarray(r_grid, float), np.asarray(rho, float)
    dm = 4 * np.pi * r_grid ** 2 * rho
    m = np.concatenate([[0.0], np.cumsum(0.5 * (dm[1:] + dm[:-1]) * np.diff(r_grid))])
    integrand = rho * m * r_grid
    return -4 * np.pi * G_KPC * np.trapezoid(integrand, r_grid)


def core_formation_energy(rho_cusp, rho_core, r_max, r_min=1e-4, n=4000):
    """
    Energy [erg] needed to turn the cusp into the core, as the difference of their self potential
    energies within r_max (Penarrubia et al. 2012). rho_cusp, rho_core: callables of r [kpc].
    Choose r_max well outside the core so both enclose nearly the same mass.
    """
    rg = np.geomspace(r_min, r_max, n)
    dW = potential_energy(rg, rho_core(rg)) - potential_energy(rg, rho_cusp(rg))
    return dW * 1.98841e33 * 1e10          # Msun (km/s)^2 -> erg


# --------------------------------------------------------------------------------------------- kinematics
def velocity_profiles(pos, vel, bins, mass=None, chunk=2_000_000):
    """
    sigma_r, sigma_theta, sigma_phi, anisotropy beta = 1 - (s_t^2 + s_p^2)/(2 s_r^2), mean v_r, per radial bin.
    Accumulates weighted sums chunk by chunk, so memory stays small for very large particle sets.
    """
    pos, vel = np.asarray(pos), np.asarray(vel)
    nb = len(bins) - 1
    acc = {k: np.zeros(nb) for k in ("w", "n", "r1", "r2", "t1", "t2", "p1", "p2")}
    for a in range(0, len(pos), chunk):
        p = pos[a:a + chunk].astype(np.float64); v = vel[a:a + chunk].astype(np.float64)
        r = np.sqrt(np.einsum("ij,ij->i", p, p))
        idx = np.digitize(r, bins) - 1
        ok = (idx >= 0) & (idx < nb)
        p, v, r, idx = p[ok], v[ok], r[ok], idx[ok]
        w = np.ones(len(r)) if mass is None else np.asarray(mass[a:a + chunk], np.float64)[ok]
        rr = np.where(r > 0, r, 1.0)
        vr = np.einsum("ij,ij->i", v, p) / rr
        R = np.hypot(p[:, 0], p[:, 1]); RR = np.where(R > 0, R, 1.0)
        vphi = (-p[:, 1] * v[:, 0] + p[:, 0] * v[:, 1]) / RR
        # theta-hat = phi-hat x r-hat; v_theta = v . theta-hat
        vth = (p[:, 2] * (p[:, 0] * v[:, 0] + p[:, 1] * v[:, 1]) / (RR * rr) - R * v[:, 2] / rr)
        for key, x in (("r", vr), ("t", vth), ("p", vphi)):
            acc[key + "1"] += np.bincount(idx, weights=w * x, minlength=nb)
            acc[key + "2"] += np.bincount(idx, weights=w * x * x, minlength=nb)
        acc["w"] += np.bincount(idx, weights=w, minlength=nb)
        acc["n"] += np.bincount(idx, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = {k: acc[k + "1"] / acc["w"] for k in "rtp"}
        var = {k: acc[k + "2"] / acc["w"] - mean[k] ** 2 for k in "rtp"}
    good = acc["n"] >= 20
    out = {"sigma_r": np.where(good, np.sqrt(np.maximum(var["r"], 0)), np.nan),
           "sigma_theta": np.where(good, np.sqrt(np.maximum(var["t"], 0)), np.nan),
           "sigma_phi": np.where(good, np.sqrt(np.maximum(var["p"], 0)), np.nan),
           "vr_mean": np.where(good, mean["r"], np.nan)}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["beta"] = np.where(good, 1 - (var["t"] + var["p"]) / (2 * var["r"]), np.nan)
    out["r"] = np.sqrt(np.asarray(bins[1:]) * np.asarray(bins[:-1]))
    out["n"] = acc["n"]
    return out


def spherical_potential(r_all, m_all, r_grid):
    """Phi(r) [(km/s)^2] of all matter in the spherical approximation, zero at the outermost particle."""
    r_all, m_all = np.asarray(r_all, float), np.asarray(m_all, float)
    o = np.argsort(r_all)
    rs, cm = r_all[o], np.cumsum(m_all[o])
    menc = np.interp(r_grid, rs, cm, left=0.0)
    # outer term: G * sum over particles outside r of m_i / r_i
    inv = np.cumsum((m_all[o] / np.maximum(rs, 1e-6))[::-1])[::-1]
    idx = np.searchsorted(rs, r_grid)
    outer = np.where(idx < len(rs), inv[np.minimum(idx, len(rs) - 1)], 0.0)
    return -G_KPC * (menc / r_grid + outer)


def potential_on_grid(r_grid, components):
    """
    Phi(r) [(km/s)^2] on r_grid [kpc] in the spherical approximation, from a list of (r, mass)
    pairs (one per particle type; mass may be a scalar). Built from shell masses on the grid, so no
    particle arrays are copied or concatenated. Zero beyond the outermost particle.
    """
    r_grid = np.asarray(r_grid, float)
    edges = np.concatenate([[0.0], np.sqrt(r_grid[1:] * r_grid[:-1]), [np.inf]])
    m_shell = np.zeros(len(r_grid))
    for r, m in components:
        if np.ndim(m) == 0:
            m_shell += np.histogram(r, bins=edges)[0] * float(m)
        else:
            m_shell += np.histogram(r, bins=edges, weights=m)[0]
    m_enc = np.cumsum(m_shell)                                    # mass inside the outer edge of each shell
    menc_at = np.concatenate([[0.0], m_enc[:-1]]) + 0.5 * m_shell # approximately M(<r_grid)
    outer = np.cumsum((m_shell / r_grid)[::-1])[::-1] - 0.5 * m_shell / r_grid   # sum of m/r outside r_grid
    phi = -G_KPC * (menc_at / r_grid + outer)
    return phi


def find_ids(ids_all, ids_wanted, chunk=5_000_000):
    """
    Positions of the wanted IDs in a (large) ID array, matching chunk by chunk against the sorted
    wanted list, so no full-size sort or copy is made. Returns (idx_all, idx_wanted).
    """
    ids_wanted = np.asarray(ids_wanted)
    order = np.argsort(ids_wanted)
    w_sorted = ids_wanted[order]
    ia, iw = [], []
    for a in range(0, len(ids_all), chunk):
        c = np.asarray(ids_all[a:a + chunk])
        pos = np.searchsorted(w_sorted, c)
        pos_c = np.minimum(pos, len(w_sorted) - 1)
        hit = w_sorted[pos_c] == c
        ia.append(np.flatnonzero(hit) + a)
        iw.append(order[pos_c[hit]])
    if not ia:
        return np.array([], int), np.array([], int)
    return np.concatenate(ia), np.concatenate(iw)


def apocentres(pos, vel, r_grid, phi_grid, chunk=20000):
    """
    Apocentre [kpc] of each particle's orbit in a spherical potential: the first radius beyond the
    particle's current radius where E - Phi(r) - L^2/(2 r^2) changes sign (linearly interpolated).
    NaN if the orbit reaches beyond the grid (unbound within it).
    """
    pos, vel = np.asarray(pos, float), np.asarray(vel, float)
    r = np.linalg.norm(pos, axis=1)
    phi_r = np.interp(r, r_grid, phi_grid)
    E = 0.5 * np.sum(vel ** 2, axis=1) + phi_r
    L2 = np.sum(np.cross(pos, vel) ** 2, axis=1)
    f_r = E - phi_r - L2 / (2 * np.maximum(r, 1e-12) ** 2)          # = v_r^2 / 2 >= 0
    out = np.full(len(r), np.nan)
    for a in range(0, len(r), chunk):
        sl = slice(a, a + chunk)
        f = E[sl, None] - phi_grid[None, :] - L2[sl, None] / (2 * r_grid[None, :] ** 2)
        beyond = r_grid[None, :] > r[sl, None]
        neg = beyond & (f < 0)
        has = neg.any(axis=1)
        j = np.argmax(neg, axis=1)                                     # first negative grid point beyond r
        jprev = j - 1
        prev_is_r = (jprev < 0) | ~beyond[np.arange(len(j)), np.maximum(jprev, 0)]
        r0 = np.where(prev_is_r, r[sl], r_grid[np.maximum(jprev, 0)])
        f0 = np.where(prev_is_r, f_r[sl], f[np.arange(len(j)), np.maximum(jprev, 0)])
        r1, f1 = r_grid[j], f[np.arange(len(j)), j]
        root = r0 + (r1 - r0) * f0 / np.where(f0 - f1 != 0, f0 - f1, 1)
        out[sl] = np.where(has, root, np.nan)
    return out


def match_ids(ids_a, ids_b):
    """Indices (ia, ib) such that ids_a[ia] == ids_b[ib], for IDs present in both."""
    ids_a, ids_b = np.asarray(ids_a), np.asarray(ids_b)
    common, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
    return ia, ib
