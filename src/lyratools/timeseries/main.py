"""
Step 1a: quantities of the main progenitor in one snapshot (group /Main of the per-snapshot file).

Particles of all types within load_R200 x R200 of the progenitor's SubhaloPos are loaded,
positions relative to that centre (float64 centre, float32 positions) and velocities in the frame
of its SubhaloVel. "gal" means r < gal_R200 x R200. Every reduction accumulates in float64.
Projected quantities are stored for the x, y, z axes (shape 3) and face-on (suffix _faceon),
face-on meaning along the angular momentum of the cold gas in gal (fallbacks: all gal gas, gal stars).
"""
import numpy as np

from .. import chemistry, stellar_evolution
from ..constants import KMS_PER_KPC_TO_PER_YR, KPC, MSUN, PROTONMASS, SN_MIN_MASS
from ..yields import twelve_log_oh
from .io import FORMATION_TIME

F64 = np.float64
F32 = np.float32
NAN = np.nan
E51 = 1e51          # energies are stored in 1e51 erg (float32 overflows above 3.4e38)

# units attribute of each /Main dataset: first matching prefix
UNITS = [("Apertures_kpc", "kpc"), ("Mass/", "Msun"), ("SFH_recent", "Msun/yr; bin i: ages [i, i+1) Myr"),
         ("SFR", "Msun/yr"), ("MassFormedSincePrev", "Msun"), ("Age/", "Myr"), ("Rhalf/", "kpc"),
         ("AxisRatios/", "b/a, c/a"), ("Sigma/", "km/s"), ("VoverSigma/", "dimensionless"), ("Mtrue_inRe/", "Msun"),
         ("OH/Gradient/per_kpc", "dex/kpc"), ("OH/Gradient/per_Re", "dex per projected face-on Re"),
         ("OH/", "12+log(O/H)"), ("Phases/Mass/", "Msun"), ("Phases/Volume/", "kpc^3"), ("Phases/nH_hist", "Msun"),
         ("BirthDensity/", "cm^-3 (16, 50, 84th percentiles of nH at birth)"), ("KS/Sigma_gas", "Msun/pc^2"),
         ("KS/Sigma_SFR", "Msun/yr/kpc^2"), ("Flows/Out/vr_p", "km/s"), ("Flows/", "Msun/yr"), ("SN/N", "count"),
         ("SN/E", "1e51 erg"), ("Times/", "Myr"), ("Contam_R200", "mass fraction"), ("RotAxis", "")]


def units_of(name):
    if "/Energy/" in name:
        return "1e51 erg/yr"
    return next((u for p, u in UNITS if name.startswith(p)), "")


# ------------------------------------------------------------------ loading
def _sphere(sn, pt, center_code, radius_code, vel0=None):
    """Positions [kpc, float32], masses [Msun], radii, and (if vel0 given) velocities [km/s] in that frame."""
    idx, rel = sn.scan_sphere(pt, center_code, radius_code)
    p = {"pos": (rel * sn.kpc_per_code).astype(F32)}
    del rel
    p["mass"] = sn.masses(pt, idx).astype(F32)
    p["r"] = np.sqrt(np.einsum("ij,ij->i", p["pos"], p["pos"], dtype=F64)).astype(F32)
    if vel0 is not None:
        v = sn.read_indices(pt, "Velocities", idx).astype(F64) * np.sqrt(sn.a) - vel0
        p["vel"] = v.astype(F32)
        with np.errstate(invalid="ignore", divide="ignore"):
            p["vr"] = (np.einsum("ij,ij->i", p["pos"], v) / np.where(p["r"] > 0, p["r"], np.inf)).astype(F32)
    return idx, p


def _optional(sn, pt, field, idx, cols=None, fill=NAN):
    if sn.has(pt, field):
        return sn.read_indices(pt, field, idx, cols)
    shape = (len(idx), len(cols)) if cols is not None else (len(idx),)
    return np.full(shape, fill, F32)


def _select(p, keep):
    return {k: v[keep] for k, v in p.items()}


def load_main(sn, center_code, vel0, radius_kpc, t_now, pt23="contamination"):
    """
    Particles of types 0-4 within radius_kpc: {pt: {field: array}} (empty dicts for absent types).
    With pt23 = "stars", also "ic": stars present in the initial conditions (PartType2/3 and
    PartType4 with GFM_InitialMass = 0), which have no formation times or abundances.
    """
    rad = radius_kpc / sn.kpc_per_code
    P = {pt: {} for pt in range(5)}
    if sn.total(0):
        idx, g = _sphere(sn, 0, center_code, rad, vel0)
        g["rho"] = (sn.read_indices(0, "Density", idx).astype(F64) * sn.density_cgs_comoving / sn.a ** 3).astype(F32)
        g["u"] = (sn.read_indices(0, "InternalEnergy", idx).astype(F64) * sn.velocity_cgs ** 2).astype(F32)
        met = _optional(sn, 0, "GFM_Metals", idx, cols=[0, 4])
        g["XH"], g["XO"] = met[:, 0].astype(F32), met[:, 1].astype(F32)
        g["XH"] = np.where(np.isfinite(g["XH"]), g["XH"], F32(0.76))
        xe = _optional(sn, 0, "ElectronAbundance", idx)
        g["T"] = chemistry.temperature(g["u"], np.where(np.isfinite(xe), xe, 1.0 + (1 - g["XH"]) / (4 * g["XH"])), g["XH"]).astype(F32)
        g["nH"] = (g["XH"].astype(F64) * g["rho"] / PROTONMASS).astype(F32)
        g["nha"] = _optional(sn, 0, "NeutralHydrogenAbundance", idx).astype(F32)
        g["cool"] = _optional(sn, 0, "GFM_CoolingRate", idx).astype(F32)
        P[0] = g
    if sn.total(1):
        P[1] = _sphere(sn, 1, center_code, rad)[1]
    for pt in (2, 3):
        if sn.total(pt):
            P[pt] = _sphere(sn, pt, center_code, rad)[1]
    if sn.total(4):
        idx, s = _sphere(sn, 4, center_code, rad, vel0)
        aform = sn.read_indices(4, sn.field_name(4, FORMATION_TIME), idx).astype(F64)
        m_init = sn.read_indices(4, "GFM_InitialMass", idx).astype(F64) * sn.msun_per_code
        keep = (m_init > 0) & (aform > 0)            # inert initial-condition stars; wind particles
        if pt23 == "stars":
            inert = m_init <= 0
            P["ic"] = {k: s[k][inert] for k in ("pos", "mass", "r")}
        s = _select(s, keep)
        idx, aform = idx[keep], aform[keep]
        s["m_init"] = m_init[keep]
        s["tform"] = sn.t_myr(aform)
        s["age"] = t_now - s["tform"]
        s["sntime"] = _optional(sn, 4, "SNTime", idx).astype(F64)
        # isolated LYRA runs store code times of death instead of lifetimes: convert to the lifetime convention
        if not sn.cosmological and stellar_evolution.sntime_convention(s["sntime"], aform, sn.time_code) == "absolute":
            s["sntime"] = stellar_evolution.sntime_as_lifetime(s["sntime"], aform, sn.myr_per_code_time)
        met = _optional(sn, 4, "GFM_Metals", idx, cols=[0, 4])
        s["XH"], s["XO"] = met[:, 0].astype(F32), met[:, 1].astype(F32)
        bd = _optional(sn, 4, "BirthDensity", idx).astype(F64)
        # comoving at the formation scale factor, not the current one
        s["nH_birth"] = (s["XH"] * bd * sn.density_cgs_comoving / sn.formation_scale_factor(aform) ** 3 / PROTONMASS).astype(F32)
        P[4] = s
    if pt23 == "stars":
        ic = [P[pt] for pt in (2, 3) if P[pt]] + ([P["ic"]] if "ic" in P else [])
        P["ic"] = {k: np.concatenate([p[k] for p in ic]) for k in ("pos", "mass", "r")} if ic else {}
        P[2], P[3] = {}, {}                           # counted once, as "ic"
    return P


# ------------------------------------------------------------------ reductions (float64)
def wsum(w, sel=None):
    w = np.asarray(w)
    return float(np.sum(w if sel is None else w[sel], dtype=F64))


def wquantile(x, w, q):
    """Weighted quantile(s): x where the float64 cumulative weight first reaches q x total; NaN if empty."""
    q = np.atleast_1d(np.asarray(q, F64))
    x = np.asarray(x)
    w = np.asarray(w)
    if len(x) == 0:
        return np.full(len(q), NAN)
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o], dtype=F64)
    if not cw[-1] > 0:
        return np.full(len(q), NAN)
    i = np.minimum(np.searchsorted(cw, q * cw[-1]), len(x) - 1)
    return x[o][i].astype(F64)


def half_mass_radius(r, m):
    return float(wquantile(r, m, 0.5)[0])


def wmean(x, w):
    w = np.asarray(w, F64)
    sw = w.sum()
    return float(np.sum(w * x, dtype=F64) / sw) if sw > 0 else NAN


def wstd(x, w):
    m = wmean(x, w)
    if not np.isfinite(m):
        return NAN
    return float(np.sqrt(wmean((np.asarray(x, F64) - m) ** 2, w)))


def enclosed(r, w, radii):
    """Sum of w with r < radius, for each radius (float64 bincount)."""
    radii = np.asarray(radii, F64)
    if len(r) == 0:
        return np.zeros(len(radii))
    o = np.argsort(radii)
    k = np.searchsorted(radii[o], r, side="right")       # particle is inside apertures o[k:]
    h = np.bincount(k, weights=None if w is None else np.asarray(w), minlength=len(radii) + 1).astype(F64)
    out = np.empty(len(radii))
    out[o] = np.cumsum(h[:-1], dtype=F64)
    return out


def oh_from_sums(xo, xh, m, sel=None):
    """12 + log(O/H) by number of a reservoir: total O and H masses."""
    if sel is not None:
        xo, xh, m = xo[sel], xh[sel], m[sel]
    so, sh = np.sum(xo * m, dtype=F64), np.sum(xh * m, dtype=F64)
    return float(twelve_log_oh(so, sh)) if (so > 0 and sh > 0) else NAN


def oh_per_particle(xo, xh):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = twelve_log_oh(xo, xh)
    return np.where(np.isfinite(v), v, NAN)


def shape_ratios(pos, m, min_n):
    """(b/a, c/a) from the eigenvalues of the mass-weighted second-moment tensor."""
    if len(m) < min_n:
        return np.array([NAN, NAN])
    S = np.zeros((3, 3))
    for a in range(0, len(m), 1 << 22):                 # float64 in chunks: R200 may hold ~1e8 particles
        x = np.asarray(pos[a:a + (1 << 22)], F64)
        S += np.einsum("i,ij,ik->jk", np.asarray(m[a:a + (1 << 22)], F64), x, x)
    S /= wsum(m)
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    if not ev[0] > 0:
        return np.array([NAN, NAN])
    return np.sqrt(np.clip(ev[1:] / ev[0], 0, None))


def rotation_matrix(axis):
    """Rows: x', y', z' with z' along axis."""
    z = np.asarray(axis, F64) / np.linalg.norm(axis)
    x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def projections(pos, vel, rot):
    """Yield (i, R_projected, v_los) for i = 0 face-on (NaN if rot is None), 1-3 along x, y, z."""
    pos = np.asarray(pos, F64)
    vel = np.asarray(vel, F64) if vel is not None else None
    if rot is None:
        yield 0, np.full(len(pos), NAN), (np.full(len(pos), NAN) if vel is not None else None)
    else:
        pf = pos @ rot.T
        yield 0, np.hypot(pf[:, 0], pf[:, 1]), ((vel @ rot.T)[:, 2] if vel is not None else None)
    for k in range(3):
        a, b = [j for j in range(3) if j != k]
        yield k + 1, np.hypot(pos[:, a], pos[:, b]), (vel[:, k] if vel is not None else None)


def _put_proj(res, name, vals):
    res[name] = np.asarray(vals[1:4], F64)
    res[name + "_faceon"] = vals[0]


def v_over_sigma(pos, vel, m, rot):
    """Mass-weighted mean v_phi over the 1D dispersion about it (3D dispersion / sqrt 3), face-on frame."""
    if rot is None or len(m) < 2:
        return NAN
    pf, vf = np.asarray(pos, F64) @ rot.T, np.asarray(vel, F64) @ rot.T
    R = np.hypot(pf[:, 0], pf[:, 1])
    ok = R > 0
    vphi = (pf[ok, 0] * vf[ok, 1] - pf[ok, 1] * vf[ok, 0]) / R[ok]
    w = np.asarray(m, F64)[ok]
    V = wmean(vphi, w)
    s3 = wmean(np.einsum("ij,ij->i", vf[ok], vf[ok]) - 2 * V * vphi + V ** 2, w)
    return V / np.sqrt(s3 / 3) if s3 > 0 else NAN


# ------------------------------------------------------------------ main quantities
def compute_main(P, R200, t_now, t_prev, cfg, log=print, pt23="contamination"):
    c = cfg["main"]
    g, d, s = P[0], P[1], P[4]
    empty = {"pos": np.zeros((0, 3), F32), "vel": np.zeros((0, 3), F32), "mass": np.zeros(0, F32), "r": np.zeros(0, F32)}
    g = g or dict(empty, vr=np.zeros(0, F32), **{k: np.zeros(0, F32) for k in ("rho", "u", "XH", "XO", "T", "nH", "nha", "cool")})
    d = d or dict(empty)
    s = s or dict(empty, **{k: np.zeros(0) for k in ("m_init", "tform", "age", "sntime", "XH", "XO", "nH_birth", "vr")})
    res = {}
    rgal = c["gal_R200"] * R200
    Tc, nHc = c["T_cold"], c["nH_cold"]

    # -- masses in apertures
    aps = np.array(list(c["apertures_kpc"]) + [f * R200 for f in c["apertures_R200"]], F64)
    res["Apertures_kpc"] = aps
    res["Mass/Gas"] = enclosed(g["r"], g["mass"], aps)
    res["Mass/DM"] = enclosed(d["r"], d["mass"], aps)
    res["Mass/Stars"] = enclosed(s["r"], s["mass"], aps)
    res["Mass/StarsInit"] = enclosed(s["r"], s["m_init"], aps)
    res["Mass/HI"] = enclosed(g["r"], g["XH"].astype(F64) * g["nha"] * g["mass"], aps)
    res["Mass/GasO"] = enclosed(g["r"], g["XO"].astype(F64) * g["mass"], aps)
    res["Mass/StarsO"] = enclosed(s["r"], s["XO"].astype(F64) * s["mass"], aps)
    if pt23 == "stars":
        ic = P.get("ic") or {"r": np.zeros(0, F32), "mass": np.zeros(0, F32)}
        res["Mass/StarsIC"] = enclosed(ic["r"], ic["mass"], aps)

    # -- galaxy subsets (float64 copies of the small gal sets)
    gg, sg = g["r"] < rgal, s["r"] < rgal
    G = {k: v[gg] for k, v in g.items()}
    S = {k: v[sg] for k, v in s.items()}
    gcold = G["T"] < Tc

    # -- rotation axis
    rot, src = None, -1
    for code, (pos, vel, m) in enumerate([(G["pos"][gcold], G["vel"][gcold], G["mass"][gcold]),
                                          (G["pos"], G["vel"], G["mass"]), (S["pos"], S["vel"], S["mass"])]):
        L = np.sum(np.cross(np.asarray(pos, F64), np.asarray(vel, F64)) * np.asarray(m, F64)[:, None], axis=0)
        if len(m) >= c["min_particles_shape"] and np.linalg.norm(L) > 0:
            rot, src = rotation_matrix(L), code
            break
    res["RotAxis"] = rot[2] if rot is not None else np.full(3, NAN)
    res["RotAxisSource"] = np.int64(src)            # 0 cold gas, 1 all gas, 2 stars (all within gal), -1 none

    # -- star formation
    edges = np.arange(0.0, c["sfh_window_Myr"] + 0.5 * c["sfh_bin_Myr"], c["sfh_bin_Myr"])
    res["SFH_recent"] = np.histogram(S["age"], bins=edges, weights=S["m_init"].astype(F64))[0] / (c["sfh_bin_Myr"] * 1e6)
    in200 = s["r"] < R200
    for w in c["sfr_windows_Myr"]:
        res[f"SFR/{w:g}"] = wsum(S["m_init"], S["age"] < w) / (w * 1e6)
        res[f"SFR_R200/{w:g}"] = wsum(s["m_init"], in200 & (s["age"] < w)) / (w * 1e6)
    res["MassFormedSincePrev"] = wsum(S["m_init"], S["tform"] > t_prev) if np.isfinite(t_prev) else NAN
    res["Age/MassWeighted"] = wmean(S["age"], S["mass"])
    res["Age/Median"] = float(wquantile(S["age"], S["mass"], 0.5)[0])

    # -- sizes
    y10, y100 = (S["age"] < c["young_Myr"][0]), (S["age"] < c["young_Myr"][1])
    Re = half_mass_radius(S["r"], S["mass"])
    res["Rhalf/Stars3D"] = Re
    res[f"Rhalf/Young{c['young_Myr'][0]:g}"] = half_mass_radius(S["r"][y10], S["m_init"][y10])
    res[f"Rhalf/Young{c['young_Myr'][1]:g}"] = half_mass_radius(S["r"][y100], S["m_init"][y100])
    res["Rhalf/Gas"] = half_mass_radius(G["r"], G["mass"])
    res["Rhalf/ColdGas"] = half_mass_radius(G["r"][gcold], G["mass"][gcold])

    # -- projected sizes and line-of-sight dispersions
    rp = c["proj_radius_kpc"]
    Re_p, sfr_re, sig_re, sig_y = (np.full(4, NAN) for _ in range(4))
    for i, Rp, vlos in projections(S["pos"], S["vel"], rot):
        Re_p[i] = half_mass_radius(Rp, S["mass"])
        inre = Rp < Re_p[i]
        sfr_re[i] = wsum(S["m_init"], inre & y10) / (c["young_Myr"][0] * 1e6)
        sig_re[i] = wstd(vlos[inre], S["mass"][inre])
        yy = y10 & (Rp < rp)
        sig_y[i] = wstd(vlos[yy], S["mass"][yy])
    _put_proj(res, "Rhalf/StarsProj", Re_p)
    _put_proj(res, "SFR10_inRe_proj", sfr_re)
    _put_proj(res, "Sigma/StarsLOS_Re", sig_re)
    _put_proj(res, f"Sigma/Young{c['young_Myr'][0]:g}LOS", sig_y)
    sig_c = np.full(4, NAN)
    for i, Rp, vlos in projections(G["pos"][gcold], G["vel"][gcold], rot):
        sel = Rp < rp
        sig_c[i] = wstd(vlos[sel], G["mass"][gcold][sel])
    _put_proj(res, f"Sigma/ColdGasLOS_{rp:g}kpc", sig_c)
    res["VoverSigma/Stars"] = v_over_sigma(S["pos"], S["vel"], S["mass"], rot)
    res["VoverSigma/ColdGas"] = v_over_sigma(G["pos"][gcold], G["vel"][gcold], G["mass"][gcold], rot)

    # -- shapes
    mn = c["min_particles_shape"]
    for name, p in (("Stars", s), ("DM", d)):
        for rr, label in [(x, f"{x:g}kpc") for x in c["axis_radii_kpc"]] + [(R200, "R200")]:
            sel = p["r"] < rr
            res[f"AxisRatios/{name}/{label}"] = shape_ratios(p["pos"][sel], p["mass"][sel], mn)

    # -- true masses within the 3D stellar half-mass radius
    for name, p in (("DM", d), ("Gas", g), ("Stars", s)):
        res[f"Mtrue_inRe/{name}"] = wsum(p["mass"], p["r"] < Re) if np.isfinite(Re) else NAN

    # -- oxygen abundances, 12 + log(O/H) by number
    cold_dense = (g["T"] < Tc) & (g["nH"] > nHc)
    in1 = g["r"] < 1.0
    res["OH/GasMW/1kpc"] = oh_from_sums(g["XO"], g["XH"], g["mass"], in1)
    res[f"OH/GasMW/{c['gal_R200']:g}R200"] = oh_from_sums(G["XO"], G["XH"], G["mass"])
    res["OH/ColdGasMW/1kpc"] = oh_from_sums(g["XO"], g["XH"], g["mass"], in1 & cold_dense)
    oh_y = oh_per_particle(S["XO"][y10], S["XH"][y10])
    ok = np.isfinite(oh_y)
    res[f"OH/Young{c['young_Myr'][0]:g}/mean"] = wmean(oh_y[ok], S["m_init"][y10][ok])
    res[f"OH/Young{c['young_Myr'][0]:g}/std"] = wstd(oh_y[ok], S["m_init"][y10][ok])
    res["OH/StarsMW_Re"] = oh_from_sums(S["XO"], S["XH"], S["mass"], S["r"] < Re) if np.isfinite(Re) else NAN
    # gradient: cold dense gas in gal, face-on projected radius (3D if no axis)
    gc = (G["T"] < Tc) & (G["nH"] > nHc)
    Rg = next(projections(G["pos"][gc], None, rot))[1] if rot is not None else G["r"][gc].astype(F64)
    be = np.linspace(c["gradient_kpc"][0], c["gradient_kpc"][1], c["gradient_nbins"] + 1)
    k = np.digitize(Rg, be) - 1
    inb = (k >= 0) & (k < len(be) - 1)
    mg = G["mass"][gc].astype(F64)
    so = np.bincount(k[inb], weights=(G["XO"][gc] * mg)[inb], minlength=len(be) - 1)
    sh = np.bincount(k[inb], weights=(G["XH"][gc] * mg)[inb], minlength=len(be) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ohb = twelve_log_oh(so, sh)
    okb = np.isfinite(ohb)
    rc = 0.5 * (be[1:] + be[:-1])
    slope = np.polyfit(rc[okb], ohb[okb], 1)[0] if okb.sum() >= 3 else NAN
    res["OH/Gradient/per_kpc"] = slope
    Re_g = Re_p[0] if rot is not None else Re
    res["OH/Gradient/per_Re"] = slope * Re_g

    # -- ISM phases within gal
    T = G["T"]
    lo, mid, hi = c["phase_T"]
    vol = G["mass"].astype(F64) * MSUN / G["rho"].astype(F64) / KPC ** 3
    for name, sel in (("CNM", T < lo), ("WNM", (T >= lo) & (T < mid)), ("WIM", (T >= mid) & (T < hi)), ("HIM", T >= hi)):
        res[f"Phases/Mass/{name}"] = wsum(G["mass"], sel)
        res[f"Phases/Volume/{name}"] = wsum(vol, sel)
    a0, a1, da = c["nH_hist"]
    nedges = np.arange(a0, a1 + 0.5 * da, da)
    with np.errstate(divide="ignore"):
        res["Phases/nH_hist"] = np.histogram(np.log10(G["nH"]), bins=nedges, weights=G["mass"].astype(F64))[0]
    nb = S["nH_birth"][y10]
    nb = nb[np.isfinite(nb)]
    res[f"BirthDensity/Young{c['young_Myr'][0]:g}"] = np.percentile(nb, [16, 50, 84]) if len(nb) else np.full(3, NAN)

    # -- Kennicutt-Schmidt: face-on cylinder of radius and half-height rgal
    area = np.pi * rgal ** 2
    if rot is not None:
        def cyl(p):
            near = p["r"] < np.sqrt(2) * rgal
            pf = np.asarray(p["pos"][near], F64) @ rot.T
            return near, (np.hypot(pf[:, 0], pf[:, 1]) < rgal) & (np.abs(pf[:, 2]) < rgal)
        ng, cg = cyl(g)
        ns, cs = cyl(s)
        res["KS/Sigma_gas"] = wsum(g["mass"][ng], cg) / (area * 1e6)
        for w in (10, 100):
            res[f"KS/Sigma_SFR{w}"] = wsum(s["m_init"][ns], cs & (s["age"][ns] < w)) / (w * 1e6) / area
    else:
        res["KS/Sigma_gas"] = res["KS/Sigma_SFR10"] = res["KS/Sigma_SFR100"] = NAN

    # -- gas flows through spherical shells
    th = c["flow_thickness"]
    t1, t2 = c["flow_phase_T"]
    shells = [(f"{x:g}kpc", x) for x in c["flow_shells_kpc"]] + [(f"{x:g}R200", x * R200) for x in c["flow_shells_R200"]]
    for name, rs in shells:
        sh_ = np.flatnonzero(np.abs(g["r"] - rs) < 0.5 * th * rs)
        vr, m, Tsh = g["vr"][sh_].astype(F64), g["mass"][sh_].astype(F64), g["T"][sh_]
        mo = m * g["XO"][sh_]
        v2 = np.einsum("ij,ij->i", g["vel"][sh_].astype(F64), g["vel"][sh_].astype(F64))
        e = m * MSUN * (0.5 * v2 * 1e10 + g["u"][sh_])            # erg
        dr = th * rs
        phases = {"all": np.ones(len(sh_), bool), "cold": Tsh < t1, "warm": (Tsh >= t1) & (Tsh < t2), "hot": Tsh >= t2}
        for direction, dsel in (("Out", vr > 0), ("In", vr < 0)):
            for ph, pm in phases.items():
                q = dsel & pm
                speed = np.abs(vr[q])
                res[f"Flows/{direction}/Mass/{name}/{ph}"] = np.sum(m[q] * speed) / dr * KMS_PER_KPC_TO_PER_YR
                res[f"Flows/{direction}/OMass/{name}/{ph}"] = np.sum(mo[q] * speed) / dr * KMS_PER_KPC_TO_PER_YR
                res[f"Flows/{direction}/Energy/{name}/{ph}"] = np.sum(e[q] * speed) / dr * KMS_PER_KPC_TO_PER_YR / E51
                if direction == "Out":
                    p50, p90 = wquantile(vr[q], m[q], [0.5, 0.9])
                    res[f"Flows/Out/vr_p50/{name}/{ph}"] = p50
                    res[f"Flows/Out/vr_p90/{name}/{ph}"] = p90

    # -- supernovae from gal stars (LYRA SNTime convention; -1 populations never explode)
    for w in c["sn_windows_Myr"]:
        sel = stellar_evolution.died_within(S["tform"], S["sntime"], S["m_init"], t_now, w, massive=True)
        res[f"SN/N_{w:g}Myr"] = np.int64(sel.sum())
        try:
            res[f"SN/E_{w:g}Myr"] = float(np.sum(stellar_evolution.sn_energies(S["m_init"][sel], S["sntime"][sel]), dtype=F64)) / E51
        except FileNotFoundError as e:
            log(f"SN energies not computed (no Sukhbold table: {e})")
            res[f"SN/E_{w:g}Myr"] = NAN

    # -- dynamical and cooling times (lyratools.chemistry)
    for label, rr in (("0.1kpc", 0.1), ("1kpc", 1.0), ("R200", R200)):
        menc = sum(wsum(p["mass"], p["r"] < rr) for p in P.values() if p)
        res[f"Times/Dyn/{label}"] = chemistry.dynamical_time(menc, rr)
        sel = g["r"] < rr
        res[f"Times/Cool/{label}"] = (chemistry.cooling_time_within(g["mass"][sel], g["u"][sel], g["rho"][sel], g["cool"][sel], g["XH"][sel])
                                      if sel.any() and np.all(np.isfinite(g["cool"][sel])) else NAN)

    # -- contamination by low-resolution particles
    m_low = sum(wsum(P[pt]["mass"], P[pt]["r"] < R200) for pt in (2, 3) if P[pt])
    m_all = sum(wsum(p["mass"], p["r"] < R200) for p in P.values() if p)
    res["Contam_R200"] = m_low / m_all if (m_all > 0 and pt23 == "contamination") else NAN
    return res
