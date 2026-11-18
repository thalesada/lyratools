"""
Step 1b: table of the subhaloes around the main progenitor (group /Sub of the per-snapshot file),
with their most-bound particle IDs and descendants in the next snapshot.

Member stars and IDs are read as each subhalo's contiguous catalogue slice, so subhalo membership
comes from the global subhalo index (never from LocalSubhaloID).
"""
import numpy as np

from ..yields import twelve_log_oh
from .io import FORMATION_TIME, match_sorted, most_bound_ids, periodic, subhalo_ranges
from .main import oh_per_particle, wmean, wquantile, wstd, wsum

F64 = np.float64
NAN = np.nan
SUB_UNITS = dict(Pos="kpc, physical, relative to the main progenitor", Vel="km/s, peculiar difference + Hubble flow",
                 dist="kpc", v_rad="km/s", Mass="Msun", MassType="Msun", HalfmassRadType="kpc", Vmax="km/s",
                 VmaxRad="kpc", VelDisp="km/s", OH_gas_cat="12+log(O/H)", OH_star_cat="12+log(O/H)", M_init="Msun",
                 M_formed_10="Msun", M_formed_50="Msun", M_formed_100="Msun", AgeMedian="Myr", AgeP10="Myr", AgeP90="Myr",
                 OH_star_mean="12+log(O/H)", OH_star_std="dex", f_born_in_main="initial-mass fraction")


def selection_mask(lentype, dist_kpc, R200, R_sel_R200, min_dm, min_stars, exclude_pt23=True):
    """Within R_sel x R200, no low-resolution particles (if exclude_pt23), and >= min_dm DM or >= min_stars stars."""
    lt = np.asarray(lentype, np.int64)
    clean = (lt[:, 2] + lt[:, 3] == 0) if exclude_pt23 else True
    return (dist_kpc < R_sel_R200 * R200) & clean & ((lt[:, 1] >= min_dm) | (lt[:, 4] >= min_stars))


def select(cat, main_sub, center_code, R200, cfg, pt23="contamination"):
    c = cfg["sub"]
    if cat.nsubs == 0:
        return np.zeros(0, np.int64)
    dist = cat.subhalo_distance(center_code) * cat.kpc_per_code
    sel = selection_mask(cat["SubhaloLenType"], dist, R200, c["R_sel_R200"], c["min_dm"], c["min_stars"],
                         pt23 == "contamination")
    sel[main_sub] = True
    return np.flatnonzero(sel)


def dry_run_counts(cat, center_code, R200, cfg, pt23="contamination"):
    """{(R_sel, min_dm): (n_dark, n_luminous)} from catalogue fields only."""
    c = cfg["sub"]
    out = {}
    if cat.nsubs == 0:
        return {(r, m): (0, 0) for r in c["dry_run_R_sel"] for m in c["dry_run_min_dm"]}
    dist = cat.subhalo_distance(center_code) * cat.kpc_per_code
    lt = np.asarray(cat["SubhaloLenType"], np.int64)
    for r in c["dry_run_R_sel"]:
        for m in c["dry_run_min_dm"]:
            sel = selection_mask(lt, dist, R200, r, m, c["min_stars"], pt23 == "contamination")
            out[(r, m)] = (int(np.sum(sel & (lt[:, 4] == 0))), int(np.sum(sel & (lt[:, 4] > 0))))
    return out


def _cat_oh(cat, name, subs):
    if not cat.has(name):
        return np.full(len(subs), NAN)
    f = np.asarray(cat[name], F64)[subs]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = twelve_log_oh(f[:, 4], f[:, 0])
    return np.where(np.isfinite(v), v, NAN)


def catalogue_table(cat, subs, main_row, R200, hubble_kms_kpc):
    """Catalogue properties; positions [kpc] and velocities [km/s] physical, relative to the main progenitor."""
    k = cat.kpc_per_code
    grnr = np.asarray(cat["SubhaloGrNr"], np.int64)[subs]
    pos = periodic(np.asarray(cat["SubhaloPos"], F64)[subs] - main_row["Pos_code"], cat.box) * k
    # peculiar velocity difference plus Hubble flow
    vel = np.asarray(cat["SubhaloVel"], F64)[subs] - main_row["Vel"] + hubble_kms_kpc * pos
    dist = np.linalg.norm(pos, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vrad = np.einsum("ij,ij->i", pos, vel) / dist
    t = dict(SubhaloID=subs.astype(np.int64), GrNr=grnr,
             is_central=np.asarray(cat["GroupFirstSub"], np.int64)[grnr] == subs,
             is_main=subs == int(main_row["SubhaloID"]),
             Pos=pos, Vel=vel, dist=dist, v_rad=vrad, inside_R200=dist < R200,
             Mass=np.asarray(cat["SubhaloMass"], F64)[subs] * cat.msun_per_code,
             MassType=np.asarray(cat["SubhaloMassType"], F64)[subs] * cat.msun_per_code,
             LenType=np.asarray(cat["SubhaloLenType"], np.int64)[subs],
             HalfmassRadType=np.asarray(cat["SubhaloHalfmassRadType"], F64)[subs] * k,
             Vmax=np.asarray(cat["SubhaloVmax"], F64)[subs],
             VmaxRad=np.asarray(cat["SubhaloVmaxRad"], F64)[subs] * k,
             VelDisp=np.asarray(cat["SubhaloVelDisp"], F64)[subs],
             OH_gas_cat=_cat_oh(cat, "SubhaloGasMetalFractions", subs),
             OH_star_cat=_cat_oh(cat, "SubhaloStarMetalFractions", subs))
    return t


def star_table(sn, cat, subs, t_now, track, cfg):
    """Properties of member stars (GFM_InitialMass > 0) per subhalo, in the order of subs."""
    c = cfg["sub"]
    n = len(subs)
    out = {k: np.full(n, NAN) for k in ["M_init", "AgeMedian", "AgeP10", "AgeP90", "OH_star_mean", "OH_star_std",
                                        "f_born_in_main"] + [f"M_formed_{w:g}" for w in c["formed_windows_Myr"]]}
    out["N_stars"] = np.zeros(n, np.int64)
    ranges, ln = subhalo_ranges(cat, subs, 4)
    if ln.sum() == 0:
        for w in c["formed_windows_Myr"]:
            out[f"M_formed_{w:g}"][:] = 0.0
        out["M_init"][:] = 0.0
        return out
    m_init = sn.read_ranges(4, "GFM_InitialMass", ranges).astype(F64) * sn.msun_per_code
    mass = sn.masses(4, ranges=ranges)
    aform = sn.read_ranges(4, sn.field_name(4, FORMATION_TIME), ranges).astype(F64)
    met = sn.read_ranges(4, "GFM_Metals", ranges, cols=[0, 4]) if sn.has(4, "GFM_Metals") else np.full((len(mass), 2), NAN)
    bpos = sn.read_ranges(4, "BirthPos", ranges).astype(F64) * sn.ckpc_per_code if sn.has(4, "BirthPos") else None
    oh = oh_per_particle(met[:, 1], met[:, 0])
    valid = (m_init > 0) & (aform > 0)
    tform = sn.t_myr(np.where(aform > 0, aform, 1.0))
    age = np.where(valid, t_now - tform, NAN)

    # born in the main progenitor: BirthPos within born_in_main_R200 x R200 of the track (comoving,
    # interpolated in a, or in time for isolated runs); stars formed before the first tracked snapshot are not counted
    born = np.full(len(mass), NAN)
    if bpos is not None and len(track["a"]) > 1:
        x, ta = (aform, np.asarray(track["a"], F64)) if sn.cosmological else (tform, np.asarray(track["t_Myr"], F64))
        intrack = valid & (x >= ta[0]) & (x <= ta[-1])
        cen = np.column_stack([np.interp(x[intrack], ta, track["Pos_ckpc"][:, j]) for j in range(3)])
        r200 = np.interp(x[intrack], ta, track["R200c_ckpc"])
        dd = periodic(bpos[intrack] - cen, sn.box * sn.ckpc_per_code)
        born[intrack] = np.linalg.norm(dd, axis=1) < c["born_in_main_R200"] * r200

    bounds = np.concatenate([[0], np.cumsum(ln)])
    for i in range(n):
        sl = slice(bounds[i], bounds[i + 1])
        v = valid[sl]
        mi, mc, ag = m_init[sl][v], mass[sl][v], age[sl][v]
        out["N_stars"][i] = v.sum()
        out["M_init"][i] = wsum(mi)
        for w in c["formed_windows_Myr"]:
            out[f"M_formed_{w:g}"][i] = wsum(mi, ag < w)
        if v.any():
            out["AgeP10"][i], out["AgeMedian"][i], out["AgeP90"][i] = wquantile(ag, mi, [0.1, 0.5, 0.9])
            o, wo = oh[sl][v], mc
            ok = np.isfinite(o)
            out["OH_star_mean"][i] = wmean(o[ok], wo[ok])
            out["OH_star_std"][i] = wstd(o[ok], wo[ok])
            b = born[sl][v]
            okb = np.isfinite(b)
            if okb.any():
                out["f_born_in_main"][i] = wsum(mi[okb] * b[okb]) / wsum(mi[okb])
    return out


def track_ids(sn, cat, subs, mode, n_track):
    """Ragged most-bound IDs per subhalo (N_track DM then N_track stars): ids, rank, type, offsets."""
    ids, rank, ptype, offsets = [], [], [], [0]
    for s in subs:
        k = 0
        for pt in (1, 4):
            x = most_bound_ids(sn, cat, int(s), pt, n_track, mode)
            ids.append(np.asarray(x, np.uint64))
            rank.append(np.arange(len(x), dtype=np.int32))
            ptype.append(np.full(len(x), pt, np.int8))
            k += len(x)
        offsets.append(offsets[-1] + k)
    cat_ = (lambda a, dt: np.concatenate(a).astype(dt) if a else np.zeros(0, dt))
    return cat_(ids, np.uint64), cat_(rank, np.int32), cat_(ptype, np.int8), np.array(offsets, np.int64)


def find_descendants(ids, rank, ptype, offsets, sn1, cat1, center1_code, radius_code, alpha):
    """
    Descendant in the next snapshot of each subhalo: the subhalo with the largest rank-weighted share
    of its track IDs, among subhaloes with SubhaloPos within radius_code of center1_code.
    -1 if the unmatched (field or not found) weight is larger. Returns (DescSubhaloID, DescShare).
    """
    nrow = len(offsets) - 1
    lab = np.full(len(ids), -1, np.int64)
    if cat1.nsubs:
        cand = np.flatnonzero(cat1.subhalo_distance(center1_code) < radius_code)
        for pt in (1, 4):
            ranges, ln = subhalo_ranges(cat1, cand, pt)
            if ln.sum() == 0 or not np.any(ptype == pt):
                continue
            pid = sn1.read_ranges(pt, "ParticleIDs", ranges)
            plab = np.repeat(cand, ln)
            o = np.argsort(pid, kind="stable")
            want = np.flatnonzero(ptype == pt)
            pos, hit = match_sorted(pid[o], ids[want])
            lab[want[hit]] = plab[o[pos[hit]]]
    w = (rank.astype(F64) + 1.0) ** -alpha
    desc = np.full(nrow, -1, np.int64)
    share = np.zeros(nrow)
    for i in range(nrow):
        sl = slice(offsets[i], offsets[i + 1])
        li, wi = lab[sl], w[sl]
        tot = wi.sum()
        m = li >= 0
        if not m.any() or tot <= 0:
            continue
        u, inv = np.unique(li[m], return_inverse=True)
        sc = np.bincount(inv, weights=wi[m])
        j = int(np.argmax(sc))
        share[i] = sc[j] / tot
        if sc[j] >= tot - sc.sum():
            desc[i] = u[j]
    return desc, share
