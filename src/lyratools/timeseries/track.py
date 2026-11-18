"""
Step 0: the main-progenitor track, followed backwards from the final snapshot by matching the
most-bound DM particles (rank-weighted, as SubLink, Rodriguez-Gomez et al. 2015).

Per snapshot, only the DM ParticleIDs of the subhaloes whose SubhaloPos lies within
search_R200 x R200 (comoving, of the later snapshot) of the previous centre are read, as
contiguous catalogue slices; their SubhaloID follows from the slice, so no coordinates are needed.

Without group catalogues (run.center = "shrink", e.g. isolated galaxies), the centre is found by a
shrinking sphere on the particles, R200 from the enclosed mean density, and the velocity frame from
the mass-weighted velocity within vel_R200 x R200.
"""
import os
import time

import h5py
import numpy as np

from ..constants import KPC, MSUN
from .config import DEFAULT_CONFIG, config_hash, to_json
from .io import Catalogue, Snapshot, binding_order_ok, match_sorted, most_bound_ids, resolve_cosmological, subhalo_ranges

G_CGS = 6.6743e-8

TRACK_FIELDS = ["Snap", "a", "z", "t_Myr", "GroupID", "SubhaloID", "is_central", "Pos_code", "Pos_ckpc", "Vel",
                "M200c", "R200c", "R200c_ckpc", "SubhaloMassType", "SubhaloLenType", "SeedFraction"]


def track_hash(cfg):
    if cfg.get("run", DEFAULT_CONFIG["run"]) == DEFAULT_CONFIG["run"]:
        return config_hash(cfg["track"])
    return config_hash({"track": cfg["track"], "run": cfg["run"]})


def resolve_run(cfg, snap_files, cat_files):
    """
    The run settings with "auto" resolved: cosmological (bool), center ("subfind" or "shrink"),
    pt23 ("contamination", "stars" or "ignore"), plus a description for the log.
    """
    r = dict(cfg["run"])
    first = snap_files[min(snap_files)]
    r["cosmological"], why = resolve_cosmological(r["cosmological"], first)
    have_cats = bool(set(snap_files) & set(cat_files))
    if r["center"] == "auto":
        if not have_cats and r["cosmological"]:     # most likely a wrong --groupdir: do not fall back silently
            raise FileNotFoundError("no snapshot number has both particle data and a group catalogue "
                                    "(set run.center = \"shrink\" to centre on particles instead)")
        r["center"] = "subfind" if have_cats else "shrink"
    elif r["center"] == "subfind" and not have_cats:
        raise FileNotFoundError("run.center = 'subfind' but no snapshot number has both particle data and a group catalogue")
    if r["pt23"] == "auto":
        r["pt23"] = "contamination" if r["cosmological"] else "stars"
    r["description"] = (f"{'cosmological' if r['cosmological'] else 'non-cosmological'} run ({why}); "
                        f"centre from {'SUBFIND' if r['center'] == 'subfind' else 'a shrinking sphere'}; "
                        f"PartType2/3 treated as {r['pt23']}")
    return r


def _row(snap, sn, cat, sub, share):
    grnr = int(cat["SubhaloGrNr"][sub])
    r200 = float(cat["Group_R_Crit200"][grnr])
    pos = np.asarray(cat["SubhaloPos"][sub], np.float64)
    return dict(Snap=snap, a=sn.a, z=sn.z, t_Myr=sn.t_now, GroupID=grnr, SubhaloID=int(sub),
                is_central=int(cat["GroupFirstSub"][grnr]) == sub, Pos_code=pos, Pos_ckpc=pos * cat.ckpc_per_code,
                Vel=np.asarray(cat["SubhaloVel"][sub], np.float64),
                M200c=float(cat["Group_M_Crit200"][grnr]) * cat.msun_per_code,
                R200c=r200 * cat.kpc_per_code, R200c_ckpc=r200 * cat.ckpc_per_code,
                SubhaloMassType=np.asarray(cat["SubhaloMassType"][sub], np.float64) * cat.msun_per_code,
                SubhaloLenType=np.asarray(cat["SubhaloLenType"][sub], np.int64), SeedFraction=share)


def find_progenitor(sn, cat, seed, center_code, radius_code, alpha, first_block=1 << 16):
    """
    Subhalo holding the largest rank-weighted share of the seed IDs (seed[0] most bound), among
    the subhaloes with SubhaloPos within radius_code. Returns (subhalo or -1, share, DM IDs read).

    Each candidate's IDs are read from the start of its slice (most bound first) in doubling blocks,
    stopping once every seed ID has been found: IDs are unique, so the result equals a full read.
    """
    if cat.nsubs == 0 or len(seed) == 0:
        return -1, 0.0, 0
    cand = np.flatnonzero((cat.subhalo_distance(center_code) < radius_code)
                          & (np.asarray(cat["SubhaloLenType"])[:, 1] > 0))
    if len(cand) == 0:
        return -1, 0.0, 0
    ranges, ln = subhalo_ranges(cat, cand, 1)
    start = np.array([r[0] for r in ranges], np.int64)
    order = np.argsort(seed, kind="stable")
    sseed = seed[order]
    label = np.full(len(seed), -1, np.int64)          # subhalo holding each seed (sorted order)
    done, nread, block = 0, 0, first_block
    while True:
        lo = np.minimum(done, ln)
        hi = np.minimum(done + block, ln)
        todo = np.flatnonzero(hi > lo)
        if len(todo) == 0:
            break
        part = [(int(start[i] + lo[i]), int(start[i] + hi[i])) for i in todo]
        ids = sn.read_ranges(1, "ParticleIDs", part)
        nread += len(ids)
        pos, hit = match_sorted(sseed, ids)
        owner = np.repeat(cand[todo], hi[todo] - lo[todo])
        label[pos[hit]] = owner[hit]
        if np.all(label >= 0):
            break
        done, block = done + block, 2 * block
    found = label >= 0
    if not found.any():
        return -1, 0.0, nread
    weights = (order[found] + 1.0) ** -alpha
    subs, inv = np.unique(label[found], return_inverse=True)
    score = np.bincount(inv, weights=weights)
    j = int(np.argmax(score))
    return int(subs[j]), float(score[j] / np.sum((np.arange(len(seed)) + 1.0) ** -alpha)), nread


# ------------------------------------------------------------------ particle-based centre (no catalogue)
def _centre_types(sn, run):
    want = {"stars": [4] + ([2, 3] if run["pt23"] == "stars" else []), "dm": [1], "all": list(range(6))}[run["center_on"]]
    for types in (want, [1], list(range(6))):
        types = [pt for pt in types if sn.total(pt)]
        if types:
            return types
    raise ValueError("snapshot has no particles")


def shrinking_sphere(rel, m, r0, factor, min_n):
    """Shift [same units as rel] of the centre: centre of mass within r, shrinking r by factor until < min_n particles."""
    shift = np.zeros(3)
    r = r0
    rel = np.asarray(rel, np.float64)
    m = np.asarray(m, np.float64)
    while True:
        d = rel - shift
        sel = np.einsum("ij,ij->i", d, d) < r * r
        n = int(sel.sum())
        if n == 0:
            break
        shift = shift + np.sum(d[sel] * m[sel, None], axis=0) / m[sel].sum()
        if n < min_n:
            break
        r *= factor
    return shift


def rho_crit_msun_kpc3(sn, run):
    if sn.cosmological:
        return float(sn.tc.cosmology.critical_density(sn.z).to("Msun/kpc3").value)
    H = run["H0_kms_Mpc"] / 3.085678e19                                     # 1/s
    return 3 * H * H / (8 * np.pi * G_CGS) * KPC ** 3 / MSUN


def particle_centre(snap, sn, run, guess_code, log=print):
    """Track row (same fields as from the catalogue) from the particles of one snapshot."""
    types = _centre_types(sn, run)
    if guess_code is None:
        X = np.concatenate([sn.read_ranges(pt, "Coordinates", [(0, sn.total(pt))]).astype(np.float64) for pt in types])
        M = np.concatenate([sn.masses(pt, ranges=[(0, sn.total(pt))]) for pt in types])
        c = np.sum(X * M[:, None], axis=0) / M.sum()
        rel = X - c
        r0 = float(np.sqrt(np.max(np.einsum("ij,ij->i", rel, rel)))) * 1.0001
        del X
    else:
        c = np.asarray(guess_code, np.float64)
        r0 = run["shrink_r0_kpc"] / sn.kpc_per_code
        parts = [sn.scan_sphere(pt, c, r0) for pt in types]
        rel = np.concatenate([p[1] for p in parts])
        M = np.concatenate([sn.masses(pt, idx) for pt, (idx, _) in zip(types, parts)])
        if len(M) == 0:
            return None
    c = c + shrinking_sphere(rel, M, r0, run["shrink_factor"], run["shrink_min_n"])
    if sn.box > 0:
        c = c % sn.box
    del rel

    # enclosed mass profile of all types -> R200, masses within it, velocity frame
    rs = run["R200_search_kpc"] / sn.kpc_per_code
    prof = {}
    for pt in range(6):
        if sn.total(pt):
            idx, rel = sn.scan_sphere(pt, c, rs)
            prof[pt] = (idx, np.sqrt(np.einsum("ij,ij->i", rel, rel)) * sn.kpc_per_code, sn.masses(pt, idx))
    r_all = np.concatenate([p[1] for p in prof.values()])
    m_all = np.concatenate([p[2] for p in prof.values()])
    o = np.argsort(r_all, kind="stable")
    r_s, cum = r_all[o], np.cumsum(m_all[o], dtype=np.float64)
    if run["R200_kpc"] == "auto":
        with np.errstate(divide="ignore"):
            dens = cum / (4.0 / 3.0 * np.pi * r_s ** 3)
        above = np.flatnonzero(dens >= 200 * rho_crit_msun_kpc3(sn, run))
        if len(above) == 0:
            raise ValueError(f"snap {snap}: mean density never reaches 200 rho_crit within {run['R200_search_kpc']} kpc; "
                             "set run.R200_kpc")
        R200 = float(r_s[above[-1]])
        if above[-1] == len(r_s) - 1:
            log(f"warning: snap {snap}: R200 = {R200:.1f} kpc is at the edge of the search sphere; increase run.R200_search_kpc")
    else:
        R200 = float(run["R200_kpc"])
    mt, lt = np.zeros(6), np.zeros(6, np.int64)
    vsum, msum = np.zeros(3), 0.0
    for pt, (idx, r, m) in prof.items():
        inside = r < R200
        mt[pt], lt[pt] = np.sum(m[inside], dtype=np.float64), int(inside.sum())
        sel = r < run["vel_R200"] * R200
        if sel.any():
            v = sn.read_indices(pt, "Velocities", idx[sel]).astype(np.float64) * np.sqrt(sn.a)
            vsum += np.sum(v * m[sel, None], axis=0)
            msum += np.sum(m[sel], dtype=np.float64)
    vel = vsum / msum if msum > 0 else np.zeros(3)
    return dict(Snap=snap, a=sn.a, z=sn.z, t_Myr=sn.t_now, GroupID=-1, SubhaloID=-1, is_central=False,
                Pos_code=c, Pos_ckpc=c * sn.ckpc_per_code, Vel=vel, M200c=float(mt.sum()), R200c=R200,
                R200c_ckpc=R200 / sn.a, SubhaloMassType=mt, SubhaloLenType=lt, SeedFraction=np.nan)


def build_track_particles(snap_files, cfg, run, log=print):
    """Shrinking-sphere track over every snapshot, backwards from the last, each started at the later centre."""
    rows = []
    for snap in sorted(snap_files, reverse=True):
        t0 = time.time()
        with Snapshot(snap_files[snap], cosmological=run["cosmological"]) as sn:
            if rows:
                guess = rows[-1]["Pos_code"]
            elif run["center_guess_kpc"] is not None:
                guess = np.asarray(run["center_guess_kpc"], np.float64) / sn.kpc_per_code
            else:
                guess = None
            row = particle_centre(snap, sn, run, guess, log)
        if row is None:
            log(f"snap {snap}: no particles within {run['shrink_r0_kpc']} kpc of the later centre; track starts at {rows[-1]['Snap']}")
            break
        rows.append(row)
        log(f"snap {snap}: t={row['t_Myr']:.1f} Myr, centre {np.round(row['Pos_ckpc'], 3)} ckpc, R200={row['R200c']:.2f} kpc, "
            f"M200={row['M200c']:.3g} Msun, v={np.round(row['Vel'], 1)} km/s, {time.time() - t0:.1f} s")
    rows = rows[::-1]
    return {k: np.array([r[k] for r in rows]) for k in TRACK_FIELDS}


# ------------------------------------------------------------------ track
def build_track(snap_files, cat_files, cfg, log=print):
    """
    Follow the main progenitor of FoF group cfg['track']['halo'] back from the last snapshot
    (SUBFIND), or by shrinking spheres when there are no catalogues (run.center).
    Returns (dict of arrays in increasing snapshot order, rank_mode: "binding", "distance" or "shrink").
    """
    run = resolve_run(cfg, snap_files, cat_files)
    log(run["description"])
    if run["center"] == "shrink":
        return build_track_particles(snap_files, cfg, run, log), "shrink"
    cosmo = run["cosmological"]
    t = cfg["track"]
    snaps = sorted(set(snap_files) & set(cat_files))
    if not snaps:
        raise FileNotFoundError("no snapshot with both particle data and a group catalogue")
    gaps = sorted(set(range(snaps[0], snaps[-1] + 1)) - set(snaps))
    if gaps:
        log(f"warning: snapshots without particle data or catalogue are skipped: {gaps}")
    final = snaps[-1]
    rows = []
    with Snapshot(snap_files[final], cosmological=cosmo) as sn:
        cat = Catalogue(cat_files[final], units=sn.units, cosmological=cosmo)
        ok, bad = binding_order_ok(sn, cat, t["check_n_subhalos"])
        mode = "binding" if ok else "distance"
        if not ok:
            log(f"warning: SubhaloIDMostbound is not the first particle of subhaloes {bad}: "
                "particles are not ordered by binding energy; ranking by distance to SubhaloPos instead")
        target = int(cat["GroupFirstSub"][t["halo"]])
        if target < 0:
            raise ValueError(f"FoF group {t['halo']} has no subhalo at snapshot {final}")
        seed = most_bound_ids(sn, cat, target, 1, t["N_seed"], mode)
        rows.append(_row(final, sn, cat, target, 1.0))
        log(f"snap {final}: target subhalo {target} of FoF {t['halo']}, {len(seed)} seed IDs")
    for snap in reversed(snaps[:-1]):
        t0 = time.time()
        prev = rows[-1]
        with Snapshot(snap_files[snap], cosmological=cosmo) as sn:
            cat = Catalogue(cat_files[snap], units=sn.units, cosmological=cosmo)
            radius = max(t["search_R200"] * prev["R200c_ckpc"], t["min_search_ckpc"]) / cat.ckpc_per_code
            prog, share, nread = find_progenitor(sn, cat, seed, prev["Pos_code"], radius, t["rank_alpha"])
            if prog < 0:
                log(f"snap {snap}: no seed IDs found; track starts at {rows[-1]['Snap']}")
                break
            ndm = int(cat["SubhaloLenType"][prog][1])
            if ndm < t["min_dm"]:
                log(f"snap {snap}: progenitor {prog} has {ndm} < {t['min_dm']} DM particles; track starts at {rows[-1]['Snap']}")
                break
            rows.append(_row(snap, sn, cat, prog, share))
            log(f"snap {snap}: progenitor subhalo {prog} (FoF {rows[-1]['GroupID']}, "
                f"{'central' if rows[-1]['is_central'] else 'satellite'}), seed share {share:.3f}, "
                f"{nread} DM IDs read, "
                f"{time.time() - t0:.1f} s")
            seed = most_bound_ids(sn, cat, prog, 1, t["N_seed"], mode)
    rows = rows[::-1]
    return {k: np.array([r[k] for r in rows]) for k in TRACK_FIELDS}, mode


def save_track(path, track, cfg, mode):
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        for k, v in track.items():
            v = np.asarray(v)
            if v.dtype.kind == "f" and k not in ("Pos_code", "a", "t_Myr", "z"):
                v = v.astype(np.float32)
            f.create_dataset(k, data=v)
        f.attrs.update(config=to_json(cfg), track_hash=track_hash(cfg), rank_mode=mode, halo=cfg["track"]["halo"])
    os.replace(tmp, path)


def load_track(path):
    """(dict of arrays, attrs) from main_track.h5."""
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f}, dict(f.attrs)


def cross_check(track, groupdir, log=print):
    """Compare with SubfindEvolution (FoF 0 at every snapshot); returns the snapshots that differ."""
    from ..evolution import SubfindEvolution
    try:
        evo = SubfindEvolution(groupdir)
    except Exception as e:
        log(f"cross-check with SubfindEvolution skipped: {e}")
        return []
    differ = []
    for i, snap in enumerate(track["Snap"]):
        j = np.flatnonzero(evo.snap == snap)
        if len(j) == 0 or not np.all(np.isfinite(evo.sub_pos_ckpc[j[0]])):
            differ.append((int(snap), "not in SubfindEvolution"))
            continue
        d = np.linalg.norm(track["Pos_ckpc"][i] - evo.sub_pos_ckpc[j[0]])
        if track["GroupID"][i] != 0:
            differ.append((int(snap), f"progenitor in FoF {track['GroupID'][i]}, not FoF 0"))
        elif d > 0.1 * track["R200c_ckpc"][i]:
            differ.append((int(snap), f"centre {d:.2f} ckpc from FoF 0's GroupPos"))
    if differ:
        log("main track differs from SubfindEvolution.halo_track() (FoF 0) at:")
        for snap, why in differ:
            log(f"  snap {snap}: {why}")
    else:
        log("main track agrees with SubfindEvolution.halo_track() at every snapshot")
    return differ
