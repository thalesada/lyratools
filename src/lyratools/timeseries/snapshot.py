"""Step 1: one per-snapshot file, <out>/snaps/snap_XXX.h5, with groups /Main and /Sub."""
import os
import resource
import sys
import time

import h5py
import numpy as np

from .config import to_json
from .io import Catalogue, Snapshot
from .main import compute_main, load_main, units_of
from .subhalos import SUB_UNITS, catalogue_table, find_descendants, select, star_table, track_ids


def snap_path(out, snap):
    return os.path.join(out, "snaps", f"snap_{snap:03d}.h5")


def file_status(path, cfg_hash):
    """'ok', 'missing', 'unreadable', 'incomplete' or 'config' (written with another config)."""
    if not os.path.exists(path):
        return "missing"
    try:
        with h5py.File(path, "r") as f:
            if not f.attrs.get("complete", False):
                return "incomplete"
            if f.attrs.get("config_hash") != cfg_hash:
                return "config"
    except Exception:
        return "unreadable"
    return "ok"


def _peak_memory_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def _store(group, name, value, dtype=None, units=None):
    v = np.asarray(value)
    if dtype is not None:
        v = v.astype(dtype)
    elif v.dtype.kind == "f":
        v = v.astype(np.float32)
    elif v.dtype.kind == "b":
        v = v.astype(np.int8)
    d = group.create_dataset(name, data=v)
    if units:
        d.attrs["units"] = units


def process_snapshot(snap, ctx, log=print):
    """Compute and write the per-snapshot file. ctx: see run.make_context."""
    t0 = time.time()
    cfg, track = ctx["config"], ctx["track"]
    k = int(np.flatnonzero(track["Snap"] == snap)[0])
    row = {key: track[key][k] for key in track}
    row["Pos_code"] = np.asarray(row["Pos_code"], np.float64)
    row["Vel"] = np.asarray(row["Vel"], np.float64)
    t_prev = float(track["t_Myr"][k - 1]) if k > 0 else np.nan
    run = ctx["run"]
    cosmo, pt23 = run["cosmological"], run["pt23"]
    with_sub = run["center"] == "subfind" and snap in ctx["cat_files"]
    with Snapshot(ctx["snap_files"][snap], cosmological=cosmo) as sn:
        t_now = sn.t_now
        R200 = float(row["R200c_ckpc"]) * sn.a
        who = f"main subhalo {row['SubhaloID']}" if run["center"] == "subfind" else "shrinking-sphere centre"
        log(f"snap {snap}: a={sn.a:.5f} z={sn.z:.3f} t={t_now:.1f} Myr, {who}, R200={R200:.2f} kpc")

        # -- 1a main progenitor
        P = load_main(sn, row["Pos_code"], row["Vel"], cfg["main"]["load_R200"] * R200, t_now, pt23)
        log("loaded " + ", ".join(f"{'PartType%d' % pt if isinstance(pt, int) else 'IC stars'}: {len(p.get('mass', []))}"
                                  for pt, p in P.items()))
        main = compute_main(P, R200, t_now, t_prev, cfg, log, pt23)
        del P

        # -- 1b substructure (needs the group catalogue)
        if with_sub:
            cat = Catalogue(ctx["cat_files"][snap], units=sn.units, cosmological=cosmo)
            subs = select(cat, int(row["SubhaloID"]), row["Pos_code"], R200, cfg, pt23)
            log(f"{len(subs)} subhaloes selected")
            table = catalogue_table(cat, subs, row, R200, sn.hubble_kms_kpc)
            table.update(star_table(sn, cat, subs, t_now, ctx["track"], cfg))
            ids, rank, ptype, offsets = track_ids(sn, cat, subs, ctx["rank_mode"], cfg["sub"]["N_track"])
        else:
            log("no group catalogue: /Sub skipped")
        a_now, z_now = sn.a, sn.z

    # -- descendants in the next tracked snapshot
    if with_sub:
        nxt = int(track["Snap"][k + 1]) if k + 1 < len(track["Snap"]) else None
        if nxt is not None and nxt in ctx["cat_files"]:
            with Snapshot(ctx["snap_files"][nxt], cosmological=cosmo) as sn1:
                cat1 = Catalogue(ctx["cat_files"][nxt], units=sn1.units, cosmological=cosmo)
                c = cfg["sub"]
                radius = (c["R_sel_R200"] + c["margin_R200"]) * float(row["R200c_ckpc"]) / cat1.ckpc_per_code
                desc, share = find_descendants(ids, rank, ptype, offsets, sn1, cat1, np.asarray(track["Pos_code"][k + 1], np.float64),
                                               radius, c["rank_alpha"])
        else:
            desc, share = np.full(len(subs), -1, np.int64), np.full(len(subs), np.nan)
        table["DescSubhaloID"], table["DescShare"] = desc, share

    path = snap_path(ctx["out"], snap)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        gm = f.create_group("Main")
        for name, v in main.items():
            _store(gm, name, v, units=units_of(name))
        gm["Phases/nH_hist"].attrs["log_nH_edges"] = np.arange(cfg["main"]["nH_hist"][0], cfg["main"]["nH_hist"][1] + 1e-9,
                                                                cfg["main"]["nH_hist"][2])
        if with_sub:
            gs = f.create_group("Sub")
            for name, v in table.items():
                _store(gs, name, v, units=SUB_UNITS.get(name))
            _store(gs, "TrackIDs", ids, np.uint64)
            _store(gs, "TrackRank", rank, np.int32)
            _store(gs, "TrackType", ptype, np.int8)
            _store(gs, "TrackOffsets", offsets, np.int64)
        f.attrs.update(snap=snap, a=a_now, z=z_now, cosmological=int(cosmo), t_Myr=t_now, config=to_json(cfg), config_hash=ctx["config_hash"],
                       runtime_s=time.time() - t0, peak_memory_bytes=_peak_memory_bytes(), **ctx["provenance"])
        f.attrs["complete"] = True
    os.replace(tmp, path)
    log(f"wrote {path} in {time.time() - t0:.1f} s, peak memory {_peak_memory_bytes() / 1e9:.2f} GB")
    return path
