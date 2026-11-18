"""
Step 2: stack the per-snapshot files into <out>/timeseries_<sim>.h5 and build subhalo tracks
from the descendant pointers.
"""
import os

import h5py
import numpy as np

from .config import to_json
from .snapshot import file_status, snap_path

FATES = ("survives", "merged_main", "merged_other", "lost")
TRACK_ARRAYS = {   # name in the track arrays: column of the per-snapshot /Sub table
    "dist": "dist", "v_rad": "v_rad", "Mass": "Mass", "MassType": "MassType", "Vmax": "Vmax",
    "AgeMedian": "AgeMedian", "OH_star_mean": "OH_star_mean", "OH_star_std": "OH_star_std",
    "M_formed_10": "M_formed_10", "DescShare": "DescShare",
}


def check_files(out, snaps, cfg_hash):
    """{snap: problem} for the per-snapshot files that are not usable."""
    bad = {}
    for s in snaps:
        st = file_status(snap_path(out, s), cfg_hash)
        if st != "ok":
            bad[int(s)] = st
    return bad


def build_tracks(tables, main_ids):
    """
    Link the per-snapshot subhalo tables into tracks.

    tables:   list (in snapshot order) of dicts with SubhaloID, Mass, DescSubhaloID
    main_ids: main-progenitor SubhaloID per snapshot (its track always continues along the main track)
    Returns (row_track: list of arrays of track index per row, scalars dict, mismatches).
    """
    nsnap = len(tables)
    row_track = [np.full(len(t["SubhaloID"]), -1, np.int64) for t in tables]
    first, last, fate, merged_into, is_main = [], [], [], [], []

    def new_track(k, main=False):
        first.append(k); last.append(k); fate.append("survives"); merged_into.append(-1); is_main.append(main)
        return len(first) - 1

    for i, sid in enumerate(tables[0]["SubhaloID"]):
        row_track[0][i] = new_track(0, sid == main_ids[0])
    mismatches = []
    for k in range(nsnap - 1):
        t0, t1 = tables[k], tables[k + 1]
        where1 = {int(s): j for j, s in enumerate(t1["SubhaloID"])}
        desc = np.array(t0["DescSubhaloID"], np.int64)
        imain = np.flatnonzero(t0["SubhaloID"] == main_ids[k])
        if len(imain):
            if desc[imain[0]] != main_ids[k + 1]:
                mismatches.append((k, int(desc[imain[0]]), int(main_ids[k + 1])))
            desc[imain[0]] = main_ids[k + 1]                 # the main track follows Step 0
        progs = {}
        for i, d in enumerate(desc):
            tr = row_track[k][i]
            if d < 0 or int(d) not in where1:
                fate[tr] = "lost"
                continue
            progs.setdefault(where1[int(d)], []).append(i)
        for j, plist in progs.items():
            # the main progenitor continues; otherwise the most massive
            cont = max(plist, key=lambda i: (t0["SubhaloID"][i] == main_ids[k], t0["Mass"][i]))
            tr_c = row_track[k][cont]
            row_track[k + 1][j] = tr_c
            last[tr_c] = k + 1
            for i in plist:
                if i != cont:
                    tr = row_track[k][i]
                    fate[tr] = "merged"
                    merged_into[tr] = tr_c
        for j in range(len(t1["SubhaloID"])):
            if row_track[k + 1][j] < 0:
                row_track[k + 1][j] = new_track(k + 1, t1["SubhaloID"][j] == main_ids[k + 1])
    for tr in range(len(fate)):
        if fate[tr] == "merged":
            fate[tr] = "merged_main" if is_main[merged_into[tr]] else "merged_other"
    scalars = dict(first=np.array(first), last=np.array(last), fate=np.array(fate), merged_into=np.array(merged_into, np.int64),
                   is_main=np.array(is_main, bool))
    return row_track, scalars, mismatches


def _stack_main(files, out_group):
    """Stack every /Main dataset to (Nsnap, ...), NaN (floats) or -1 (integers) where undefined."""
    shapes = {}

    def collect(name, obj):         # must return None, or visititems stops
        if isinstance(obj, h5py.Dataset) and name not in shapes:
            shapes[name] = (obj.shape, obj.dtype, dict(obj.attrs))

    for fn in files:
        with h5py.File(fn, "r") as f:
            f["Main"].visititems(collect)
    for name, (shape, dtype, attrs) in sorted(shapes.items()):
        kind = np.dtype(dtype).kind
        arr = np.full((len(files),) + shape, -1 if kind in "iu" else np.nan, dtype if kind in "iu" else np.float32)
        for i, fn in enumerate(files):
            with h5py.File(fn, "r") as f:
                if name in f["Main"] and f["Main"][name].shape == shape:
                    arr[i] = f["Main"][name][()]
        out_group.create_dataset(name, data=arr).attrs.update(attrs)


def combine(out, sim, track, cfg, cfg_hash, provenance, log=print):
    """Write <out>/timeseries_<sim>.h5. Returns 0, or 1 (nothing written) if any snapshot file is unusable."""
    snaps = [int(s) for s in track["Snap"]]
    bad = check_files(out, snaps, cfg_hash)
    if bad:
        log(f"cannot combine: {len(bad)} of {len(snaps)} snapshot files are missing or unusable:")
        for s, why in sorted(bad.items()):
            log(f"  snap {s:03d}: {why}")
        return 1
    files = [snap_path(out, s) for s in snaps]
    path = os.path.join(out, f"timeseries_{sim}.h5")
    tmp = path + ".tmp"
    tables = []
    with h5py.File(tmp, "w") as fo:
        h = fo.create_group("Header")
        attrs, cosmological = [], 1
        has_sub = True
        for fn in files:
            with h5py.File(fn, "r") as f:
                has_sub &= "Sub" in f
        for fn in files:
            with h5py.File(fn, "r") as f:
                attrs.append((f.attrs["snap"], f.attrs["a"], f.attrs["z"], f.attrs["t_Myr"]))
                cosmological = int(f.attrs.get("cosmological", 1))
                if has_sub:
                    tables.append({k: f["Sub"][k][()] for k in f["Sub"]})
                    fo.copy(f["Sub"], f"Sub/Catalogue/snap_{int(f.attrs['snap']):03d}")
        snap_arr, a, z, t = (np.array(x) for x in zip(*attrs))
        h.create_dataset("Snap", data=snap_arr.astype(np.int64))
        for name, v in (("a", a), ("z", z), ("t_Myr", t)):
            h.create_dataset(name, data=v.astype(np.float32))
        _stack_main(files, fo.create_group("Main"))
        mt = fo.create_group("MainTrack")
        for k, v in track.items():
            mt.create_dataset(k, data=v)

        fo.attrs.update(sim=sim, config=to_json(cfg), config_hash=cfg_hash, n_snap=len(snaps), cosmological=cosmological,
                        **provenance)
        if not has_sub:
            fo.attrs.update(n_tracks=0, main_desc_mismatches=0)
            log("no /Sub in the snapshot files (no group catalogues): subhalo tracks skipped")
    if not has_sub:
        os.replace(tmp, path)
        log(f"wrote {path}: {len(snaps)} snapshots, no subhalo tracks")
        return 0
    with h5py.File(tmp, "a") as fo:
        main_ids = np.asarray(track["SubhaloID"], np.int64)
        row_track, sc, mismatches = build_tracks(tables, main_ids)
        if mismatches:
            log("main-progenitor descendant does not match the Step 0 track at:")
            for k, d, m in mismatches:
                log(f"  snap {snaps[k]:03d} -> {snaps[k + 1]:03d}: descendant {d}, track {m}")
        else:
            log("main-progenitor descendants agree with the Step 0 track")
        ntr, ns = len(sc["first"]), len(snaps)
        g = fo.create_group("Sub/Tracks")
        sid = np.full((ntr, ns), -1, np.int64)
        arrays = {}
        for name, col in TRACK_ARRAYS.items():
            tail = tables[0][col].shape[1:] if col in tables[0] else ()
            arrays[name] = np.full((ntr, ns) + tail, np.nan, np.float32)
        arrays["Rhalf_star"] = np.full((ntr, ns), np.nan, np.float32)
        inside = np.zeros((ntr, ns), bool)
        fborn = np.full((ntr, ns), np.nan)
        for k, tab in enumerate(tables):
            tr = row_track[k]
            sid[tr, k] = tab["SubhaloID"]
            for name, col in TRACK_ARRAYS.items():
                if col in tab:
                    arrays[name][tr, k] = tab[col]
            arrays["Rhalf_star"][tr, k] = tab["HalfmassRadType"][:, 4]
            inside[tr, k] = tab["inside_R200"].astype(bool)
            fborn[tr, k] = tab["f_born_in_main"]
        g.create_dataset("SubhaloID", data=sid)
        for name, v in arrays.items():
            g.create_dataset(name, data=v)
        snaps_a = np.array(snaps)
        mass = arrays["Mass"].astype(np.float64)
        has = np.isfinite(mass).any(axis=1)
        kpk = np.where(has, np.nanargmax(np.where(np.isfinite(mass), mass, -np.inf), axis=1), 0)
        g.create_dataset("first_snap", data=snaps_a[sc["first"]])
        g.create_dataset("last_snap", data=snaps_a[sc["last"]])
        g.create_dataset("fate", data=sc["fate"].astype("S12"))
        g["fate"].attrs["values"] = ", ".join(FATES)
        g.create_dataset("merged_into", data=sc["merged_into"])
        g.create_dataset("M_peak", data=np.where(has, mass[np.arange(ntr), kpk], np.nan).astype(np.float32))
        g.create_dataset("snap_peak", data=np.where(has, snaps_a[kpk], -1))
        vpk = np.max(np.where(np.isfinite(arrays["Vmax"]), arrays["Vmax"], -np.inf), axis=1)
        g.create_dataset("Vmax_peak", data=np.where(np.isfinite(vpk), vpk, np.nan).astype(np.float32))
        g.create_dataset("snap_infall", data=np.where(inside.any(axis=1), snaps_a[np.argmax(inside, axis=1)], -1))
        last_fb = np.full(ntr, np.nan)
        for i in range(ntr):
            ok = np.flatnonzero(np.isfinite(fborn[i]))
            if len(ok):
                last_fb[i] = fborn[i, ok[-1]]
        g.create_dataset("f_born_in_main", data=last_fb.astype(np.float32))
        g.create_dataset("is_main", data=sc["is_main"].astype(np.int8))
        g.attrs["description"] = ("tracks follow DescSubhaloID; where several progenitors share a descendant, the main "
                                  "progenitor or else the most massive continues and the others end as merged")
        fo.attrs.update(n_tracks=ntr, main_desc_mismatches=len(mismatches))
    os.replace(tmp, path)
    log(f"wrote {path}: {ns} snapshots, {ntr} subhalo tracks")
    return 0
