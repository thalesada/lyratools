"""
Command-line driver: Step 0 (track, serial, cached), Step 1 (one process per snapshot, spawn),
Step 2 (combine). Entry point of scripts/lyra_timeseries.py.
"""
import argparse
import logging
import multiprocessing as mp
import os
import sys
import queue
import re
import shutil
import time
import traceback

import numpy as np

from .combine import combine
from .config import config_hash, load_config, provenance, to_json
from .io import Catalogue, Snapshot, _read_units, catalogue_files, describe_dir, snapshot_files
from .snapshot import file_status, process_snapshot, snap_path
from .subhalos import dry_run_counts
from .track import resolve_run, build_track, cross_check, load_track, save_track, track_hash

# rough bytes per loaded particle (arrays in memory plus transient copies), per type
BYTES_PER_PARTICLE = {0: 170, 1: 60, 2: 40, 3: 40, 4: 200}
SAFETY = 1.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LYRA time series of the main progenitor and its substructure (Tier 1)")
    p.add_argument("--snapdir", required=True, help="directory with snapdir_XXX/ or snap_XXX.hdf5")
    p.add_argument("--groupdir", required=True, help="directory with groups_XXX/ or fof_subhalo_tab_XXX.hdf5")
    p.add_argument("--out", required=True)
    p.add_argument("--halo", type=int, default=0, help="FoF group at the final snapshot")
    p.add_argument("--snaps", default=None, help="a:b (inclusive; either end may be omitted)")
    p.add_argument("--nproc", default="auto")
    p.add_argument("--config", default=None, help="JSON file overriding the default configuration")
    p.add_argument("--sim", default=None, help="name in timeseries_<sim>.h5 (default: from --snapdir)")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--combine-only", action="store_true")
    p.add_argument("--cleanup", action="store_true", help="delete the per-snapshot files after combining")
    return p.parse_args(argv)


def sim_name(snapdir):
    p = os.path.abspath(snapdir).rstrip("/")
    if re.fullmatch(r"snapdir_\d+", os.path.basename(p)):
        p = os.path.dirname(p)
    return os.path.basename(os.path.dirname(p)) if os.path.basename(p) == "output" else os.path.basename(p)


def parse_range(text, snaps):
    if not text:
        return list(snaps)
    a, _, b = text.partition(":")
    lo = int(a) if a else min(snaps)
    hi = int(b) if b else max(snaps)
    return [s for s in snaps if lo <= s <= hi]


# ------------------------------------------------------------------ resources
def available_memory():
    """Bytes available to this job: the smallest of the SLURM allocation, cgroup limit and free memory."""
    cands = []
    if os.environ.get("SLURM_MEM_PER_NODE"):
        cands.append(float(os.environ["SLURM_MEM_PER_NODE"]) * 2 ** 20)
    if os.environ.get("SLURM_MEM_PER_CPU") and os.environ.get("SLURM_CPUS_ON_NODE"):
        cands.append(float(os.environ["SLURM_MEM_PER_CPU"]) * float(os.environ["SLURM_CPUS_ON_NODE"]) * 2 ** 20)
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(path).read().strip()
            if v.isdigit() and int(v) < 2 ** 60:
                cands.append(float(v))
        except OSError:
            pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                cands.append(float(line.split()[1]) * 1024)
    except OSError:
        cands.append(float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")))
    return min(cands)


def available_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def estimate_memory(snap_files, cat_files, track, cfg, k, run):
    """
    Bytes needed for snapshot track[k]: particles of the FoF groups whose centres lie within the
    loading sphere (plus 50% for unbound particles) times bytes per loaded particle, plus the
    member IDs read for descendants. Without catalogues: every particle of the snapshot.
    """
    snap = int(track["Snap"][k])
    scan = (1 << 22) * 24 * 3
    with Snapshot(snap_files[snap], cosmological=run["cosmological"]) as sn:
        if run["center"] != "subfind" or snap not in cat_files:
            n = sn.counts.sum(axis=0).astype(np.float64)
            return SAFETY * sum(n[pt] * b for pt, b in BYTES_PER_PARTICLE.items()) + scan + 3e8
        cat = Catalogue(cat_files[snap], units=sn.units, cosmological=run["cosmological"])
    if cat.ngroups == 0:
        return 1e9
    from .io import periodic
    center = np.asarray(track["Pos_code"][k], np.float64)
    r_load = cfg["main"]["load_R200"] * float(track["R200c_ckpc"][k]) / cat.ckpc_per_code
    d = periodic(np.asarray(cat["GroupPos"], np.float64) - center, cat.box)
    near = np.linalg.norm(d, axis=1) < r_load + np.asarray(cat["Group_R_Crit200"], np.float64)
    n = np.asarray(cat["GroupLenType"], np.float64)[near].sum(axis=0) * 1.5
    r_sel = (cfg["sub"]["R_sel_R200"] + cfg["sub"]["margin_R200"]) * float(track["R200c_ckpc"][k]) / cat.ckpc_per_code
    near_sel = np.linalg.norm(d, axis=1) < r_sel + np.asarray(cat["Group_R_Crit200"], np.float64)
    n_ids = np.asarray(cat["GroupLenType"], np.float64)[near_sel][:, [1, 4]].sum()
    return SAFETY * (sum(n[pt] * b for pt, b in BYTES_PER_PARTICLE.items()) + 32 * n_ids) + scan + 3e8


# ------------------------------------------------------------------ parallel execution
def _child(snap, ctx, q):
    log_path = os.path.join(ctx["out"], "logs", f"snap_{snap:03d}.log")
    logger = logging.getLogger(f"lyratools.timeseries.snap{snap}")
    logger.handlers[:] = [logging.FileHandler(log_path, mode="w")]
    logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        process_snapshot(snap, ctx, log=logger.info)
        q.put((snap, True, ""))
    except Exception:
        tb = traceback.format_exc()
        logger.error(tb)
        q.put((snap, False, tb.strip().splitlines()[-1]))


def run_parallel(snaps, ctx, nproc, log=print):
    """
    One spawned process per snapshot (fresh memory each time), at most nproc at once, in the
    given order. A failing or killed process does not stop the others. Returns {snap: (ok, msg)}.
    """
    mpc = mp.get_context("spawn")
    q = mpc.Queue()
    pending, running, results = list(snaps), {}, {}

    def drain(block):
        try:
            while True:
                s, ok, msg = q.get(timeout=1.0) if block else q.get_nowait()
                results[s] = (ok, msg)
                log(f"snap {s:03d}: {'done' if ok else 'FAILED: ' + msg}")
                block = False
        except queue.Empty:
            pass

    while pending or running:
        while pending and len(running) < nproc:
            s = pending.pop(0)
            p = mpc.Process(target=_child, args=(s, ctx, q))
            p.start()
            running[s] = p
        drain(block=True)
        for s, p in list(running.items()):
            if not p.is_alive():
                p.join()
                drain(block=False)
                if s not in results:
                    results[s] = (False, f"process died (exit code {p.exitcode}; out of memory?)")
                    log(f"snap {s:03d}: FAILED: {results[s][1]}")
                del running[s]
    return results


def make_context(args, cfg, track, mode, snap_files, cat_files, run):
    return dict(out=os.path.abspath(args.out), config=cfg, config_hash=config_hash(cfg), track=track, rank_mode=mode,
                snap_files=snap_files, cat_files=cat_files, provenance=provenance(), run=run)


# ------------------------------------------------------------------ main
def get_track(args, cfg, snap_files, cat_files, write=True):
    path = os.path.join(args.out, "main_track.h5")
    if os.path.exists(path) and not args.redo:
        track, attrs = load_track(path)
        if attrs.get("track_hash") == track_hash(cfg):
            print(f"Step 0: using cached track {path} (snaps {track['Snap'][0]}-{track['Snap'][-1]})")
            return track, str(attrs["rank_mode"])
        print("Step 0: cached track was made with other track parameters; recomputing")
    print("Step 0: main-progenitor track (sequential: each snapshot needs the seed IDs of the next one)")
    t0 = time.time()
    track, mode = build_track(snap_files, cat_files, cfg)
    print(f"Step 0: {len(track['Snap'])} snapshots tracked in {time.time() - t0:.0f} s")
    if mode != "shrink":
        cross_check(track, args.groupdir)
    if write:
        save_track(path, track, cfg, mode)
        track, _ = load_track(path)     # identical arrays to those reused from the cache later
    return track, mode


def dry_run(cfg, track, cat_files, snap_files, run):
    if run["center"] != "subfind":
        print("\nDry run: no group catalogues, so no subhalo selection; main-progenitor track:")
        print("snap   t [Myr]   R200 [kpc]    M200 [Msun]")
        for k, snap in enumerate(track["Snap"]):
            print(f"{int(snap):4d} {track['t_Myr'][k]:9.1f} {track['R200c'][k]:12.2f} {track['M200c'][k]:14.4g}")
        return
    c = cfg["sub"]
    combos = [(r, m) for r in c["dry_run_R_sel"] for m in c["dry_run_min_dm"]]
    print(f"\nDry run: selected subhaloes as dark+luminous (>= {c['min_stars']} stars always included; "
          "no PartType2/3 members)")
    print("snap     z  " + "  ".join(f"{f'R{r:g}/dm{m}':>11s}" for r, m in combos) + "   (dark+luminous)")
    for k, snap in enumerate(track["Snap"]):
        cat = Catalogue(cat_files[int(snap)], units=_read_units(snap_files[int(snap)][:1]), cosmological=run["cosmological"])
        R200 = float(track["R200c_ckpc"][k]) * cat.a
        cnt = dry_run_counts(cat, np.asarray(track["Pos_code"][k], np.float64), R200, cfg, run["pt23"])
        print(f"{int(snap):4d} {cat.z:5.2f}  " + "  ".join(f"{'%d+%d' % cnt[cm]:>11s}" for cm in combos))


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)        # progress shows up in SLURM logs as it happens
    args = parse_args(argv)
    cfg = load_config(args.config, track={"halo": args.halo})
    chash = config_hash(cfg)
    sim = args.sim or sim_name(args.snapdir)
    snap_files, cat_files = snapshot_files(args.snapdir), catalogue_files(args.groupdir)
    print(f"sim {sim}: {len(snap_files)} snapshots, {len(cat_files)} group catalogues; config hash {chash}")
    if not snap_files:
        print(f"error: no snapshots found\n  --snapdir  {describe_dir(args.snapdir, snap_files)}", file=sys.stderr)
        return 2
    try:
        run = resolve_run(cfg, snap_files, cat_files)
    except FileNotFoundError as e:
        print(f"error: {e}\n"
              f"  --snapdir  {describe_dir(args.snapdir, snap_files)}\n"
              f"  --groupdir {describe_dir(args.groupdir, cat_files)}\n"
              "  expected snapdir_XXX/*.hdf5 and groups_XXX/*.hdf5 directly inside these directories", file=sys.stderr)
        return 2
    print(run["description"])

    if args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        track, _ = get_track(args, cfg, snap_files, cat_files)       # cached: the full run reuses it
        dry_run(cfg, track, cat_files, snap_files, run)
        return 0

    for d in ("", "snaps", "logs"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        f.write(to_json(cfg) + "\n")
    track, mode = get_track(args, cfg, snap_files, cat_files)
    ctx = make_context(args, cfg, track, mode, snap_files, cat_files, run)

    if not args.combine_only:
        todo = parse_range(args.snaps, [int(s) for s in track["Snap"]])
        if not args.redo:
            done = [s for s in todo if file_status(snap_path(args.out, s), chash) == "ok"]
            if done:
                print(f"Step 1: {len(done)} snapshots already done with this config (use --redo to recompute)")
            todo = [s for s in todo if s not in done]
        todo = sorted(todo, reverse=True)              # latest (largest) first
        if todo:
            if args.nproc == "auto":
                k = int(np.flatnonzero(track["Snap"] == todo[0])[0])
                need = estimate_memory(snap_files, cat_files, track, cfg, k, run)
                avail = available_memory()
                nproc = max(1, min(available_cores(), int(0.7 * avail // need), len(todo)))
                print(f"Step 1: ~{need / 1e9:.1f} GB per snapshot, {avail / 1e9:.1f} GB available -> {nproc} processes")
            else:
                nproc = max(1, int(args.nproc))
            print(f"Step 1: {len(todo)} snapshots, {nproc} processes; logs in {os.path.join(args.out, 'logs')}")
            t0 = time.time()
            res = run_parallel(todo, ctx, nproc)
            failed = sorted(s for s, (ok, _) in res.items() if not ok)
            print(f"Step 1: {len(todo) - len(failed)} done, {len(failed)} failed in {time.time() - t0:.0f} s"
                  + (f": {failed}" if failed else ""))
        if args.snaps:
            print("Step 2 skipped (--snaps given); run --combine-only once every snapshot is done")
            return 0

    print("Step 2: combine")
    rc = combine(args.out, sim, track, cfg, chash, ctx["provenance"])
    if rc == 0 and args.cleanup:
        shutil.rmtree(os.path.join(args.out, "snaps"))
        print("removed the per-snapshot files")
    return rc
