"""Checks of lyratools.timeseries and scripts/lyra_timeseries.py on a synthetic 3-snapshot zoom."""
import json, os, runpy, shutil, subprocess, sys
import h5py, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "_data")
TS = os.path.join(DATA, "ts")
SCRIPT = os.path.join(HERE, "..", "scripts", "lyra_timeseries.py")
res = {}
def check(name, cond, info=""):
    res[name] = bool(cond); print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")

def cli(*args, out=None):
    env = dict(os.environ, LYRATOOLS_DATA=os.path.join(DATA, "tables"))
    out = out or os.path.join(TS, "out")
    p = subprocess.run([sys.executable, SCRIPT, "--snapdir", os.path.join(TS, "output"), "--groupdir", os.path.join(TS, "output"),
                        "--out", out, *args], capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr

if __name__ == "__main__":
    if not os.path.isdir(os.path.join(DATA, "tables")):
        runpy.run_path(os.path.join(HERE, "make_tables.py"))
    shutil.rmtree(TS, ignore_errors=True)
    runpy.run_path(os.path.join(HERE, "make_timeseries.py"))
    truth = json.load(open(os.path.join(TS, "truth.json")))
    from lyratools.timeseries import io, main as tsmain, subhalos
    from lyratools.timeseries.track import load_track

    # ---------------- reading: multi-file snapshot, binding order
    with io.Snapshot(io.snapshot_files(os.path.join(TS, "output"))[12]) as sn:
        cat = io.Catalogue(io.catalogue_files(os.path.join(TS, "output"))[12], units=sn.units)
        check("snapshot split over 2 files", len(sn.files) == 2 and sn.total(1) == int(np.asarray(cat["GroupLenType"])[:, 1].sum()))
        ok, bad = io.binding_order_ok(sn, cat)
        check("binding-order check passes on ordered subhaloes", ok, str(bad))
        check("most-bound IDs: binding order = distance ranking here",
              np.array_equal(io.most_bound_ids(sn, cat, 0, 1, 50, "binding"), io.most_bound_ids(sn, cat, 0, 1, 50, "distance")))
        met = sn.read_ranges(0, "GFM_Metals", [(0, 10)], cols=[0, 4])
        check("only the H and O columns of GFM_Metals are read", met.shape == (10, 2))

    # ---------------- full run
    rc, log = cli("--nproc", "2")
    print(log)
    check("script runs (track, 3 snapshots in parallel, combine)", rc == 0 and "wrote" in log)
    out = os.path.join(TS, "out")
    track, attrs = load_track(os.path.join(out, "main_track.h5"))
    check("track covers all snapshots", list(track["Snap"]) == [10, 11, 12] and attrs["rank_mode"] == "binding")
    check("track follows the moving main subhalo", list(track["SubhaloID"]) == [truth[s]["subs"]["main"] for s in ("10", "11", "12")]
          and np.allclose(track["Pos_ckpc"], [truth[s]["Pos_ckpc"] for s in ("10", "11", "12")], atol=1e-3))
    check("track: centre stored in float64", track["Pos_code"].dtype == np.float64)

    f = h5py.File(os.path.join(out, "timeseries_ts.h5"), "r")
    M = f["Main"]
    ap = M["Apertures_kpc"][()]
    t = [truth[s] for s in ("10", "11", "12")]
    MU = 1.989 / 1.98841          # generator's Msun (code mass 1.989e43 g = 1e10 Msun) vs lyratools.constants.MSUN
    check("header times from Header/Time", np.allclose(f["Header/t_Myr"][()], [x["t"] for x in t], rtol=1e-4))
    check("stellar mass within R200 (inert stars excluded)", np.allclose(M["Mass/Stars"][:, 8], [x["Mstars_R200"] * MU for x in t], rtol=1e-5),
          f"{M['Mass/Stars'][:, 8]} vs {[x['Mstars_R200'] for x in t]}")
    check("DM mass within R200", np.allclose(M["Mass/DM"][:, 8], [x["MDM_R200"] * MU for x in t], rtol=1e-5))
    check("gas mass within 1 kpc", np.isclose(ap[0, 4], 1.0) and np.allclose(M["Mass/Gas"][:, 4], [x["Mgas_1kpc"] * MU for x in t], rtol=1e-5))
    check("Contam_R200", np.allclose(M["Contam_R200"][()], [x["Contam_R200"] for x in t], rtol=1e-4))
    check("SFR10 of gal stars from ages", np.allclose(M["SFR/10"][()], [x["SFR10"] * MU for x in t], rtol=1e-5), str(M["SFR/10"][()]))
    check("rotation axis from the cold-gas disc", np.all(np.abs(M["RotAxis"][:, 2]) > 0.99) and np.all(M["RotAxisSource"][()] == 0))
    check("MassFormedSincePrev: NaN at the first snapshot only", np.isnan(M["MassFormedSincePrev"][0]) and np.all(M["MassFormedSincePrev"][1:] > 0))
    check("disc kinematics: cold gas V/sigma ~ 5", np.allclose(M["VoverSigma/ColdGas"][()], 5, rtol=0.1))
    check("SN counts and energies (1e51 erg)", np.all(M["SN/N_10Myr"][()] > 0) and np.all(M["SN/E_10Myr"][()] > 0)
          and M["SN/E_10Myr"].attrs["units"] == "1e51 erg")
    check("flows present for every shell and phase", all(f"Main/Flows/{d}/{q}/{s}/{p}" in f for d in ("Out", "In") for q in ("Mass", "OMass", "Energy")
          for s in ("0.04kpc", "0.2kpc", "0.5kpc", "1kpc", "0.25R200", "0.5R200", "1R200") for p in ("all", "cold", "warm", "hot")))
    floats = []
    M.visititems(lambda n, o: floats.append(o.dtype) if isinstance(o, h5py.Dataset) and o.dtype.kind == "f" else None)
    check("stored floats are float32", all(d == np.float32 for d in floats) and M["SN/N_10Myr"].dtype.kind == "i")

    # ---------------- substructure and tracks
    C = {s: f[f"Sub/Catalogue/snap_{s:03d}"] for s in (10, 11, 12)}
    ids = {s: truth[str(s)]["subs"] for s in (10, 11, 12)}
    desc = lambda s, name: int(C[s]["DescSubhaloID"][list(C[s]["SubhaloID"][()]).index(ids[s][name])])
    check("contaminated subhalo excluded", ids[10]["C"] not in C[10]["SubhaloID"][()])
    check("descendants: satellites at 10 -> 11", desc(10, "S1") == ids[11]["S1"] and desc(10, "S2") == ids[11]["S2"] and desc(10, "S3") == ids[11]["S3"])
    check("descendants: merging subhalo -> main, disrupted -> -1", desc(11, "S1") == ids[12]["main"] and desc(11, "S2") == -1
          and desc(11, "S3") == ids[12]["S3"])
    check("TrackIDs ragged and uint64", C[10]["TrackIDs"].dtype == np.uint64 and C[10]["TrackOffsets"][-1] == len(C[10]["TrackIDs"]))
    T = f["Sub/Tracks"]
    fate = {int(T["SubhaloID"][i, 0]): T["fate"][i].decode() for i in range(len(T["fate"]))}
    check("track fates", fate == {ids[10]["main"]: "survives", ids[10]["S1"]: "merged_main", ids[10]["S2"]: "lost", ids[10]["S3"]: "survives"}, str(fate))
    i1 = list(T["SubhaloID"][:, 0]).index(ids[10]["S1"])
    check("merged_into points at the main track", T["is_main"][T["merged_into"][i1]] == 1 and T["last_snap"][i1] == 11)
    check("f_born_in_main: 0 for stars born in the satellite, ~1 for the main",
          T["f_born_in_main"][i1] == 0 and T["f_born_in_main"][list(T["is_main"][()]).index(1)] > 0.9)
    check("combine: main descendants agree with the track", f.attrs["main_desc_mismatches"] == 0)
    f.close()

    # ---------------- reruns, refusal to combine
    rc, log = cli("--nproc", "2")
    check("rerun skips finished snapshots", rc == 0 and "3 snapshots already done" in log)
    bad = os.path.join(TS, "out_missing")
    shutil.copytree(out, bad)
    os.remove(os.path.join(bad, "snaps", "snap_011.h5")); os.remove(os.path.join(bad, "timeseries_ts.h5"))
    rc, log = cli("--combine-only", out=bad)
    check("refuses to combine with a missing snapshot", rc == 1 and "snap 011: missing" in log
          and not os.path.exists(os.path.join(bad, "timeseries_ts.h5")), log.strip().splitlines()[-1])
    cfgfile = os.path.join(TS, "other.json")
    json.dump({"sub": {"N_track": 50}}, open(cfgfile, "w"))
    rc, log = cli("--combine-only", "--config", cfgfile, out=bad)
    check("refuses to combine files written with another config", rc == 1 and "snap 010: config" in log)
    rc, log = cli("--dry-run")
    check("dry run prints counts for every R_sel x min_dm", rc == 0 and "R3/dm100" in log and "R5/dm300" in log
          and len([l for l in log.splitlines() if l.strip().startswith(("10 ", "11 ", "12 "))]) == 3, "\n" + log.split("Dry run")[-1])

    # ---------------- progenitor search reads only until every seed is found
    from lyratools.timeseries.track import find_progenitor
    sf, cf = io.snapshot_files(os.path.join(TS, "output")), io.catalogue_files(os.path.join(TS, "output"))
    with io.Snapshot(sf[12]) as sn12:
        cat12 = io.Catalogue(cf[12], units=sn12.units)
        seed = io.most_bound_ids(sn12, cat12, truth["12"]["subs"]["main"], 1, 1000, "binding")
        seed = np.concatenate([seed, [10 ** 12]])                          # one ID found nowhere
    with io.Snapshot(sf[11]) as sn11:
        cat11 = io.Catalogue(cf[11], units=sn11.units)
        c0, rad = np.asarray(track["Pos_code"][1], np.float64), 50.0
        full = find_progenitor(sn11, cat11, seed[:-1], c0, rad, 1.0, first_block=1 << 40)
        part = find_progenitor(sn11, cat11, seed[:-1], c0, rad, 1.0, first_block=64)
        miss = find_progenitor(sn11, cat11, seed, c0, rad, 1.0, first_block=64)
        fullm = find_progenitor(sn11, cat11, seed, c0, rad, 1.0, first_block=1 << 40)
    check("progenitor search: blockwise read = full read, with fewer IDs",
          part[:2] == full[:2] and miss[:2] == fullm[:2] and part[2] < full[2] and miss[2] == fullm[2],
          f"{part} {full} {miss} {fullm}")

    # ---------------- binding order not satisfied: fall back to distance ranking
    from lyratools.timeseries.config import DEFAULT_CONFIG
    from lyratools.timeseries.track import build_track
    alt = os.path.join(TS, "output_unordered")
    shutil.copytree(os.path.join(TS, "output"), alt)
    with h5py.File(os.path.join(alt, "groups_012", "fof_subhalo_tab_012.0.hdf5"), "a") as g:
        g["Subhalo/SubhaloIDMostbound"][...] = 10 ** 9
    msgs = []
    tr2, mode2 = build_track(io.snapshot_files(alt), io.catalogue_files(alt), DEFAULT_CONFIG, log=msgs.append)
    check("binding-order check fails -> distance ranking, same track", mode2 == "distance" and any("ranking by distance" in m for m in msgs)
          and list(tr2["SubhaloID"]) == list(track["SubhaloID"]))

    # ---------------- empty selections and float64 accumulation
    r = tsmain.compute_main({pt: {} for pt in range(5)}, 10.0, 500.0, 450.0, json.loads(json.dumps(__import__("lyratools").timeseries.DEFAULT_CONFIG)))
    check("empty selections give NaN, not errors", np.isnan(r["Rhalf/Stars3D"]) and np.isnan(r["Age/Median"]) and r["Mass/Stars"][0] == 0)
    n = 1 << 25                                              # 3.4e7 cells of 4 Msun = 1.3e8 Msun > 2^26
    m = np.full(n, 4.0, np.float32)
    rr = np.linspace(0, 1, n, dtype=np.float32)
    stalled = np.cumsum(m)[-1]                               # float32: stops growing at 2^26 ~ 6.7e7
    tot = tsmain.enclosed(rr, m, [2.0])[0]
    check("float32 cumsum stalls (test is sensitive)", stalled < 0.6 * n * 4, f"{stalled:.3g} vs {n * 4:.3g}")
    check("enclosed masses accumulate in float64", tot == n * 4.0, f"{tot:.10g}")
    check("half-mass radius accumulates in float64", abs(tsmain.half_mass_radius(rr, m) - 0.5) < 1e-3, f"{tsmain.half_mass_radius(rr, m):.4f}")
    from lyratools import profiles
    check("profiles.half_mass_radius accumulates in float64", abs(profiles.half_mass_radius(rr, m) - 0.5) < 1e-3)
    del m, rr

    print(f"\n{sum(res.values())}/{len(res)} time-series checks passed")
    if not all(res.values()):
        sys.exit(1)
