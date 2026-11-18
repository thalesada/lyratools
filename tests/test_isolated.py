"""Checks of lyratools.timeseries on a synthetic non-cosmological (isolated) galaxy without group catalogues."""
import json, os, runpy, shutil, subprocess, sys
import h5py, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "_data")
ISO = os.path.join(DATA, "iso")
SCRIPT = os.path.join(HERE, "..", "scripts", "lyra_timeseries.py")
res = {}
def check(name, cond, info=""):
    res[name] = bool(cond); print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")

def cli(*args, out=None):
    env = dict(os.environ, LYRATOOLS_DATA=os.path.join(DATA, "tables"))
    p = subprocess.run([sys.executable, SCRIPT, "--snapdir", os.path.join(ISO, "output"), "--groupdir", os.path.join(ISO, "output"),
                        "--out", out or os.path.join(ISO, "out"), *args], capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr

if __name__ == "__main__":
    if not os.path.isdir(os.path.join(DATA, "tables")):
        runpy.run_path(os.path.join(HERE, "make_tables.py"))
    shutil.rmtree(ISO, ignore_errors=True)
    runpy.run_path(os.path.join(HERE, "make_isolated.py"))
    truth = {int(k): v for k, v in json.load(open(os.path.join(ISO, "truth.json"))).items()}
    from lyratools.timeseries import io
    from lyratools.timeseries.config import DEFAULT_CONFIG, config_hash, load_config

    # ---------------- run-type detection and units
    files = io.snapshot_files(os.path.join(ISO, "output"))
    with io.Snapshot(files[1]) as sn:
        check("isolated run detected from the header (Redshift = 0, Omega0 = 0)", not sn.cosmological)
        check("isolated units: a = 1, h = 1 (header HubbleParam 0.7 ignored), physical kpc and Msun",
              sn.a == 1 and sn.h == 1 and np.isclose(sn.kpc_per_code, 1.0) and np.isclose(sn.msun_per_code, 1.989e43 / 1.98841e33))
        check("isolated clock: Time in code time units", np.isclose(sn.t_now, truth[1]["t"], rtol=1e-6) and sn.hubble_kms_kpc == 0)
    check("detection: ComovingIntegrationOn wins", io.detect_cosmological({"Time": 0.5, "Redshift": 1.0}, {"ComovingIntegrationOn": 0})[0] is False)
    check("detection: Time > 1 is not cosmological", io.detect_cosmological({"Time": 2.5, "Redshift": 0.0, "Omega0": 0.3})[0] is False)
    check("detection: z = 0 with Omega0 > 0 is cosmological", io.detect_cosmological({"Time": 1.0, "Redshift": 0.0, "Omega0": 0.3})[0] is True)
    ts_files = io.snapshot_files(os.path.join(DATA, "ts", "output"))
    if ts_files:
        with io.Snapshot(ts_files[max(ts_files)]) as sn:
            check("cosmological zoom still detected as cosmological", sn.cosmological and sn.a < 1)
    check("default config hash unchanged by the run section", config_hash(DEFAULT_CONFIG) == "f6b68244d6353363")
    check("non-default run section changes the hash", config_hash(load_config(None, run={"pt23": "ignore"})) != "f6b68244d6353363")

    # ---------------- dry run and full run without catalogues
    rc, log = cli("--dry-run")
    check("dry run without catalogues prints the track", rc == 0 and "non-cosmological run" in log and "shrinking sphere" in log
          and "no group catalogues" in log, log[-600:])
    rc, log = cli("--nproc", "2")
    print(log)
    check("full run without catalogues", rc == 0 and "no subhalo tracks" in log)
    out = os.path.join(ISO, "out")
    f = h5py.File(os.path.join(out, "timeseries_iso.h5"), "r")
    T = f["MainTrack"]
    t = [truth[s] for s in (0, 1, 2)]
    check("times from code time units", np.allclose(f["Header/t_Myr"][()], [x["t"] for x in t], rtol=1e-5)
          and np.all(f["Header/a"][()] == 1) and np.all(f["Header/z"][()] == 0))
    dc = np.linalg.norm(T["Pos_ckpc"][()] - np.array([x["centre"] for x in t]), axis=1)
    check("shrinking-sphere centre follows the moving galaxy (< 0.1 kpc)", np.all(dc < 0.1), str(dc))
    dv = np.linalg.norm(T["Vel"][()] - np.array([x["vel"] for x in t]), axis=1)
    check("velocity frame = bulk velocity (< 5 km/s)", np.all(dv < 5), str(dv))
    check("R200 from the particle profile (200 rho_crit, H0 = 70) within 3%",
          np.allclose(T["R200c"][()], [x["R200"] for x in t], rtol=0.03), f"{T['R200c'][()]} vs {[round(x['R200'], 2) for x in t]}")
    check("track rows have no SUBFIND IDs", np.all(T["SubhaloID"][()] == -1) and np.all(np.isnan(T["SeedFraction"][()])))
    M = f["Main"]
    MU = 1.989 / 1.98841          # generator's Msun (code mass 1.989e43 g = 1e10 Msun) vs lyratools.constants.MSUN
    check("SFR10 from code-time formation times", np.allclose(M["SFR/10"][()], [x["SFR10"] * MU for x in t], rtol=1e-5),
          f"{M['SFR/10'][()]} vs {[x['SFR10'] for x in t]}")
    check("Mass/Stars: stars formed in the run only", np.allclose(M["Mass/Stars"][:, 8], [x["Mstars"] * MU for x in t], rtol=1e-5))
    check("Mass/StarsIC: PartType2 + inert PartType4", np.allclose(M["Mass/StarsIC"][:, 8], [x["MIC"] * MU for x in t], rtol=1e-5))
    check("Contam_R200 not computed when PartType2/3 are stars", np.all(np.isnan(M["Contam_R200"][()])))
    check("rotation axis from the cold gas disc", np.all(np.abs(M["RotAxis"][:, 2]) > 0.99) and np.all(M["RotAxisSource"][()] == 0))
    check("disc rotation: cold gas V/sigma >> 1", np.all(M["VoverSigma/ColdGas"][()] > 5), str(M["VoverSigma/ColdGas"][()]))
    check("SN counts from absolute SNTime (code times of death) and code-time ages", np.array_equal(M["SN/N_10Myr"][()], [x["N_SN10"] for x in t])
          and M["SN/N_10Myr"][()].sum() > 0, f"{M['SN/N_10Myr'][()]} vs {[x['N_SN10'] for x in t]}")
    check("SN energies from absolute SNTime", np.all(np.asarray(M["SN/E_10Myr"][()])[np.asarray(M["SN/N_10Myr"][()]) > 0] > 0))
    from lyratools import stellar_evolution as se
    s_abs, tf = np.array([0.35, -0.12, -1.0, 0.31]), np.array([0.2, 0.1, 0.05, 0.3])        # code times; Time = 0.3
    check("SNTime convention: absolute detected, lifetime left alone",
          se.sntime_convention(s_abs, tf, 0.3) == "absolute" and se.sntime_convention([0.01, -0.01, -1.0], [0.1, 0.25, 0.2], 0.3) == "lifetime_gyr"
          and se.sntime_convention([-1.0, -1.0], [0.1, 0.2], 0.3) == "lifetime_gyr")
    myr = 3.085678e21 / 1e5 / 3.15576e13
    conv = se.sntime_as_lifetime(s_abs, tf, myr)
    check("absolute SNTime -> +-lifetime [Gyr], -1 kept", np.allclose(conv, [0.15 * myr / 1e3, -0.02 * myr / 1e3, -1.0, 0.01 * myr / 1e3])
          and np.allclose(se.death_time(tf * myr, conv)[1], 0.12 * myr) and list(se.sn_state(conv)) == list(se.sn_state(s_abs)))
    check("BirthDensity at a = 1 (physical)", np.allclose(M["BirthDensity/Young10"][:, 1][1:], 100, rtol=1e-3), str(M["BirthDensity/Young10"][:, 1]))
    check("no /Sub without catalogues", "Sub" not in f and f.attrs["n_tracks"] == 0 and f.attrs["cosmological"] == 0)
    f.close()

    # ---------------- explicit settings
    cfgfile = os.path.join(ISO, "subfind.json")
    json.dump({"run": {"center": "subfind"}}, open(cfgfile, "w"))
    rc, log = cli("--dry-run", "--config", cfgfile, out=os.path.join(ISO, "out_subfind"))
    check("run.center = subfind without catalogues is an error", rc == 2 and "no snapshot number has both" in log, log.strip()[-200:])
    cfgfile = os.path.join(ISO, "fixed.json")
    json.dump({"run": {"R200_kpc": 40.0, "pt23": "ignore"}}, open(cfgfile, "w"))
    rc, log = cli("--config", cfgfile, "--nproc", "1", out=os.path.join(ISO, "out_fixed"))
    with h5py.File(os.path.join(ISO, "out_fixed", "timeseries_iso.h5"), "r") as f:
        check("fixed R200 and pt23 = ignore", rc == 0 and np.allclose(f["MainTrack/R200c"][()], 40.0)
              and "Mass/StarsIC" not in f["Main"] and np.all(np.isnan(f["Main/Contam_R200"][()])))
    zoom_out = os.path.join(ISO, "out_zoom")
    p = subprocess.run([sys.executable, SCRIPT, "--snapdir", os.path.join(DATA, "ts", "output"), "--groupdir", ISO,
                        "--out", zoom_out, "--dry-run"], capture_output=True, text=True)
    check("cosmological run with a wrong --groupdir is still an error (no silent fallback)",
          p.returncode == 2 and 'run.center = "shrink"' in p.stdout + p.stderr, (p.stdout + p.stderr).strip()[-200:])

    print(f"\n{sum(res.values())}/{len(res)} isolated-run checks passed")
    if not all(res.values()):
        sys.exit(1)
