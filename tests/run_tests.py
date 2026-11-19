"""Build synthetic tables, snapshot and log files, then check every lyratools module."""
import os, runpy, shutil, traceback
import numpy as np

LOG_OUT = """READIC: Type 0 (gas):           1000  (tot=           1000) masstab= 0
READIC: Type 1:                 2000  (tot=           2000) masstab= 0
READIC: Type 4:                    0  (tot=              0) masstab= 0
Injecting SN 3 1.0e-2 12345 2.0e-9 77 0.5 50.1 50.2 50.3 10.0 -5.0 2.0 0.12
Injecting mass 4 54321 4.0e-10 78 0.4 50.0 50.0 50.0 1.0 1.0 1.0 0.13
VIRGODWARF: Task=3, Injecting SN with Energy=0.01 from star ID=12346, Mstar=2e-09 into cell ID=80, rho=0.5, pos=( 50.1, 50.2, 50.3 ), vel=( 10, -5, 2 )  at time=0.14, x=-0.7, interp=1, vrot=150, fate=1, efate=1
VIRGODWARF: Task=5, Injecting SN with Energy=1e-05 from star ID=12347 into cell ID=81, rho=0.5, pos=( 50.1, 50.2, 50.3 ), vel=( 10, -5, 2 )  at time=0.15, x=-2.447, interp=0, vrot=300, fate=0, efate=1
VIRGODWARF: Task=4, Injecting mass and metals from star ID=54322, Mstar=4e-10 into cell ID=82, rho=0.4, pos=( 50, 50, 50 ), vel=( 1, 1, 1 )  at time=0.16, x=nan, interp=-1, vrot=-1, fate=-1, efate=-1
POPIII: Task 5 HaloNr 12 Mcrit200 0.0001 Zhalo 1e-4 redshift 9.5
"""
TIMEBINS = """Occupied timebins: gravity  hydro  dt  cumul-grav  cumul-sph
   X    26    1000   2000   0.01   3000
Occupied timebins: gravity  hydro  dt  cumul-grav  cumul-sph
   X    26    1100   2100   0.01   3200
"""
blk = lambda step, t, tot: (f"Step {step}, Time: {t}, CPUs: 32, MultiDomains: 8, HighestActiveTimeBin: 26\n"
                            f"total  1.00  100.0%  {tot}  100.0%\ntreegrav  0.40  40.0%  4.0  40.0%\n"
                            f"voronoi  0.20  20.0%  2.0  20.0%\nhydro  0.20  20.0%  2.0  20.0%\ndomain  0.10  10.0%  1.0  10.0%\n")

# scida creates the cache of a multi-file snapshot in a process pool: under the spawn start method
# (macOS default) the workers re-import this module, so everything runs behind the main guard.
if __name__ == "__main__":
    # The test files run through runpy as __main__ without a main guard of their own, so spawned or
    # forkserver workers (macOS default; Linux from Python 3.14) would re-run a whole test file.
    if os.name == "posix":
        import multiprocessing
        multiprocessing.set_start_method("fork", force=True)
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA = os.path.join(HERE, "_data")
    shutil.rmtree(DATA, ignore_errors=True)
    runpy.run_path(os.path.join(HERE, "make_tables.py"))
    runpy.run_path(os.path.join(HERE, "make_snapshot.py"))
    runpy.run_path(os.path.join(HERE, "make_zoom.py"))

    sim = os.path.join(DATA, "sim")
    os.makedirs(os.path.join(sim, "logs")); os.makedirs(os.path.join(sim, "output"))
    open(os.path.join(sim, "logs", "log.out.1"), "w").write(LOG_OUT)
    open(os.path.join(sim, "output", "timebins.txt"), "w").write(TIMEBINS)
    open(os.path.join(sim, "output", "cpu.txt"), "w").write(blk(0, 0.0078125, 10.0) + blk(1, 0.008, 12.0))
    failed = []
    for name in ["test_all", "test_voronoi", "test_yields", "test_darkmatter", "test_timeseries", "test_isolated",
                 "test_region", "test_fuv", "test_cooling"]:
        path = os.path.join(HERE, name + ".py")
        if not os.path.exists(path):                  # later-release tests left out of a release tree
            print(f"[skip] {name}.py not in this release")
            continue
        print(f"\n======== {name}")
        try:
            runpy.run_path(path, run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0):
                failed.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)

    # optional: reproduce FUV_G0 of a real isolated FUV snapshot (LYRATOOLS_FUV_SNAP=snapdir, LYRATOOLS_LUV=LUV_cube.txt)
    snap, luv = os.environ.get("LYRATOOLS_FUV_SNAP"), os.environ.get("LYRATOOLS_LUV")
    if not (snap and luv):
        print("[skip] real-snapshot FUV_G0 check (set LYRATOOLS_FUV_SNAP and LYRATOOLS_LUV to run it)")
    elif not os.path.exists(os.path.join(HERE, "check_fuv_snapshot.py")):
        print("[skip] real-snapshot FUV_G0 check: check_fuv_snapshot.py not in this release")
    else:
        import sys; sys.path.insert(0, HERE)
        import check_fuv_snapshot
        out = check_fuv_snapshot.main(snap, luv, 20000)
        ok = all(np.percentile(o["rel"], 99) < 0.02 and o["zero_mismatch"] == 0 for o in out.values())
        print(f"[{'ok' if ok else 'FAIL'}] FUV_G0 of the snapshot reproduced (99th percentile of |G/G_snap - 1| < 2%)")
        if not ok:
            failed.append("check_fuv_snapshot")

    print(f"\n======== {'FAILED: ' + ', '.join(failed) if failed else 'all test files passed'}")
    raise SystemExit(1 if failed else 0)
