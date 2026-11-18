"""Checks of lyratools.yields on the synthetic tables (identity, recovery, interfaces)."""
import os
import numpy as np
import lyratools as lt
from lyratools import yields as Y

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
lt.set_data_dir(os.path.join(HERE, "tables"))
res = {}
def check(name, cond, info=""):
    res[name] = bool(cond); print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")

ms = Y.MassiveStarYields.from_file()
kar = Y.AGBYields.from_karakas()
fru = Y.AGBYields.from_fruity()
check("tables read", len(ms.m) == 40 and kar.table["N"].shape == (6, 30) and fru.table["N"].shape[0] == 11)
check("z = 1 reproduces massive table", all(np.allclose(ms.per_star(el, 1.0, 0.8), ms.sn[el] + ms.wind[el]) for el in "CNO"))

ident = Y.YieldCorrection(ms, kar, kar)
rng = np.random.default_rng(1)
a, b = rng.uniform(0.1, 1, 100), rng.uniform(0, 1, 100)
X = {el: a * ident.massive_yield(el) + b * ident.agb.imf_yield(el, Y.Z_SUN_ABS, None, "sim") for el in "CNO"}
share = b * ident.agb.imf_yield("O", Y.Z_SUN_ABS, None, "sim") / X["O"]
out = ident.correct(X, z_eff=1.0, o_agb_share=share)
check("identity", all(np.allclose(out[el], X[el]) for el in "CNO"))

corr = Y.YieldCorrection(ms, kar, fru)
z, age = 0.05, 500.0
Xs = {el: a * corr.massive_yield(el, 1.0) + b * corr.agb.imf_yield(el, z * Y.Z_SUN_ABS, age, "sim") for el in "CNO"}
expect = {el: a * corr.massive_yield(el, z) + b * corr.agb.imf_yield(el, z * Y.Z_SUN_ABS, age, "alt") for el in "CNO"}
share = b * corr.agb.imf_yield("O", z * Y.Z_SUN_ABS, age, "sim") / Xs["O"]
out = corr.correct(Xs, z_eff=z, agb_age_myr=age, o_agb_share=share)
check("recovery of known answer", all(np.allclose(out[el], expect[el]) for el in "CNO"))
check("negative-residual diagnostic", corr.correct({"C": [1e-9], "N": [1e-9], "O": [1e-2]}, z_eff=1.0)["negative_N"][0])

r_sim = corr.population_ratios(0.1, 1000, corrected=False); r_cor = corr.population_ratios(0.1, 1000, corrected=True)
check("population_ratios", np.isfinite(list(r_sim.values()) + list(r_cor.values())).all())

# LC18 BR26 massive-star option (synthetic tables: 5 masses x 4 X_O bins x 3 velocities) -- later release (lyratools/yields_lc18.py)
if Y.LC18MassiveYields is not None:
    lc = Y.LC18MassiveYields.from_files()
    tabs = [Y.LC18MassiveYields.read_table(lt.config.data_path(f"lc18_br26_v{v}")) for v in Y.LC18_VELOCITIES]
    tot = [sn + wd for _, _, sn, wd in tabs]
    mix = sum(Y.PRANTZOS18_WEIGHTS[:, k, None] * tot[k][:, :, 6] for k in range(3))
    check("LC18 tables read and mixed", lc.ejecta["N"].shape == (4, 5) and np.allclose(lc.ejecta["C"], mix)
          and np.allclose(Y.PRANTZOS18_WEIGHTS.sum(1), 1))
    z_bin = lc.x_o / Y.SOLAR_MASS_FRACTION["O"]
    check("LC18 bins exact, clamped, log-X_O interpolation",
          all(np.allclose(lc.per_star("N", z_bin[k]), lc.ejecta["N"][k]) for k in range(4))
          and np.allclose(lc.per_star("N", 1e-6), lc.ejecta["N"][0]) and np.allclose(lc.per_star("N", 10.0), lc.ejecta["N"][-1])
          and np.allclose(lc.per_star("N", np.sqrt(z_bin[1] * z_bin[2])), 0.5 * (lc.ejecta["N"][1] + lc.ejecta["N"][2])))
    c18 = Y.YieldCorrection(ms, kar, fru, high_mass_table="LC18", lc18=lc)
    check("LC18 keeps simulated (Sukhbold) yields", all(np.isclose(c18.sim_massive_yield(el), corr.massive_yield(el, 1.0)) for el in "CNO")
          and all(np.isclose(c18.massive_yield(el, z), lc.imf_yield(el, z)) for el in "CNO"))
    check("default massive_yield unchanged", Y.YieldCorrection(ms, kar, fru).high_mass_table == "sukhbold"
          and all(np.isclose(corr.massive_yield(el, 1.0), corr.sim_massive_yield(el)) for el in "CNO"))
    expect = {el: a * c18.massive_yield(el, z) + b * c18.agb.imf_yield(el, z * Y.Z_SUN_ABS, age, "alt") for el in "CNO"}
    out = c18.correct(Xs, z_eff=z, agb_age_myr=age, o_agb_share=share)
    check("LC18 recovery of known answer", all(np.allclose(out[el], expect[el]) for el in "CNO"))
    check("LC18 via from_files", np.isfinite(list(Y.YieldCorrection.from_files(high_mass_table="LC18").population_ratios(0.1, 1000).values())).all())
else:
    print("[skip] lyratools.yields_lc18 not in this release")
    try:
        Y.YieldCorrection(ms, kar, fru, high_mass_table="LC18"); check("LC18 rejected without yields_lc18", False)
    except ValueError:
        check("LC18 rejected without yields_lc18", True)
try:
    Y.YieldCorrection(ms, kar, fru, high_mass_table="nope"); check("bad high_mass_table rejected", False)
except ValueError:
    check("bad high_mass_table rejected", True)

rows = Y.plateau_diagnostic(10 ** rng.uniform(-5, -2.5, 2000), 10 ** rng.uniform(-6, -3, 2000), 10 ** rng.uniform(-5, -3, 2000))
check("plateau_diagnostic", len(rows) > 5 and "log_CO" in rows[0])
check("lifetimes", 8 < Y.lifetime_myr(1.0) / 1e3 < 12 and 30 < Y.lifetime_myr(8.0) < 50)
print(f"\n{sum(res.values())}/{len(res)} yield checks passed")
if not all(res.values()):
    raise SystemExit(1)
