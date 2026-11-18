"""Region field access on a small two-file zoom snapshot (h5py reader): lazy raw/derived fields, values(),
unit conversion against hand-computed values from the header's a and h, metadata, selections, isolated runs."""
import json, os, runpy
import numpy as np
import lyratools as lt
from lyratools.constants import BOLTZMANN, GAMMA_MINUS1, PROTONMASS, KPC, MSUN
from lyratools.yields import ATOMIC_MASS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "_data")
ZOOM = os.path.join(DATA, "zoom")
if not os.path.exists(os.path.join(ZOOM, "zoom_truth.npz")):
    runpy.run_path(os.path.join(HERE, "make_zoom.py"))
lt.set_data_dir(os.path.join(DATA, "tables"))
T = {k: (v.astype(np.float64) if v.dtype == np.float32 else v) for k, v in np.load(os.path.join(ZOOM, "zoom_truth.npz")).items()}
h, a = float(T["h"]), float(T["a"])
UL, UM, UV = float(T["UL"]), float(T["UM"]), float(T["UV"])
MU = UM / 1e10 / MSUN               # generator's Msun (1.989e33 g) over lyratools.constants.MSUN
N0, N1, N4 = T["N0"], T["N1"], T["N4"]          # (group 0, group 1, fuzz) for gas, dm, stars
SNAP, GROUPS = os.path.join(ZOOM, "snapdir_010"), os.path.join(ZOOM, "groups_010")


def main():
    results = {}
    def check(name, cond, info=""):
        results[name] = bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")

    # group 1: second group, so offsets are not zero; its gas and dm slices cross the file boundary
    halo = lt.load_halo(SNAP, halo_id=1, fields={"gas": ["Density"], "stars": ["GFM_InitialMass", "StellarFormationTime"], "dm": []},
                        dtype=np.float64)
    g0, g1 = N0[0], N0[0] + N0[1]
    s0, s1 = N4[0], N4[0] + N4[1]

    # ---------------- metadata
    from astropy.cosmology import FlatLambdaCDM
    t_age = FlatLambdaCDM(H0=100 * h, Om0=0.307).age(1 / a - 1).to("Myr").value
    check("catalogue found next to the snapshot", halo.cat is not None and os.path.basename(os.path.dirname(halo.cat.files[0])) == "groups_010")
    check("region metadata z, a", np.isclose(halo.z, 3.0) and np.isclose(halo.a, 0.25) and halo.cosmological)
    check("region metadata t_now [Myr]", isinstance(halo.t_now, float) and abs(halo.t_now - t_age) < 0.5, f"{halo.t_now:.1f} Myr")
    check("region metadata r200, m200 physical", np.isclose(halo.r200, T["r200_kpc"][1], rtol=1e-5) and np.isclose(halo.R200, halo.r200)
          and np.isclose(halo.m200, T["m200_msun"][1] * MU, rtol=1e-6), f"r200 {halo.r200:.3f} kpc, m200 {halo.m200:.4g} Msun")
    check("region centre used", np.allclose(halo.center_code, T["cen"][1]) and halo.halo_id == 1
          and np.allclose(halo.center_kpc, T["cen"][1] * UL / KPC * a / h, rtol=1e-6))

    # ---------------- attribute access
    check("attribute access standard arrays", len(halo.gas.pos) == N0[1] and halo.gas.pos.shape == (N0[1], 3)
          and halo.dm.mass.shape == (N1[1],) and all(hasattr(halo.stars, k) for k in ("tform", "age", "m_init", "pos", "vel", "r", "vr")))
    valid = T["4/StellarFormationTime"][s0:s1] > 0
    check("stars: wind particles dropped", len(halo.stars) == valid.sum()
          and np.allclose(halo.stars.m_init, T["4/GFM_InitialMass"][s0:s1][valid] / h * 1e10 * MU, rtol=1e-6))

    # ---------------- unit conversion by hand
    pos_hand = (T["0/Coordinates"][g0:g1] - T["cen"][1]) * UL / KPC * a / h
    check("pos [kpc] = code x UL a / h", np.allclose(halo.gas.pos, pos_hand, rtol=1e-6, atol=1e-6))
    check("vel [km/s] = code x sqrt(a)", np.allclose(halo.gas.vel, T["0/Velocities"][g0:g1] * np.sqrt(a), rtol=1e-6))
    rho_hand = T["0/Density"][g0:g1] * UM / UL ** 3 * h ** 2 / a ** 3
    check("values('Density') g/cm^3 (preloaded, physical)", np.allclose(halo.gas.values("Density"), rho_hand, rtol=1e-6, atol=0),
          f"median {np.median(rho_hand):.3g}")
    check("values('Masses', 'g') = code x UM / h", np.allclose(halo.gas.values("Masses", "g"), T["0/Masses"][g0:g1] * UM / h, rtol=1e-6, atol=0))
    check("values('mass') Msun / values('mass', 'g')", np.allclose(halo.gas.values("mass", "g"), halo.gas.mass * MSUN, rtol=1e-6)
          and np.allclose(halo.gas.values("mass"), halo.gas.mass))
    check("values('InternalEnergy') erg/g (lazy)", np.allclose(halo.gas.values("InternalEnergy"), T["0/InternalEnergy"][g0:g1] * 1e10, rtol=1e-6))
    check("values('pos', 'pc')", np.allclose(halo.gas.values("pos", "pc"), 1e3 * halo.gas.pos))
    check("p['Density'] is the stored code value", np.allclose(halo.gas["Density"], T["0/Density"][g0:g1], rtol=1e-6))

    # ---------------- lazy raw fields (not preloaded)
    check("lazy raw gas field ElectronAbundance", "ElectronAbundance" not in halo.gas
          and np.allclose(halo.gas["ElectronAbundance"], T["0/ElectronAbundance"][g0:g1]) and "ElectronAbundance" in halo.gas)
    check("lazy raw star field SNTime (attribute access, winds dropped)", np.allclose(halo.stars.SNTime, T["4/SNTime"][s0:s1][valid]))
    bd = halo.stars.values("BirthDensity")
    check("lazy raw BirthDensity, comoving -> physical", np.allclose(bd, T["4/BirthDensity"][s0:s1][valid] * UM / UL ** 3 * h ** 2 / a ** 3, rtol=1e-6, atol=0))
    check("stored Temperature used (LYRA)", np.allclose(halo.gas.values("Temperature"), T["0/Temperature"][g0:g1], rtol=1e-6))

    # ---------------- lazy derived fields
    nH = halo.gas.values("nH")
    X = T["0/GFM_Metals"][g0:g1, 0]
    check("lazy derived nH [cm^-3]", np.allclose(nH, X * rho_hand / PROTONMASS, rtol=1e-5, atol=0), f"median {np.median(nH):.3g}")
    xe = T["0/ElectronAbundance"][g0:g1]
    T_derived = lt.chemistry.temperature(T["0/InternalEnergy"][g0:g1] * 1e10, xe, X)
    check("derived temperature formula mu = 4/(1+3X+4X xe)", np.allclose(T_derived, GAMMA_MINUS1 * T["0/InternalEnergy"][g0:g1] * 1e10 / BOLTZMANN
          * 4 / (1 + 3 * X + 4 * X * xe) * PROTONMASS, rtol=1e-6))
    oh = halo.gas.values("12logOH")
    mt = T["0/GFM_Metals"][g0:g1]
    check("12logOH by number", np.allclose(oh, 12 + np.log10(mt[:, 4] / ATOMIC_MASS["O"] / (mt[:, 0] / ATOMIC_MASS["H"])), rtol=1e-5),
          f"median {np.median(oh):.2f}")
    sol = np.genfromtxt(lt.config.data_path("solar"), skip_header=1, unpack=True)[1]
    r_no = halo.gas.values("N_O")
    check("[X/O] relative to solar", np.all(np.isfinite(r_no) | (mt[:, 3] == 0)))
    feh = halo.stars.values("Fe_H")
    check("[Fe/H] stars", np.allclose(feh, np.log10(T["4/GFM_Metals"][s0:s1][valid][:, 8] / T["4/GFM_Metals"][s0:s1][valid][:, 0] / (sol[25] / sol[0])), rtol=1e-5))
    res = lt.chemistry.twelve_log_oh_reservoir(mt, halo.gas.mass)
    from lyratools.timeseries.main import oh_from_sums
    check("reservoir 12logOH from summed masses (time-series definition)", np.isclose(res, oh_from_sums(mt[:, 4], mt[:, 0], halo.gas.mass)),
          f"{res:.3f}")
    check("derived star fields Age, ESN, SNState", np.allclose(halo.stars["Age"], halo.stars.age)
          and np.array_equal(halo.stars["SNState"], lt.stellar_evolution.sn_state(halo.stars.SNTime))
          and np.allclose(halo.stars.values("ESN"), lt.stellar_evolution.sn_energies(halo.stars.m_init, halo.stars.SNTime)))
    check("values() returns numpy", type(halo.gas.values("Density")) is np.ndarray and type(halo.gas.values("nH")) is np.ndarray
          and type(halo.gas.values("G0TableBypass")) is np.ndarray)

    # ---------------- FUV-run field registry
    pe = halo.gas.values("FUV_PEHeatingRate")
    check("registry: FUV_PEHeatingRate in erg cm^3/s", np.allclose(pe, T["0/FUV_PEHeatingRate"][g0:g1])
          and np.allclose(halo.gas.values("FUV_PEHeatingRate", "J*m**3/s"), pe * 1e-13))
    check("registry: FUV_G0 dimensionless (Habing)", np.allclose(halo.gas.values("FUV_G0"), T["0/FUV_G0"][g0:g1]))
    nt = halo.gas.values("net_true")
    check("net_true = GFM_CoolingRate + FUV_PEHeatingRate (old scheme)", np.allclose(nt, T["0/GFM_CoolingRate"][g0:g1] + T["0/FUV_PEHeatingRate"][g0:g1], rtol=1e-6))
    from lyratools.fields import net_true_rate
    lam, pr, by = np.array([-2.0, -2.0]), np.array([1.0, 1.0]), np.array([0, 1])
    check("net_true with the G0 table: PE already in GFM_CoolingRate, except bypassed cells",
          np.allclose(net_true_rate(lam, pr, True), lam) and np.allclose(net_true_rate(lam, pr, True, by), [-2.0, -1.0]))
    check("G0TableDebugBypass reads the older G0TableBypass", np.array_equal(halo.gas["G0TableDebugBypass"], T["0/G0TableBypass"][g0:g1]))

    # ---------------- errors
    try:
        halo.gas["NoSuchField"]; ok = False; msg = ""
    except KeyError as e:
        msg = str(e); ok = "Temperature" in msg and "nH" in msg and "BirthDensity" not in msg and "raw" in msg
    check("KeyError lists raw and derived names", ok, msg[:90] + "...")
    try:
        halo.stars.NoSuchField; ok = False
    except AttributeError:
        ok = True
    check("missing attribute raises AttributeError", ok)
    check("available() lists raw + derived", {"FUV_G0", "nH", "pos", "12logOH"} <= set(halo.gas.available()))

    # ---------------- selections
    inner = halo.within(5.0)
    sel = halo.gas.r < 5.0
    check("within(r) masks every type", len(inner.gas) == sel.sum() and len(inner.dm) == (halo.dm.r < 5).sum() and len(inner.stars) == (halo.stars.r < 5).sum()
          and inner.z == halo.z and inner.r200 == halo.r200)
    check("lazy field after a selection", np.allclose(inner.gas["StarFormationRate"], T["0/StarFormationRate"][g0:g1][sel])
          and np.allclose(inner.gas.values("FUV_G0"), T["0/FUV_G0"][g0:g1][sel]))
    check("lazy star field after within_r200", np.allclose(halo.within_r200(0.5).stars.values("BirthDensity"), bd[halo.stars.r < 0.5 * halo.r200]))
    rev = halo.gas.select(np.arange(len(halo.gas))[::-1])
    check("lazy field for an unsorted selection", np.allclose(rev["GFM_Metallicity"], T["0/GFM_Metallicity"][g0:g1][::-1]))
    inner.recenter(center_kpc=[1.0, 0, 0])
    check("views are independent of the parent", not np.allclose(halo.gas.pos[sel][:3], inner.gas.pos[:3]))

    # ---------------- box selection: lazy fields read by index
    hb = lt.load_halo(SNAP, halo_id=0, selection="box", size_r200=1.0, fields={"gas": [], "stars": ["StellarFormationTime"]}, groupdir=GROUPS)
    ids = np.asarray(hb.gas["ParticleIDs"]).astype(np.int64)
    check("box selection: lazy field matches by ParticleIDs", len(ids) > 0 and np.allclose(hb.gas.values("FUV_G0"), T["0/FUV_G0"][ids])
          and np.all(np.abs(hb.gas.pos) < 0.5 * hb.r200 * (1 + 1e-6)), f"{len(ids)} cells incl. fuzz")
    sid = np.asarray(hb.stars["ParticleIDs"]).astype(np.int64) - 4 * 10 ** 6
    check("box selection: lazy star field after dropping winds", np.allclose(hb.stars["SNTime"], T["4/SNTime"][sid])
          and np.all(T["4/StellarFormationTime"][sid] > 0))

    # ---------------- default fields, dtype, and the explicit list
    hd = lt.load_halo(SNAP, halo_id=0)
    check("default fields preloaded (GFM_Metals read on demand)", all(k in hd.gas for k in ("Density", "InternalEnergy"))
          and "GFM_Metals" not in hd.gas and "SNTime" in hd.stars and "t_death" in hd.stars)
    check("default dtype float32", hd.gas.pos.dtype == np.float32 and hd.gas.mass.dtype == np.float32 and hd.stars.r.dtype == np.float32)
    check("explicit list loads only what is asked", "InternalEnergy" not in lt.load_halo(SNAP, 1, fields={"gas": ["Density"]}).gas
          and len(lt.load_halo(SNAP, 1, fields={"gas": ["Density"]}).stars) == 0)
    pl = lt.load_halo(SNAP, 1, fields={"gas": []}).gas.load("Density", "GFM_Metals")
    check("Particles.load reads several fields at once", np.allclose(pl["Density"], T["0/Density"][g0:g1]))
    check("rest_frame='particles' removes the mass-weighted velocity",
          np.allclose(sum((p.vel.astype(float) * p.mass[:, None]).sum(0) for p in lt.load_halo(SNAP, 1, rest_frame="particles", dtype=np.float64).parts.values()), 0, atol=1e-3 * halo.gas.mass.sum()))

    # ---------------- live stars
    st = halo.stars
    sn = np.asarray(st["SNTime"])
    alive = st.alive(m_min=4)
    from lyratools.yields import lifetime_myr
    zst = np.asarray(st["GFM_Metallicity"])
    expect = (st.m_init >= 4) & np.where(sn == -1, st.age < lifetime_myr(st.m_init, zst), sn > 0)
    check("stars.alive() from SNTime", np.array_equal(alive, expect) and alive.sum() > 0, f"{alive.sum()} of {len(st)}")

    # ---------------- a scida dataset is accepted (only its paths are used)
    try:
        from scida import load
        ds = load(SNAP, catalog=GROUPS, units=True)
        hs = lt.load_halo(ds, halo_id=1, dtype=np.float64)
        check("load_halo(scida dataset) reads the same particles", np.allclose(hs.gas.pos, halo.gas.pos) and np.isclose(hs.r200, halo.r200))
    except ImportError:
        print("[skip] scida not installed")

    # ---------------- isolated runs: load_isolated (header units)
    iso = os.path.join(DATA, "iso", "output")
    if not os.path.exists(os.path.join(DATA, "iso", "truth.json")):
        runpy.run_path(os.path.join(HERE, "make_isolated.py"))
    tr = json.load(open(os.path.join(DATA, "iso", "truth.json")))["2"]
    hi = lt.load_isolated(os.path.join(iso, "snapdir_002", "snap_002.0.hdf5"))
    check("load_isolated: a = 1, z = 0, h = 1 (header HubbleParam ignored), t_now from code time",
          hi.a == 1 and hi.z == 0 and hi.h == 1 and np.isclose(hi.t_now, tr["t"], rtol=1e-9) and not hi.cosmological, f"t_now {hi.t_now:.2f} Myr")
    check("load_isolated: centred on the median position of the inert stars", np.allclose(hi.center_code, tr["centre"], atol=0.1),
          f"offset {np.linalg.norm(hi.center_code - tr['centre']):.3f} kpc")
    MUi = 1.989 / 1.98841
    check("load_isolated: inert stars excluded from stars", len(hi.inert) == 500 and np.all(hi.stars.m_init > 0) and np.all(np.isfinite(hi.stars.tform))
          and np.isclose(hi.stars.mass.sum(dtype=float), tr["Mstars"] * MUi, rtol=5e-4))
    check("load_isolated: velocities [km/s], formation times [Myr]", np.allclose(np.median(hi.gas.vel, 0), tr["vel"], atol=3)
          and hi.stars.tform.max() <= hi.t_now and np.allclose(hi.stars.age, hi.t_now - hi.stars.tform))
    check("load_isolated: values() from header units, derived nH and T", np.allclose(hi.gas.values("nH"), 50.0, rtol=5e-4)
          and np.allclose(hi.gas.values("Temperature"), 5e3, rtol=1e-5) and np.allclose(hi.gas.values("Density", "Msun/pc**3"),
          hi.gas.values("Density") / 1.98841e33 * lt.constants.PC ** 3, rtol=1e-6))
    check("load_isolated: directory path works too", len(lt.load_isolated(os.path.join(iso, "snapdir_002")).gas) == len(hi.gas))
    check("DEFAULT_FIELDS untouched", "GFM_InitialMass" in lt.region.DEFAULT_FIELDS["PartType4"]
          and lt.region.DEFAULT_FIELDS["PartType0"][0] == "Velocities")
    from lyratools.region import _sntime_convention
    check("SNTime convention detection", _sntime_convention([1.3, -0.5, -1], np.array([1.0, 0.4, 0.2]), 1.2) == "absolute"
          and _sntime_convention([0.01, -0.01, -1], np.array([0.1, 0.2, 0.2]), 0.3) == "lifetime_gyr")

    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} region checks passed")
    if n_ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
