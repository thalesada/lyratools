import os, warnings, numpy as np, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from scida import load
    HAVE_SCIDA = True
except ImportError:
    HAVE_SCIDA = False
import lyratools as lt
from lyratools import (sfh, flows, profiles, chemistry, stellar_evolution as se, models, photometry,
                       substructure, evolution, logs, observations, storage, plotting, cosmology)

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
FAKE = os.path.join(HERE, "output")
SIM = os.path.join(HERE, "sim")
lt.set_data_dir(os.path.join(HERE, "tables"))
results = {}
def check(name, cond, info=""):
    results[name] = bool(cond)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")

SNAP = f"{FAKE}/snapdir_042/snap_042.0.hdf5"
GROUPS = f"{FAKE}/groups_042"
halo = lt.load_halo(SNAP, halo_id=0, groupdir=GROUPS, dtype=np.float64)     # all particles are in group 0 here
g, s = halo.gas, halo.stars

# ---------------- derived fields (numpy, lyratools.fields)
nH = g.values("nH");                 check("field nH", np.all(nH > 0) and g.unit_of("nH") == "cm**-3", f"median {np.median(nH):.3g}")
T = g.values("Temperature");         check("field Temperature", np.all(np.isfinite(T)) and g.unit_of("Temperature") == "K", f"median {np.median(T):.3g} K")
P = g.values("Pressure_k");          check("field Pressure_k", np.all(P > 0))
tc = g.values("CoolingTime");        check("field CoolingTime", np.all(tc >= 0) and (tc == 0).any())
ss = g.values("SelfShieldingFactor"); check("field SelfShieldingFactor", np.all((ss >= 0) & (ss <= 1)))
feh = g.values("Fe_H");              check("field Fe_H gas", np.all(np.isfinite(feh)), f"median {np.median(feh):.2f}")
zz = s.values("Z_Zsun");             check("field Z_Zsun stars", np.all(zz > 0))
tf, age = s.values("FormationTime"), s.values("Age")
check("field FormationTime/Age", np.allclose(tf + age, lt.TimeConversion.from_params(h=0.6777).t_at_z(6.5), atol=0.5))
state = s["SNState"]; dt_ = s.values("DeathTime")
check("field SNState/DeathTime", set(np.unique(state)) <= {0, 1, 2} and np.all(np.isfinite(dt_[state == 2])) and np.all(np.isnan(dt_[state != 2])))
esn = s.values("ESN")
check("field ESN", np.all(esn[state != 2] == 0) and np.all(esn[state == 2] > 0), f"{(esn > 0).sum()} SNe, total {esn.sum():.3g} erg")

# ---------------- region
halo_box = lt.load_halo(SNAP, halo_id=0, selection="box", dtype=np.float64)
check("load_halo fof vs box", len(halo.stars) == len(halo_box.stars) and np.allclose(np.sort(halo.stars.tform), np.sort(halo_box.stars.tform)),
      "(all particles are in group 0 in this test snapshot)")
check("load_halo", len(halo.gas) > 0 and len(halo.stars) > 0 and abs(halo.R200 - 12) < 0.1, f"R200={halo.R200:.2f} kpc, centre offset {np.linalg.norm(halo.center):.3f} kpc")
# load_halo options: catalogue centre, explicit centre, catalogue and particle rest frames
check("load_halo default: catalogue centre, velocities as stored", np.allclose(halo.center, 0) and np.allclose(halo.bulk_velocity, 0))
h_sub = lt.load_halo(SNAP, halo_id=0, center="subhalo", dtype=np.float64)
check("load_halo center='subhalo'", np.allclose(h_sub.dm.pos, halo.dm.pos))
_off = np.array([0.001, 0.0, 0.0])                     # code units
_grp_pos = np.asarray(halo.cat["GroupPos"][0], float)
h_off = lt.load_halo(SNAP, halo_id=0, center=_grp_pos + _off, dtype=np.float64)
check("load_halo explicit centre", np.allclose(h_off.dm.pos, halo.dm.pos - _off * halo.kpc_per_code, atol=1e-6))
h_gv = lt.load_halo(SNAP, halo_id=0, rest_frame="group", dtype=np.float64)
check("load_halo rest_frame='group' (GroupVel)", np.allclose(halo.dm.vel - h_gv.dm.vel, [10.0, -20.0, 5.0], atol=1e-4),
      f"shift {np.round((halo.dm.vel - h_gv.dm.vel)[0], 3)}")
h_sv = lt.load_halo(SNAP, halo_id=0, rest_frame="subhalo", dtype=np.float64)
check("load_halo rest_frame='subhalo' (SubhaloVel)", np.allclose(halo.dm.vel - h_sv.dm.vel, [12.0, -18.0, 4.0], atol=1e-4))
h_pv = lt.load_halo(SNAP, halo_id=0, rest_frame="particles", dtype=np.float64)
_m = np.concatenate([p["mass"] for p in h_pv.parts.values()]); _v = np.concatenate([p["vel"] for p in h_pv.parts.values()])
check("load_halo rest_frame='particles'", np.allclose((_v * _m[:, None]).sum(0) / _m.sum(), 0, atol=1e-6))
# formation-time field: LYRA name in the snapshot; both names can be requested
_f_gfm = {"PartType4": ["Masses", "GFM_InitialMass", "GFM_StellarFormationTime", "SNTime"]}
_f_lyra = {"PartType4": ["Masses", "GFM_InitialMass", "StellarFormationTime", "SNTime"]}
h_gfm, h_lyra = lt.load_halo(SNAP, halo_id=0, fields=_f_gfm), lt.load_halo(SNAP, halo_id=0, fields=_f_lyra)
check("formation time from LYRA's StellarFormationTime (requested by either name)",
      all(k in h_gfm.stars for k in ("tform", "m_init", "t_death")) and np.allclose(h_gfm.stars.tform, h_lyra.stars.tform)
      and np.allclose(h_gfm.stars.tform, halo.stars.tform))
# masses: always read
h_nom = lt.load_halo(SNAP, halo_id=0, fields={"PartType0": ["Velocities"], "PartType4": ["Velocities"]}, dtype=np.float64)
check("Masses read even if not requested", np.allclose(h_nom.gas.mass, halo.gas.mass) and np.allclose(h_nom.stars.mass, halo.stars.mass))
h32 = lt.load_halo(SNAP, halo_id=0)
check("load_halo default dtype float32", h32.dm.pos.dtype == np.float32 and np.allclose(h32.dm.pos, halo.dm.pos, atol=1e-4))
st, ga = halo.stars, halo.gas
check("region stellar times", np.allclose(st.tform + st.age, halo.t_now) and "t_death" in st)
check("region units", np.isclose(np.median(ga.mass), 4.0, rtol=1e-3) and ga.pos.dtype.kind == "f")
halo.face_on(1.0)
check("face_on rotation keeps radii", np.isfinite(halo.stars.r).all())

# ---------------- sfh
t, sfr = sfh.sfh(st.tform, st.m_init, dt=10)
check("sfh mass conservation", np.isclose(np.sum(sfr * 10e6), st.m_init.sum(), rtol=1e-6))
tg, avg = sfh.sfr_trailing(st.tform, st.m_init, windows=(10, 100), t_end=halo.t_now)
bursts = sfh.find_bursts(tg, avg[10], avg[100], t_min=tg[0] + 100)
check("find_bursts", len(bursts) >= 3, f"{len(bursts)} bursts, peaks at {[round(b['t_peak']) for b in bursts]}")
check("burstiness_stats", np.isfinite(sfh.burstiness_stats(tg, avg[10], avg[100])["sigma_log_sfr10"]))
sg, ssfr = sfh.compute_ks(ga.pos, ga.mass, st.pos, st.m_init, st.age, radius=1.0, timebin=10)
check("compute_ks", sg > 0 and ssfr >= 0, f"Sigma_gas={sg:.3g}, Sigma_SFR={ssfr:.3g}")
sgg, sfg = sfh.compute_ks_grid(ga.pos, ga.mass, st.pos, st.m_init, st.age, box=2, ncell=8)
check("compute_ks_grid", sgg.shape == (64,))
rho_cgs = ga.get_as("Density", "g/cm**3")
check("sfr_from_density", np.all(sfh.sfr_from_density(ga.mass, rho_cgs) > 0))
img, ext = sfh.phase_diagram(g.values("nH"), g.values("Temperature"), weights=np.full(len(g), 4.0))
check("phase_diagram", img.shape == (99, 99) and np.isfinite(img).any())

# ---------------- flows
u = ga.get_as("InternalEnergy", "cm**2/s**2"); Zg = ga.get_as("GFM_Metallicity")
fr = flows.spherical_flow_rate(ga.pos, ga.vel, ga.mass, radius=1.0, thickness=0.2, Z=Zg, u=u)
check("spherical_flow_rate", fr["out"]["mass"] > 0 and fr["out"]["energy"] > 0, f"out {fr['out']['mass']:.3g}, in {fr['in']['mass']:.3g} Msun/yr")
dfr = flows.disk_flow_rate(ga.pos, ga.vel, ga.mass, height=0.5, thickness=0.1, Z=Zg, u=u)
check("disk_flow_rate", dfr["out"]["mass"] >= 0)
gas_d = dict(pos=ga.pos, vel=ga.vel, mass=ga.mass, Z=Zg, u=u)
stars_d = dict(tform=st.tform, m_init=st.m_init, mass=st.mass, Z=st.get_as("GFM_Metallicity"), sntime=st.get_as("SNTime"))
lf = flows.loading_factors(gas_d, stars_d, halo.t_now, radius=1.0, dt=100)
check("loading_factors", all(np.isfinite(list(lf.values()))), str({k: round(v, 3) for k, v in lf.items()}))
lfi = flows.loading_factors_isolated(gas_d, stars_d, halo.t_now, height=0.5, thickness=0.1, dt=50)
check("loading_factors_isolated", all(np.isfinite(list(lfi.values()))))

# ---------------- stellar evolution
check("sn_energy", se.sn_energy(5.0) == 0 and se.sn_energy(20.0) > 0)
check("sn returns", se.sn_mass_return([10, 20, 120]).shape == (3,) and np.all(se.sn_metal_return([10, 50]) > 0))
mret, zret = se.agb_returns([2.0, 5.0], [1e-3, 1e-3])
_k = np.loadtxt(os.path.join(HERE, "tables", "StellarEvolution_Karakas.txt"), skiprows=1)
_exp = [_k[(np.isclose(_k[:, 0], 1e-3)) & (np.isclose(_k[:, 1], m))][0, 123:].sum() for m in (2.0, 5.0)]
check("agb_returns (RegularGridInterpolator)", np.allclose(mret, _exp), f"{mret} vs table {np.round(_exp, 4)}")
check("died_within", se.died_within(st.tform, stars_d["sntime"], st.m_init, halo.t_now, 100).sum() > 0)

# ---------------- chemistry
met = ga.get_as("GFM_Metals")
check("abundance_ratio", np.all(np.isfinite(chemistry.abundance_ratio(met, "O", "Fe"))))
check("ion_fraction", np.allclose(chemistry.ion_fraction([4, 5], [-1, -1], [-2, -2], redshift=6.5), -1))
check("self_shielding_factor", 0 < chemistry.self_shielding_factor(1e-2, 6.5) < 1)
lam = ga.get_as("GFM_CoolingRate")
check("cooling_time", np.all(chemistry.cooling_time(u, rho_cgs, lam) >= 0))
check("cooling/dynamical time within", np.isfinite(chemistry.cooling_time_within(ga.mass[ga.r < 1], u[ga.r < 1], rho_cgs[ga.r < 1], lam[ga.r < 1]))
      and chemistry.dynamical_time_within(halo.all_types()["r"], halo.all_types()["mass"], 1.0) > 0)

# ---------------- models
check("profiles models", np.isfinite(models.nfw(1.0, 1e7, 2.0)) and np.isfinite(models.einasto(1.0, 0.17, 1e7, 2.0)) and np.isfinite(models.sersic(1, 1, 1, 1)))
lo, hi = models.moster_abundance_matching(1e11, 1.0, scatter=True)
check("abundance matching", np.isfinite([lo, hi]).all() and models.moster_abundance_matching(1e11, 1.0) > 0 and models.guo_abundance_matching(1e11) > 0
      and models.behroozi_abundance_matching(10.0) > 0)
trh = models.relaxation_time(1e5, 3.0)
check("relaxation_time", 0.05 < trh < 5, f"{trh:.3f} Gyr for 1e5 Msun, 3 pc")
w, delta = models.evaporation_factor(1e5, 10.0, 3.0, 5e4)
check("evaporation_factor", 0 < w <= 1)
check("rho_crit", np.isclose(models.rho_crit(0, halo.tconv.cosmology), 1.8784e-29 * 0.6777 ** 2, rtol=1e-3))
check("jeans_temperature/mag_at_sb", models.jeans_temperature(1.0, 1e3) > 0 and np.isfinite(models.mag_at_sb(100, 25)))

# ---------------- profiles
rc, dens = profiles.density_profile(ga.pos, ga.mass, np.linspace(0, 3, 31))
check("density_profile", np.isfinite(dens).sum() > 20)
rc, vc = profiles.rotation_curve(halo.all_types()["pos"], halo.all_types()["mass"], np.linspace(0.05, 5, 30))
check("rotation_curve", np.all(np.diff(vc[rc > 1]) < 5) and vc.max() > 0, f"vmax {vc.max():.1f} km/s")
rh = profiles.half_mass_radius(st.r, st.mass)
check("half_mass_radius", 0 < rh < 2, f"{rh:.3f} kpc")
check("ellipticity", 0 <= profiles.ellipticity(st.mass, st.pos) < 1)

# ---------------- photometry
mags = photometry.magnitudes(st.get_as("GFM_Metallicity"), st.age * 1e6, st.m_init)
V = mags["Vmag"]
mags_clip = photometry.magnitudes(st.get_as("GFM_Metallicity"), st.age * 1e6, st.m_init, clip_to_table=True)
lyra_like = st.m_init >= 4      # in LYRA, stars below 4 Msun only exist as 4 Msun unresolved particles
check("magnitudes", set(mags) == {"Umag", "Bmag", "Vmag"} and np.isclose(st.m_init, 4, rtol=1e-3).any())
check("magnitudes clip_to_table", (mags_clip["Vmag"][lyra_like] < 35).mean() > 0.99,
      f"in tables: {(V[lyra_like] < 35).mean():.3f} default -> {(mags_clip['Vmag'][lyra_like] < 35).mean():.3f} clipped")
Lv = photometry.luminosity(V)
check("luminosity/total_magnitude", np.nansum(Lv) > 0 and np.isfinite(photometry.total_magnitude(V)), f"M_V,tot = {photometry.total_magnitude(V):.2f}")
zp, ref = photometry.jwst_zeropoint("F200W")
check("jwst_zeropoint", np.isclose(zp, 25.91) and np.isclose(ref, 759.6), f"{zp}, {ref}")
rcs, sb = profiles.surface_brightness_profile(st.pos * 1e3, Lv, np.linspace(0, 1000, 21))
check("surface_brightness_profile", np.isfinite(sb).sum() > 10)
check("half_light_radius", 0 < profiles.half_light_radius(st.r, Lv) < 2)
if HAVE_SCIDA:     # optional scida helpers
    ds = load(SNAP, catalog=GROUPS, units=True)
    lt.add_fields(ds)
    check("add_fields (scida recipes) match the numpy fields", np.allclose(np.sort(ds.data["PartType0"]["nH"].compute().magnitude), np.sort(nH), rtol=1e-3))   # scida uses its own Msun
    bands = photometry.add_magnitude_fields(ds)
    Vf = ds.data["PartType4"]["Vmag"][:2000].compute()
    check("add_magnitude_fields (lazy)", len(bands) == 3 and np.isfinite(Vf).all())

# ---------------- substructure
mask = substructure.luminous_subhalos(halo, fof=0, min_n_particles=1, min_half_rad_kpc=0, min_stellar_half_rad_kpc=0)
check("luminous_subhalos", mask.dtype == bool and len(mask) == 1)
sid = np.asarray(s["SubhaloID"])
ages_sub = substructure.per_subhalo(sid, age, 1, stat="mean")
mass_sub = substructure.per_subhalo(sid, s.m_init, 1)
check("per_subhalo", np.isclose(ages_sub[0], age.mean()) and mass_sub[0] > 0)
bp = substructure.birth_positions_ckpc(halo, st["BirthPos"])
track_t = np.array([0, halo.t_now + 10]); track_pos = np.tile(substructure.birth_positions_ckpc(halo, halo.cat["GroupPos"][0])[None, :], (2, 1))
ex = substructure.exsitu_stars(bp, st.tform, track_t, track_pos, np.full(2, 5.0 * 7.5))
check("exsitu_stars", ex.dtype == bool and ex.mean() < 0.2, f"ex-situ fraction {ex.mean():.3f}")

# ---------------- evolution, cosmology
evo = evolution.SubfindEvolution(FAKE)
check("SubfindEvolution", np.isclose(evo.Mvir[0], 3e9, rtol=1e-3) and np.isclose(evo.Rvir[0], 12, rtol=1e-2) and np.isfinite(evo.time[0]),
      f"Mvir={evo.Mvir[0]:.3g}, Rvir={evo.Rvir[0]:.3g} kpc, Mstar={evo.Mstar[0]:.3g}")
tconv = halo.tconv
check("TimeConversion", np.isclose(tconv.t_at_z(6.5), halo.t_now) and np.isclose(tconv.z_at_t(halo.t_now), 6.5, rtol=1e-4))
tab = cosmology.snapshot_table(FAKE)
check("snapshot_table/snap_at_z", cosmology.snap_at_z(tab, 6.4) == (42, 6.5))

# ---------------- logs, observations, storage, plotting
sn = logs.SNLogData(SIM)
check("SNLogData", len(sn.ESN) == 5 and np.isclose(sn.Mstar[0], 2.0e-9 * 1.989e43 / 1.98841e33), f"ESN {sn.ESN}")
check("SNLogData SNII_ZDEP columns", np.allclose(sn.x, [np.nan, np.nan, -0.7, -2.447, np.nan], equal_nan=True)
      and np.array_equal(sn.interp, [-1, -1, 1, 0, -1]) and np.array_equal(sn.vrot, [-1, -1, 150, 300, -1])
      and np.array_equal(sn.fate, [-1, -1, 1, 0, -1]) and np.array_equal(sn.efate, [-1, -1, 1, 1, -1])
      and np.array_equal(sn.fate_mismatch, [False, False, False, True, False])
      and np.array_equal(sn.StarID, [12345, 54321, 12346, 12347, 54322])
      and np.allclose(sn.timeSN[2:], [0.14, 0.15, 0.16]) and sn.Mstar[3] == 0,
      f"x {sn.x}, interp {sn.interp}, vrot {sn.vrot}, fate {sn.fate}, efate {sn.efate}, ID {sn.StarID}")
import tempfile
snsim = tempfile.mkdtemp()
os.makedirs(os.path.join(snsim, "output"))
open(os.path.join(snsim, "output", "supernovae.txt"), "w").write(
    "# header\n"
    "0.12 4000000000000123 77 3 1.05e+51 20 18.1 0.0021 0.00057 0.5 1.2 2e-09 50.1 50.2 50.3 10 -5 2 -1 1 150 1 1\n"
    "0.13 54321 78 4 0 2 0.5 0.0021 nan 0.4 0.9 4e-10 50 50 50 1 1 1 nan -1 -1 -1 -1\n"
    "0.12 4000000000000123 77 3 1.05e+51 20 18.1 0.0021 0.00057 0.5 1.2 2e-09 50.1 50.2 50.3 10 -5 2 -1 1 150 1 1\n"
    "0.14 12347 79 5 1e+49 40 30 0.0001 2e-05 0.3 0.7 3e-09 50 50 50 1 1 1 -2.447 0 300 0 1\n")
snf = logs.SNLogData(snsim)
check("SNLogData from supernovae.txt", len(snf.ESN) == 3 and snf.StarID[1] == 4000000000000123 and snf.StarID.dtype == np.int64
      and np.allclose(snf.ESN, [0, 1.05, 0.01]) and np.allclose(snf.Minit, [2, 20, 40]) and np.allclose(snf.timeSN, [0.13, 0.12, 0.14])
      and np.array_equal(snf.fate_mismatch, [False, False, True]) and np.allclose(snf.nH_arepo, [0.9, 1.2, 0.7])
      and np.allclose(snf.rhoSN, sn.rhoSN[0] / 0.5 * np.array([0.4, 0.5, 0.3]) * (0.12 / np.array([0.13, 0.12, 0.14])) ** 3),
      f"IDs {snf.StarID}, ESN {snf.ESN}, t {snf.timeSN}")
p3 = logs.PopIIILogData(SIM)
check("PopIIILogData", p3.nhalos == 1)
check("nparticles", np.array_equal(logs.nparticles_ic(SIM)[:2], [1000, 2000])
      and np.array_equal(logs.nparticles_timebins(SIM), [3000, 3200]))
cpu = logs.CPUData(os.path.join(SIM, "output", "cpu.txt"))
check("CPUData", cpu.n_cores == 32 and np.allclose(cpu.seconds_running, [10, 12]) and list(cpu.highest_timebin) == [26, 26])
mc = observations.McConnachie2012()
check("McConnachie2012", np.isnan(mc.FeH[0]) and np.isfinite(mc.dist_kpc[1]))
check("KS data", len(observations.ks_sigma_figgs()[0]) == 6)
path = os.path.join(HERE, "store", "test_042.hdf5")
if os.path.exists(path): os.remove(path)
storage.add_to_storage(np.arange(3), "x", path); storage.add_to_storage(np.arange(3) + 1, "x", path)
with h5py.File(path, "a") as f:
    storage.write_to_file([1.0, 2.0], ["a", "b"], f); storage.write_to_file([3.0, 4.0], ["a", "b"], f)
    check("storage", f["x"].shape == (2, 3) and f["a"].shape == (1, 2))
lt.use_style("ttt"); lt.use_style("lyra")
fig, ax = plt.subplots(); ax.plot([200, 840], [0, 1]); ax.set_xlim(200, 840)
ax2 = plotting.add_zaxis(ax, tconv, zticks=(15, 10, 8, 7), time_unit="Myr")
img, ext = plotting.particle_image(st.pos, st.mass, boxsize=2.0, res=100, sigma=1)
plotting.add_scale_bar(ax, 100, "100 Myr", color="k")
check("plotting", len(ax2.get_xticks()) >= 3 and img.shape == (100, 100))

# ---------------- yields: correction applied to a particle set
from lyratools import yields as Yd
yc = Yd.YieldCorrection.from_files()
yo = Yd.correct_particles(halo.gas, yc, agb_age_myr=500)
check("yields.correct_particles", all(k in yo for k in ("log_NO", "log_CO", "12logOH", "log_NO_sim")) and np.isfinite(yo["log_NO"]).all(),
      f"median log N/O {np.median(yo['log_NO_sim']):.2f} -> {np.median(yo['log_NO']):.2f}")

# ---------------- data-file registry: every entry has a reference and a source; errors say where to get the file
check("DATA_FILES have reference and source", all(d.reference and d.source and str(d) == d.path for d in lt.config.DATA_FILES.values()))
try:
    lt.set_data_dir(os.path.join(HERE, "no_such_dir")); lt.config.data_path("mcconnachie"); msg = ""
except FileNotFoundError as e:
    msg = str(e)
lt.set_data_dir(os.path.join(HERE, "tables"))
check("missing-file error names reference and download", "McConnachie 2012" in msg and "fetch_data.py" in msg, msg.splitlines()[0] if msg else "")
import subprocess, sys
fd = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "..", "scripts", "fetch_data.py"),
                     os.path.join(HERE, "fetch"), "--dry-run"], capture_output=True, text=True)
check("scripts/fetch_data.py --dry-run", fd.returncode == 0 and "[get]  mcconnachie" in fd.stdout and "sukhbold" in fd.stdout
      and not os.path.exists(os.path.join(HERE, "fetch")), fd.stderr[-300:])

print(f"\n{sum(results.values())}/{len(results)} checks passed")
if not all(results.values()):
    raise SystemExit(1)
