"""Checks of lyratools.darkmatter against analytic haloes."""
import numpy as np
from lyratools import darkmatter as dm

res = {}
def check(name, cond, info=""):
    res[name] = bool(cond); print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")

rng = np.random.default_rng(3)
def sample(rho_func, r_max, n):
    rg = np.geomspace(1e-4, r_max, 5000)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (4 * np.pi * rg[1:] ** 2 * rho_func(rg[1:]) + 4 * np.pi * rg[:-1] ** 2 * rho_func(rg[:-1])) * np.diff(rg))])
    r = np.interp(rng.uniform(0, cdf[-1], n), cdf, rg)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    return r[:, None] * v, cdf[-1]

rho_s, r_s = 1e7, 2.0                      # Msun/kpc^3, kpc
N = 400000
pos, Mtot = sample(lambda r: dm.nfw(r, rho_s, r_s), 20.0, N)
m = np.full(N, Mtot / N)
r = np.linalg.norm(pos, axis=1)
bins = np.geomspace(0.05, 20, 40)
p = dm.radial_profile(r, m, bins)
f = dm.fit_profile(p["r"], p["rho"], "nfw", 0.1, 20, n=p["n"])
check("NFW fit recovers parameters", abs(f["params"]["r_s"] / r_s - 1) < 0.05 and abs(f["params"]["rho_s"] / rho_s - 1) < 0.1,
      f"r_s = {f['params']['r_s']:.3f} (2.0), rho_s = {f['params']['rho_s']:.3g} (1e7)")
g = dm.fit_profile(p["r"], p["rho"], "gnfw", 0.1, 20, n=p["n"])
check("gNFW inner slope of a cusp ~ 1", abs(g["params"]["gamma"] - 1) < 0.1, f"gamma = {g['params']['gamma']:.2f}")
sl, err = dm.inner_slope(p["r"], p["rho"], 0.1, 0.4, n=p["n"])
check("inner_slope of NFW between 0.1-0.4 kpc", abs(sl - np.mean(dm.local_slope(np.geomspace(0.1, 0.4, 20), dm.nfw(np.geomspace(0.1, 0.4, 20), rho_s, r_s)))) < 0.08, f"{sl:.2f} +- {err:.2f}")

r_c = 0.5
pos_c, Mc = sample(lambda r: dm.core_nfw(r, rho_s, r_s, r_c, 1.0), 20.0, N)
rc_ = np.linalg.norm(pos_c, axis=1)
pc = dm.radial_profile(rc_, np.full(N, Mc / N), bins)
fc = dm.fit_profile(pc["r"], pc["rho"], "core_nfw", 0.05, 20, n=pc["n"])
check("coreNFW fit recovers core radius", abs(fc["params"]["r_c"] / r_c - 1) < 0.2 and fc["params"]["n"] > 0.8,
      f"r_c = {fc['params']['r_c']:.3f} (0.5), n = {fc['params']['n']:.2f} (1)")
sl_c, _ = dm.inner_slope(pc["r"], pc["rho"], 0.05, 0.25, n=pc["n"])
check("inner slope of a core inside r_c is shallow", sl_c > -0.4, f"{sl_c:.2f} (NFW there: {np.mean(dm.local_slope(np.geomspace(0.05, 0.25, 20), dm.nfw(np.geomspace(0.05, 0.25, 20), rho_s, r_s))):.2f})")
check("core_nfw with n = 0 is NFW", np.allclose(dm.core_nfw(np.geomspace(0.01, 10, 50), rho_s, r_s, r_c, 0.0), dm.nfw(np.geomspace(0.01, 10, 50), rho_s, r_s)))
E = dm.core_formation_energy(lambda rr: dm.nfw(rr, rho_s, r_s), lambda rr: dm.core_nfw(rr, rho_s, r_s, r_c, 1.0), r_max=20 * r_c)
check("core formation energy positive", E > 0, f"{E:.2e} erg (~{E / 1e51:.0f} SNe of 1e51 erg)")

rho_crit = 1.27e2 * 400                      # Msun/kpc^3 (z ~ 6.5)
rp = dm.power_radius(r, m, rho_crit)
check("power_radius sensible", 0.001 < rp < 1.0 and np.isfinite(rp), f"{rp * 1e3:.1f} pc for {N} particles")
rp_low = dm.power_radius(r[:N // 20], m[:N // 20] * 20, rho_crit)
check("power_radius grows with fewer particles", rp_low > rp, f"{rp_low * 1e3:.1f} pc with N/20")

rg = np.geomspace(1e-3, 50, 2000)
phi = dm.spherical_potential(r, m, rg)
vc = np.sqrt(dm.G_KPC * np.interp([0.5, 2.0, 5.0], rg, np.interp(rg, np.sort(r), np.cumsum(m[np.argsort(r)]), left=0)) / np.array([0.5, 2.0, 5.0]))
cpos = np.array([[0.5, 0, 0], [0, 2.0, 0], [0, 0, 5.0]]); cvel = np.array([[0, vc[0], 0], [0, 0, vc[1]], [vc[2], 0, 0]])
apo = dm.apocentres(cpos, cvel, rg, phi)
check("apocentres of circular orbits = their radius", np.allclose(apo, [0.5, 2.0, 5.0], rtol=0.02), str(np.round(apo, 3)))
apo_r = dm.apocentres(np.array([[1.0, 0, 0]]), np.array([[0, 0, 0]]), rg, phi)
check("radial orbit at rest: apocentre = position", np.isclose(apo_r[0], 1.0, rtol=0.02))

vel = rng.normal(0, 30, (N, 3))
vp = dm.velocity_profiles(pos, vel, bins)
check("isotropic velocities: beta ~ 0", np.nanmedian(np.abs(vp["beta"][5:30])) < 0.05 and np.nanmax(np.abs(vp["beta"][5:30])) < 0.25, f"median |beta| = {np.nanmedian(np.abs(vp['beta'][5:30])):.3f}")
pos_r = pos.copy(); vel_r = (pos_r / np.linalg.norm(pos_r, axis=1)[:, None]) * rng.normal(0, 30, (N, 1))   # purely radial velocities
check("radial velocities: beta ~ 1", np.nanmedian(dm.velocity_profiles(pos_r, vel_r, bins)["beta"][5:30]) > 0.95)
ids_a = rng.permutation(1000); ids_b = rng.permutation(np.arange(500, 1500))
ia, ib = dm.match_ids(ids_a, ids_b)
check("match_ids", len(ia) == 500 and np.array_equal(ids_a[ia], ids_b[ib]))
a, b, gam = dm.dicintio14_shape(10 ** np.array([-4.0, -2.7, -1.5]))
check("Di Cintio 2014: flattest near M*/Mh ~ 0.2-0.5%", gam[1] < gam[0] and gam[1] < gam[2], f"gamma = {np.round(gam, 2)}")
# memory-light variants agree with the direct ones
rg2 = np.geomspace(1e-3, 50, 3000)
phi_h = dm.potential_on_grid(rg2, [(r[: N // 2], m[0]), (r[N // 2:], m[N // 2:])])
phi_d = dm.spherical_potential(r, m, rg2)
sel = (rg2 > 0.05) & (rg2 < 15)
check("potential_on_grid matches spherical_potential", np.max(np.abs(phi_h[sel] - phi_d[sel]) / np.abs(phi_d[sel])) < 0.01,
      f"max rel. difference {np.max(np.abs(phi_h[sel] - phi_d[sel]) / np.abs(phi_d[sel])):.1e}")
apo_h = dm.apocentres(cpos, cvel, rg2, phi_h)
check("apocentres in the histogram potential", np.allclose(apo_h, [0.5, 2.0, 5.0], rtol=0.03), str(np.round(apo_h, 3)))
big = rng.permutation(np.arange(1, 2_000_001, dtype=np.uint64)); want = rng.choice(big, 5000, replace=False)
ia2, iw2 = dm.find_ids(big, want, chunk=300_000)
check("find_ids", len(ia2) == 5000 and np.array_equal(big[ia2], want[iw2]))
check("power_radius with scalar mass = with array", np.isclose(dm.power_radius(r, m[0], rho_crit), dm.power_radius(r, m, rho_crit), rtol=0.02))
print(f"\n{sum(res.values())}/{len(res)} dark matter checks passed")
if not all(res.values()):
    raise SystemExit(1)
