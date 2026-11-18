# synthetic LYRA-like zoom snapshot at z=6.5 in Mpc/h code units
import os, h5py, numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_data', 'output')
for d in ('snapdir_042', 'groups_042'): os.makedirs(os.path.join(OUT, d), exist_ok=True)
rng = np.random.default_rng(1)
h, z = 0.6777, 6.5; a = 1/(1+z)
Mpc = 3.085678e24; Msun = 1.989e33; UL, UM, UV = Mpc, 1e10*Msun, 1e5
L = 100.0          # Mpc/h box
cen = np.array([50.0, 50.0, 50.0])
kpc_code = 1e-3 * h / a      # 1 physical kpc in code length
def attrs(d, ascl, hscl, l, m, v):
    for k, val in dict(a_scaling=ascl, h_scaling=hscl, length_scaling=l, mass_scaling=m, velocity_scaling=v).items():
        d.attrs[k] = float(val)
    d.attrs["to_cgs"] = UL**l * UM**m * UV**v
def hdr(f, extra):
    g = f.create_group("Header")
    for k, v in dict(BoxSize=L, HubbleParam=h, Omega0=0.307, OmegaLambda=0.693, OmegaBaryon=0.048,
                     Redshift=z, Time=a, UnitLength_in_cm=UL, UnitMass_in_g=UM, UnitVelocity_in_cm_per_s=UV, **extra).items():
        g.attrs[k] = v
    p = f.create_group("Parameters")
    for k, v in dict(UnitLength_in_cm=UL, UnitMass_in_g=UM, UnitVelocity_in_cm_per_s=UV, ComovingIntegrationOn=1, HubbleParam=h).items():
        p.attrs[k] = v
    f.create_group("Config")
def sphere(n, scale_kpc):
    r = rng.exponential(scale_kpc, n); v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    return v * r[:, None]

# stars: bursty SFH between a=0.04 and a_now, plus two dense clumps
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.307)
ag = np.geomspace(0.02, a, 3000); tg = cosmo.age(1/ag-1).to("Myr").value
t_now = tg[-1]
Ns = 60000
tb = np.concatenate([rng.uniform(200, t_now, Ns//3)] + [rng.normal(c, 4, Ns//6) for c in (350, 520, 700, t_now-15)])
tb = tb[(tb > 150) & (tb < t_now)][:Ns]; Ns = len(tb)
aform = np.interp(tb, tg, ag)
spos = sphere(Ns, 0.3)
spos[:3000] = rng.normal(0, 0.01, (3000, 3)) + [0.4, 0.1, 0.0]     # clump 1 (old, ~350 Myr burst)
aform[:3000] = np.interp(rng.normal(350, 3, 3000), tg, ag)
spos[3000:5000] = rng.normal(0, 0.008, (2000, 3)) + [-0.3, 0.25, 0.05] # clump 2 (young)
aform[3000:5000] = np.interp(rng.normal(t_now-8, 1.5, 2000), tg, ag)
smass = rng.choice([0.5, 1, 2, 5, 20], Ns, p=[.4, .3, .15, .1, .05]).astype("f4")
tlife = np.where(smass >= 8, 10.0, 1e5)
tsn_myr = np.interp(aform, ag, tg) + tlife
# LYRA convention: -1 never a SN, +lifetime[Gyr] pending, -lifetime[Gyr] exploded
smass[rng.random(Ns) < 0.3] = 4.0              # unresolved populations
smass[rng.random(Ns) < 0.03] = 30.0
massive = smass >= 8
tlife = np.where(massive, 40.0 * (smass / 8) ** -1.0, 1e5)
tsn_myr = np.interp(aform, ag, tg) + tlife
sntime = np.where(~massive, -1.0, np.where(tsn_myr < t_now, -tlife / 1e3, tlife / 1e3))
zst = 10**(-3.5 + 2.0 * (np.interp(aform, ag, tg) - 150) / (t_now - 150) + rng.normal(0, 0.3, Ns)) * 0.0134

Ng = 80000
gpos = sphere(Ng, 1.0)
gmass = np.full(Ng, 4.0, "f4")
grho_cgs = 10**rng.normal(-25, 1.5, Ng)
u = 10**rng.normal(np.log10(1e2), 1.0, Ng)  # (km/s)^2
gz = 10**rng.normal(-2.5, 0.5, Ng) * 0.0134
gsfr = np.where(grho_cgs > 1e-22, 1e-6, 0.0)
gvel = rng.normal(0, 20, (Ng, 3)) + 30 * gpos / np.maximum(np.linalg.norm(gpos, axis=1)[:, None], 1e-3)

Nd = 50000
dpos = sphere(Nd, 3.0)

with h5py.File(os.path.join(OUT, "snapdir_042", "snap_042.0.hdf5"), "w") as f:
    npt = np.array([Ng, Nd, 0, 0, Ns, 0])
    hdr(f, dict(NumPart_ThisFile=npt, NumPart_Total=npt, NumPart_Total_HighWord=np.zeros(6, int),
                MassTable=np.array([0, 80/1e10*h, 0, 0, 0, 0.]), NumFilesPerSnapshot=1, Flag_DoublePrecision=0))
    def put(pt, pos_kpc, vel, mass=None):
        g = f.create_group(f"PartType{pt}")
        attrs(g.create_dataset("Coordinates", data=(cen + pos_kpc * kpc_code)), 1, -1, 1, 0, 0)
        attrs(g.create_dataset("Velocities", data=(vel / np.sqrt(a)).astype("f4")), 0.5, 0, 0, 0, 1)
        g.create_dataset("ParticleIDs", data=np.arange(len(pos_kpc), dtype="u8"))
        if mass is not None:
            attrs(g.create_dataset("Masses", data=(mass * h / 1e10).astype("f4")), 0, -1, 0, 1, 0)
        return g
    g0 = put(0, gpos, gvel, gmass)
    rho_code = grho_cgs / (UM / h / (UL * a / h)**3)
    attrs(g0.create_dataset("Density", data=rho_code.astype("f4")), -3, 2, -3, 1, 0)
    attrs(g0.create_dataset("InternalEnergy", data=u.astype("f4")), 0, 0, 0, 0, 2)
    g0.create_dataset("GFM_Metallicity", data=gz.astype("f4"))
    g0.create_dataset("StarFormationRate", data=gsfr.astype("f4"))
    met = np.zeros((Ng, 10), "f4"); met[:, 0] = 0.75; met[:, 1] = 0.25 - gz; met[:, 4] = 0.4 * gz; met[:, 8] = 0.1 * gz; met[:, 9] = 0.5 * gz
    g0.create_dataset("GFM_Metals", data=met)
    g0.create_dataset("ElectronAbundance", data=rng.uniform(0, 1.2, Ng).astype("f4"))
    g0.create_dataset("GFM_CoolingRate", data=(-10**rng.normal(-23, 0.5, Ng) * np.where(rng.random(Ng) < 0.1, -1, 1)).astype("f4"))
    put(1, dpos, rng.normal(0, 30, (Nd, 3)))
    g4 = put(4, spos, rng.normal(0, 10, (Ns, 3)), np.where(sntime < -0.5e-3 * 0 - 1e-12, smass * 0.3, smass) if False else np.where((sntime < 0) & (sntime != -1), smass * 0.3, smass))
    g4.create_dataset("StellarFormationTime", data=aform.astype("f4"))
    attrs(g4.create_dataset("GFM_InitialMass", data=(smass * h / 1e10).astype("f4")), 0, -1, 0, 1, 0)
    g4.create_dataset("GFM_Metallicity", data=zst.astype("f4"))
    g4.create_dataset("SNTime", data=sntime.astype("f4"))
    smet = np.zeros((Ns, 10), "f4"); smet[:, 0] = 0.75; smet[:, 1] = 0.25 - zst; smet[:, 4] = 0.4 * zst; smet[:, 8] = 0.1 * zst; smet[:, 9] = 0.5 * zst
    g4.create_dataset("GFM_Metals", data=smet)
    attrs(g4.create_dataset("BirthPos", data=(cen + spos * kpc_code)), 1, -1, 1, 0, 0)

r200_code = 12.0 * kpc_code
with h5py.File(os.path.join(OUT, "groups_042", "fof_subhalo_tab_042.0.hdf5"), "w") as f:
    hdr(f, dict(Ngroups_ThisFile=1, Ngroups_Total=1, Nsubgroups_ThisFile=1, Nsubgroups_Total=1, NumFiles=1, Nids_ThisFile=0, Nids_Total=0))
    g = f.create_group("Group")
    g.create_dataset("GroupLenType", data=np.array([[Ng, Nd, 0, 0, Ns, 0]], "i4"))
    g.create_dataset("GroupLen", data=np.array([Ng + Nd + Ns], "i4"))
    g.create_dataset("GroupNsubs", data=np.array([1], "i4")); g.create_dataset("GroupFirstSub", data=np.array([0], "i4"))
    attrs(g.create_dataset("GroupPos", data=cen[None, :]), 1, -1, 1, 0, 0)
    attrs(g.create_dataset("Group_M_Crit200", data=np.array([3e9 * h / 1e10], "f4")), 0, -1, 0, 1, 0)
    attrs(g.create_dataset("Group_R_Crit200", data=np.array([r200_code], "f4")), 1, -1, 1, 0, 0)
    attrs(g.create_dataset("GroupMassType", data=np.ones((1, 6), "f4")), 0, -1, 0, 1, 0)
    s = f.create_group("Subhalo")
    s.create_dataset("SubhaloLenType", data=np.array([[Ng, Nd, 0, 0, Ns, 0]], "i4"))
    s.create_dataset("SubhaloLen", data=np.array([Ng + Nd + Ns], "i4"))
    s.create_dataset("SubhaloGrNr", data=np.array([0], "i4"))
    attrs(s.create_dataset("SubhaloPos", data=cen[None, :]), 1, -1, 1, 0, 0)
    attrs(s.create_dataset("SubhaloMassType", data=np.array([[3e8, 2e9, 0, 0, 1.4e5, 0]], "f4") * h / 1e10), 0, -1, 0, 1, 0)
    for nm in ["SubhaloHalfmassRadType", "SubhaloMassInHalfRadType"]:
        s.create_dataset(nm, data=np.full((1, 6), 1e-4, "f4"))
    for nm in ["SubhaloHalfmassRad", "SubhaloVmaxRad", "SubhaloVelDisp", "SubhaloVmax"]:
        s.create_dataset(nm, data=np.array([2e-4], "f4"))
    s.create_dataset("SubhaloIDMostbound", data=np.array([7], "u8"))
    # catalogue velocities in AREPO's conventions (GroupVel: stored value x 1/a = peculiar km/s): known test values
    attrs(g.create_dataset("GroupVel", data=np.array([[10.0, -20.0, 5.0]]) * a), -1, 0, 0, 0, 1)
    attrs(s.create_dataset("SubhaloVel", data=np.array([[12.0, -18.0, 4.0]])), 0, 0, 0, 0, 1)   # AREPO: peculiar km/s, no a-factor
# a fake sfr.txt (AREPO format: a, total_sm, totsfrrate, rate_in_msunperyear, total_sum_mass_stars, cum_mass_stars)
aa = np.linspace(0.04, a, 500)
np.savetxt(os.path.join(OUT, "sfr.txt"), np.c_[aa, aa*0, 1e-3*(1+np.sin(aa*400)), aa*0, aa*0, aa*0])
print("made", Ns, "stars", Ng, "gas; t_now =", round(t_now, 1), "Myr")
