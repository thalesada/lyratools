# small synthetic LYRA zoom snapshot (two files, two FoF groups plus unbound "fuzz") with a
# group catalogue, for the Region field-access tests. Truth values are saved to zoom_truth.npz.
import os, h5py, numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data", "zoom")
for d in ("snapdir_010", "groups_010"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
rng = np.random.default_rng(7)
h, z = 0.6777, 3.0
a = 1 / (1 + z)
Mpc, Msun = 3.085678e24, 1.989e33
UL, UM, UV = Mpc, 1e10 * Msun, 1e5
L = 10.0                                  # Mpc/h
kpc_code = 1e-3 * h / a                   # one physical kpc in code length

# particle counts per type: group 0, group 1, fuzz
N = {0: (1500, 800, 700), 1: (2000, 1200, 800), 4: (1200, 600, 200)}
cen = np.array([[5.0, 5.0, 5.0], [5.2, 4.9, 5.1]])          # group centres [code]
r200_kpc = np.array([20.0, 12.0])                             # physical kpc
m200_msun = np.array([4e10, 1.2e10])


def attrs(d, ascl, hscl, l, m, v):
    for k, val in dict(a_scaling=ascl, h_scaling=hscl, length_scaling=l, mass_scaling=m, velocity_scaling=v).items():
        d.attrs[k] = float(val)
    d.attrs["to_cgs"] = UL ** l * UM ** m * UV ** v


def header(f, extra):
    g = f.create_group("Header")
    for k, v in dict(BoxSize=L, HubbleParam=h, Omega0=0.307, OmegaLambda=0.693, OmegaBaryon=0.048, Redshift=z,
                     Time=a, UnitLength_in_cm=UL, UnitMass_in_g=UM, UnitVelocity_in_cm_per_s=UV, **extra).items():
        g.attrs[k] = v
    p = f.create_group("Parameters")
    for k, v in dict(UnitLength_in_cm=UL, UnitMass_in_g=UM, UnitVelocity_in_cm_per_s=UV, ComovingIntegrationOn=1,
                     HubbleParam=h).items():
        p.attrs[k] = v
    f.create_group("Config")


def positions(pt):
    n0, n1, nf = N[pt]
    out = [cen[0] + rng.normal(0, 4.0, (n0, 3)) * kpc_code, cen[1] + rng.normal(0, 2.5, (n1, 3)) * kpc_code,
           rng.uniform(4.0, 6.0, (nf, 3))]
    return np.concatenate(out)


data = {}
for pt in (0, 1, 4):
    n = sum(N[pt])
    d = dict(Coordinates=positions(pt), Velocities=rng.normal(0, 50, (n, 3)).astype("f4"),
             ParticleIDs=np.arange(n, dtype="u8") + 10 ** 6 * pt)
    if pt != 1:
        d["Masses"] = (rng.uniform(1, 3, n) * 1e3 * h / 1e10).astype("f4")
    data[pt] = d
g = data[0]; n0 = sum(N[0])
g["Density"] = (10 ** rng.normal(5.5, 1.5, n0)).astype("f4")                      # code units, comoving (nH ~ 0.01-100)
g["InternalEnergy"] = (10 ** rng.normal(2, 1, n0)).astype("f4")                  # (km/s)^2
g["ElectronAbundance"] = rng.uniform(0, 1.2, n0).astype("f4")
gz = (10 ** rng.normal(-2.5, 0.4, n0) * 0.0134).astype("f4")
g["GFM_Metallicity"] = gz
met = np.zeros((n0, 10), "f4"); met[:, 0] = 0.75; met[:, 1] = 0.25 - gz
met[:, 4] = 0.4 * gz; met[:, 8] = 0.1 * gz; met[:, 2] = 0.2 * gz; met[:, 9] = 0.3 * gz
g["GFM_Metals"] = met
g["Temperature"] = (10 ** rng.uniform(1, 6, n0)).astype("f4")                    # stored by LYRA: unrelated to u
g["GFM_CoolingRate"] = (-10 ** rng.normal(-23, 0.5, n0)).astype("f4")
g["StarFormationRate"] = np.zeros(n0, "f4")
g["FUV_G0"] = (10 ** rng.normal(0, 1, n0)).astype("f4")
g["FUV_PEHeatingRate"] = (10 ** rng.normal(-25, 0.5, n0)).astype("f4")
g["G0TableBypass"] = (rng.random(n0) < 0.1).astype("i4")

s = data[4]; n4 = sum(N[4])
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100 * h, Om0=0.307)
ag = np.geomspace(0.02, a, 2000); tg = cosmo.age(1 / ag - 1).to("Myr").value
tform = rng.uniform(100, tg[-1] - 1, n4)
aform = np.interp(tform, tg, ag)
aform[rng.random(n4) < 0.05] *= -1                                              # wind particles (excluded)
s["StellarFormationTime"] = aform.astype("f4")
minit = rng.choice([1.0, 4.0, 10.0, 25.0], n4, p=[.4, .3, .2, .1])
s["GFM_InitialMass"] = (minit * h / 1e10).astype("f4")
s["Masses"] = (minit * 0.9 * h / 1e10).astype("f4")
life = np.where(minit >= 8, 40.0 * (minit / 8) ** -1, 1e5)                       # Myr
sn = np.where(minit < 8, -1.0, np.where(np.abs(tform) + life < tg[-1], -life / 1e3, life / 1e3))
s["SNTime"] = sn.astype("f4")
zs = (10 ** rng.normal(-2.5, 0.4, n4) * 0.0134).astype("f4")
s["GFM_Metallicity"] = zs
smet = np.zeros((n4, 10), "f4"); smet[:, 0] = 0.75; smet[:, 1] = 0.25 - zs; smet[:, 4] = 0.4 * zs; smet[:, 8] = 0.1 * zs
s["GFM_Metals"] = smet
s["BirthDensity"] = (10 ** rng.normal(7, 0.5, n4)).astype("f4")                   # code units, comoving
s["BirthPos"] = s["Coordinates"].copy()

UNITS = dict(Coordinates=(1, -1, 1, 0, 0), Velocities=(0.5, 0, 0, 0, 1), Masses=(0, -1, 0, 1, 0),
             GFM_InitialMass=(0, -1, 0, 1, 0), Density=(-3, 2, -3, 1, 0), BirthDensity=(-3, 2, -3, 1, 0),
             InternalEnergy=(0, 0, 0, 0, 2), BirthPos=(1, -1, 1, 0, 0))

npart = np.array([sum(N[0]), sum(N[1]), 0, 0, sum(N[4]), 0])
split = np.array([1100, 2500, 0, 0, 1500, 0])           # file 0 holds the first `split` particles of each type
for i, (lo, hi) in enumerate([(np.zeros(6, int), split), (split, npart)]):
    with h5py.File(os.path.join(OUT, "snapdir_010", f"snap_010.{i}.hdf5"), "w") as f:
        header(f, dict(NumPart_ThisFile=hi - lo, NumPart_Total=npart, NumPart_Total_HighWord=np.zeros(6, int),
                       MassTable=np.array([0, 5e4 * h / 1e10, 0, 0, 0, 0.]), NumFilesPerSnapshot=2,
                       Flag_DoublePrecision=0))
        for pt in (0, 1, 4):
            grp = f.create_group(f"PartType{pt}")
            for k, v in data[pt].items():
                ds_ = grp.create_dataset(k, data=v[lo[pt]:hi[pt]])
                if k in UNITS:
                    attrs(ds_, *UNITS[k])

with h5py.File(os.path.join(OUT, "groups_010", "fof_subhalo_tab_010.0.hdf5"), "w") as f:
    header(f, dict(Ngroups_ThisFile=2, Ngroups_Total=2, Nsubgroups_ThisFile=2, Nsubgroups_Total=2, NumFiles=1,
                   Nids_ThisFile=0, Nids_Total=0))
    gr = f.create_group("Group")
    lt_ = np.array([[N[0][k], N[1][k], 0, 0, N[4][k], 0] for k in range(2)], "i4")
    gr.create_dataset("GroupLenType", data=lt_)
    gr.create_dataset("GroupLen", data=lt_.sum(1).astype("i4"))
    gr.create_dataset("GroupNsubs", data=np.array([1, 1], "i4"))
    gr.create_dataset("GroupFirstSub", data=np.array([0, 1], "i4"))
    attrs(gr.create_dataset("GroupPos", data=cen), 1, -1, 1, 0, 0)
    attrs(gr.create_dataset("Group_M_Crit200", data=(m200_msun * h / 1e10).astype("f4")), 0, -1, 0, 1, 0)
    attrs(gr.create_dataset("Group_R_Crit200", data=(r200_kpc * kpc_code).astype("f4")), 1, -1, 1, 0, 0)
    attrs(gr.create_dataset("GroupMassType", data=np.ones((2, 6), "f4")), 0, -1, 0, 1, 0)
    attrs(gr.create_dataset("GroupVel", data=np.zeros((2, 3))), -1, 0, 0, 0, 1)
    sh = f.create_group("Subhalo")
    sh.create_dataset("SubhaloLenType", data=lt_)
    sh.create_dataset("SubhaloLen", data=lt_.sum(1).astype("i4"))
    sh.create_dataset("SubhaloGrNr", data=np.array([0, 1], "i4"))
    attrs(sh.create_dataset("SubhaloPos", data=cen), 1, -1, 1, 0, 0)
    attrs(sh.create_dataset("SubhaloVel", data=np.zeros((2, 3))), 0, 0, 0, 0, 1)
    attrs(sh.create_dataset("SubhaloMassType", data=np.ones((2, 6), "f4")), 0, -1, 0, 1, 0)
    for nm in ["SubhaloHalfmassRadType", "SubhaloMassInHalfRadType"]:
        sh.create_dataset(nm, data=np.full((2, 6), 1e-4, "f4"))
    for nm in ["SubhaloHalfmassRad", "SubhaloVmaxRad", "SubhaloVelDisp", "SubhaloVmax"]:
        sh.create_dataset(nm, data=np.full(2, 2e-4, "f4"))
    sh.create_dataset("SubhaloIDMostbound", data=np.array([1, 2], "u8"))

truth = dict(h=h, a=a, z=z, UL=UL, UM=UM, UV=UV, cen=cen, r200_kpc=r200_kpc, m200_msun=m200_msun,
             t_now=tg[-1], N0=np.array(N[0]), N1=np.array(N[1]), N4=np.array(N[4]))
for pt in (0, 1, 4):
    for k, v in data[pt].items():
        truth[f"{pt}/{k}"] = v
np.savez(os.path.join(OUT, "zoom_truth.npz"), **truth)
print("zoom fixture:", dict(gas=npart[0], dm=npart[1], stars=npart[4]))
