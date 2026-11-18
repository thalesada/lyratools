# synthetic 3-snapshot LYRA-like zoom (Mpc/h code units) for lyratools.timeseries:
# a moving main halo, a subhalo that merges (own FoF -> satellite -> merged), a satellite that is
# disrupted (-> group fuzz), a dark satellite that survives, a group contaminated by PartType2,
# unbound gas, inert stars (GFM_InitialMass = 0). Snapshots are split over 2 files.
import json, os
import h5py, numpy as np
from astropy.cosmology import FlatLambdaCDM

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data", "ts", "output")
h, L = 0.6777, 100.0
Mpc, Msun, mp, kB = 3.085678e24, 1.989e33, 1.67262192e-24, 1.380649e-16
UL, UM, UV = Mpc, 1e10 * Msun, 1e5
SNAPS = {10: 8.0, 11: 7.5, 12: 7.0}
R200 = {10: 14.0, 11: 15.0, 12: 16.0}       # physical kpc of the main group
cosmo = FlatLambdaCDM(H0=100 * h, Om0=0.307, Ob0=0.048)
A = {s: 1 / (1 + z) for s, z in SNAPS.items()}
T = {s: cosmo.age(z).to("Myr").value for s, z in SNAPS.items()}
ag = np.geomspace(0.02, 0.2, 4000); tg = cosmo.age(1 / ag - 1).to("Myr").value
a_of_t = lambda t: np.interp(t, tg, ag)
kpc_code = lambda a: 1e-3 * h / a           # code length per physical kpc
rng = np.random.default_rng(7)

def sphere(n, scale):
    r = rng.exponential(scale, n); v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    return v * r[:, None]

# main-halo centre (comoving code) and peculiar bulk velocity; satellites' offsets [physical kpc]
CEN = {s: np.array([50.0, 50.0, 50.0]) + (s - 10) * np.array([0.003, 0.001, 0.0]) for s in SNAPS}
VMAIN = np.array([60.0, -10.0, 5.0])
OFF = {"S1": {10: (40, 0, 0), 11: (20, 0, 0)}, "S2": {10: (0, 25, 0), 11: (0, 22, 0), 12: (0, 18, 0)},
       "S3": {10: (0, 0, -30), 11: (0, 0, -28), 12: (0, 0, -26)}, "C": {10: (0, -35, 0), 11: (0, -35, 0), 12: (0, -35, 0)},
       "main": {s: (0, 0, 0) for s in SNAPS}}
VREL = {"S1": (-80, 0, 0), "S2": (0, -60, 0), "S3": (0, 0, 40), "C": (0, 0, 0), "main": (0, 0, 0)}
centre = lambda obj, s: CEN[s] + np.array(OFF[obj][s], float) * kpc_code(A[s])

def host_centre(obj, a):
    """Comoving centre of obj at scale factor a (linear in a between snapshots, clamped)."""
    ss = sorted(OFF[obj]); aa = np.array([A[s] for s in ss]); cc = np.array([centre(obj, s) for s in ss])
    return np.column_stack([np.interp(a, aa, cc[:, k]) for k in range(3)])

# ------------------------------------------------------------------ particle pools (fixed per object)
next_id = [1]
def ids(n):
    i = np.arange(next_id[0], next_id[0] + n, dtype=np.uint64); next_id[0] += n; return i

def dm(n, scale, mass=80.0):
    return dict(ids=ids(n), rel=sphere(n, scale), vrel=rng.normal(0, 30, (n, 3)), mass=np.full(n, mass))

def stars(n, scale, t_lo, t_hi, obj, inert=0):
    rel = sphere(n, scale); tf = rng.uniform(t_lo, t_hi, n)
    m = np.where(rng.random(n) < 0.1, 20.0, 4.0); m[:inert] = 0.0
    z = 10 ** rng.normal(-2.5, 0.3, n) * 0.0134
    a_f = a_of_t(tf)
    bpos = host_centre(obj, a_f) + rel * kpc_code(a_f)[:, None]
    nH_birth = 10 ** rng.normal(2, 0.3, n)
    return dict(ids=ids(n), rel=rel, vrel=rng.normal(0, 20, (n, 3)), m_init=m, tform=tf, XH=np.full(n, 0.75), XO=0.4 * z,
                bpos=bpos, bd_phys=nH_birth * mp / 0.75)

def gas(n, scale, cold):
    rel = sphere(n, scale); vrel = rng.normal(0, 10 if cold else 60, (n, 3))
    if cold:   # disc in the x-y plane rotating about +z
        rel[:, 2] *= 0.1
        R = np.hypot(rel[:, 0], rel[:, 1]); vrel[:, 0] += -50 * rel[:, 1] / R; vrel[:, 1] += 50 * rel[:, 0] / R
        nH = 10 ** rng.uniform(1, 3, n); Tk = 10 ** rng.uniform(2, 3.8, n); xe = np.full(n, 0.01)
    else:
        vrel += 40 * rel / np.linalg.norm(rel, axis=1)[:, None]
        nH = 10 ** rng.uniform(-3, 0, n); Tk = 10 ** rng.uniform(4.2, 6, n); xe = np.full(n, 1.15)
    X = 0.75; mu = 4 / (1 + 3 * X + 4 * X * xe)
    u = Tk * kB / (2 / 3) / (mu * mp)          # cm^2/s^2
    z = 10 ** rng.normal(-2.0 if cold else -2.8, 0.2, n) * 0.0134
    return dict(ids=ids(n), rel=rel, vrel=vrel, mass=np.full(n, 4.0), rho=nH * mp / X, u=u, xe=xe, nha=rng.uniform(0, 1, n),
                XH=np.full(n, X), XO=0.4 * z, cool=-10 ** rng.normal(-23, 0.3, n))

t10 = T[10]
P = {
    "main": {0: [gas(8000, 1.0, True), gas(12000, 4.0, False)], 1: [dm(20000, 3.0)],
             4: [stars(6000, 0.5, 200, t10 - 20, "main", inert=100), stars(500, 0.4, t10 - 30, t10, "main"),
                 stars(1000, 0.4, t10, T[11], "main"), stars(1000, 0.4, T[11], T[12], "main")]},
    "S1": {1: [dm(3000, 0.6)], 4: [stars(300, 0.2, 200, t10 - 50, "S1"), stars(100, 0.2, t10, T[11], "S1")]},
    "S2": {1: [dm(1000, 0.5)], 4: [stars(100, 0.2, 200, t10 - 50, "S2")]},
    "S3": {1: [dm(500, 0.4)]},
    "C": {1: [dm(200, 0.5)], 2: [dict(ids=ids(50), rel=sphere(50, 0.5), vrel=rng.normal(0, 30, (50, 3)), mass=np.full(50, 1000.0))]},
    "fuzz2": {2: [dict(ids=ids(10), rel=sphere(10, 2.0), vrel=rng.normal(0, 30, (10, 3)), mass=np.full(10, 1000.0))]},
    "igm": {0: [gas(2000, 25.0, False)]},
}
for p in P["igm"][0]:
    p["rel"] = rng.uniform(-60, 60, (2000, 3))
DISPERSED = rng.normal(0, 8.0, (1100, 3))     # S2 at snapshot 12

def pool(obj, pt, s):
    """Concatenated particles of obj/type at snapshot s (stars formed by then), rel to obj centre."""
    parts = P[obj].get(pt, [])
    if not parts:
        return None
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if pt == 4:
        keep = out["tform"] <= T[s]
        out = {k: v[keep] for k, v in out.items()}
    return out

def SNtime(p, s):
    age = T[s] - p["tform"]; massive = p["m_init"] >= 8
    return np.where(~massive, -1.0, np.where(age > 10.0, -0.010, 0.010))

# ------------------------------------------------------------------ structure per snapshot
def layout(s):
    """[(group: [(subhalo name, members: [(obj, rel offset obj->subhalo centre)])], fuzz members)], unbound."""
    if s == 10:
        groups = [([("main", ["main"]), ("S2", ["S2"]), ("S3", ["S3"])], ["fuzz2"]), ([("S1", ["S1"])], []), ([("C", ["C"])], [])]
    elif s == 11:
        groups = [([("main", ["main"]), ("S1", ["S1"]), ("S2", ["S2"]), ("S3", ["S3"])], ["fuzz2"]), ([("C", ["C"])], [])]
    else:
        groups = [([("main", ["main", "S1"]), ("S3", ["S3"])], ["fuzz2", "S2"]), ([("C", ["C"])], [])]
    return groups, ["igm"]

def place(obj, pt, s, host):
    """Particles of obj at snapshot s: comoving code positions, peculiar velocities, rel [kpc] to host centre."""
    p = pool(obj, pt, s)
    if p is None:
        return None
    a = A[s]
    if obj == "S1" and s == 12:                    # merged: inside the main halo
        rel_host = p["rel"] * 1.5
    elif obj == "S2" and s == 12:                  # disrupted: spread around its last position
        rel_host = np.array(OFF["S2"][12]) + DISPERSED[: len(p["ids"])]
    elif obj in ("fuzz2",):
        rel_host = p["rel"]
    elif obj == "igm":
        rel_host = p["rel"]
    else:
        rel_host = p["rel"] + np.array(OFF[obj][s]) - np.array(OFF[host][s] if host in OFF else (0, 0, 0))
    hostc = centre(host, s) if host in OFF else CEN[s]
    q = dict(p)
    q["pos"] = hostc + rel_host * kpc_code(a)
    q["vel"] = VMAIN + np.array(VREL.get(obj, (0, 0, 0)), float) + p["vrel"] if obj not in ("S1",) or s < 12 else VMAIN + p["vrel"]
    q["rel_host"] = rel_host
    if pt == 4:
        q["sntime"] = SNtime(p, s)
        q["mass"] = np.where(q["sntime"] < 0, np.where(q["sntime"] == -1, p["m_init"], 0.3 * p["m_init"]), p["m_init"])
        q["mass"] = np.where(p["m_init"] == 0, 4.0, q["mass"])     # inert stars keep a mass
    return q

def cat_arrays(parts):
    keys = set.intersection(*[set(p) for p in parts]) if parts else set()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}

FIELDS = {0: ["mass", "rho", "u", "xe", "nha", "XH", "XO", "cool"], 1: ["mass"], 2: ["mass"],
          4: ["mass", "m_init", "tform", "XH", "XO", "bpos", "bd_phys", "sntime"]}

def build(s):
    groups, unbound = layout(s)
    order = {pt: [] for pt in (0, 1, 2, 4)}      # list of particle dicts in file order
    subs, grps, truth_ids = [], [], {}
    for gi, (sublist, fuzz) in enumerate(groups):
        glen = np.zeros(6, int); first_sub = len(subs)
        for name, members in sublist:
            slen = np.zeros(6, int); sub = dict(name=name, grnr=gi, pos=centre(name, s), vel=VMAIN + np.array(VREL[name], float), parts={})
            for pt in (0, 1, 2, 4):
                ps = [place(o, pt, s, name) for o in members]
                ps = [p for p in ps if p is not None]
                if not ps:
                    continue
                p = {k: np.concatenate([q[k] for q in ps]) for k in ["ids", "pos", "vel", "rel_host"] + FIELDS[pt]}
                o = np.argsort(np.linalg.norm(p["rel_host"], axis=1), kind="stable")     # binding order ~ radius
                p = {k: v[o] for k, v in p.items()}
                order[pt].append(p); sub["parts"][pt] = p; slen[pt] = len(p["ids"])
            sub["lentype"] = slen; glen += slen; subs.append(sub)
            truth_ids[name] = len(subs) - 1
        for o_ in fuzz:
            for pt in (0, 1, 2, 4):
                p = place(o_, pt, s, "main")
                if p is not None:
                    order[pt].append({k: p[k] for k in ["ids", "pos", "vel", "rel_host"] + FIELDS[pt]}); glen[pt] += len(p["ids"])
        grps.append(dict(lentype=glen, first=first_sub, nsubs=len(sublist), pos=subs[first_sub]["pos"], vel=subs[first_sub]["vel"],
                         r200=R200[s] if gi == 0 else 3.0, m200=3e9 if gi == 0 else 1e8))
    for o_ in unbound:
        p = place(o_, 0, s, "main")
        order[0].append({k: p[k] for k in ["ids", "pos", "vel", "rel_host"] + FIELDS[0]})
    return {pt: cat_arrays(v) for pt, v in order.items()}, subs, grps, truth_ids

def write(s):
    a, z = A[s], SNAPS[s]
    allp, subs, grps, truth_ids = build(s)
    d = os.path.join(OUT, f"snapdir_{s:03d}"); os.makedirs(d, exist_ok=True)
    n = np.array([len(allp[pt]["ids"]) if pt in allp and "ids" in allp[pt] else 0 for pt in range(6)])
    halves = [n // 2, n - n // 2]
    def hdr(f, npt):
        g = f.create_group("Header")
        for k, v in dict(BoxSize=L, HubbleParam=h, Omega0=0.307, OmegaLambda=0.693, OmegaBaryon=0.048, Redshift=z, Time=a,
                         UnitLength_in_cm=UL, UnitMass_in_g=UM, UnitVelocity_in_cm_per_s=UV, NumPart_ThisFile=npt,
                         NumPart_Total=n, NumPart_Total_HighWord=np.zeros(6, int), MassTable=np.zeros(6),
                         NumFilesPerSnapshot=2, Flag_DoublePrecision=1).items():
            g.attrs[k] = v
    dens_code = UM * h ** 2 / UL ** 3
    for fi in range(2):
        with h5py.File(os.path.join(d, f"snap_{s:03d}.{fi}.hdf5"), "w") as f:
            hdr(f, halves[fi])
            for pt in (0, 1, 2, 4):
                if n[pt] == 0:
                    continue
                sl = slice(0, halves[0][pt]) if fi == 0 else slice(halves[0][pt], n[pt])
                p = {k: v[sl] for k, v in allp[pt].items()}
                g = f.create_group(f"PartType{pt}")
                g["Coordinates"] = p["pos"]; g["Velocities"] = (p["vel"] / np.sqrt(a)).astype("f4")
                g["ParticleIDs"] = p["ids"].astype("u8"); g["Masses"] = (p["mass"] * h / 1e10).astype("f4")
                if pt == 0:
                    g["Density"] = (p["rho"] * a ** 3 / dens_code).astype("f4")
                    g["InternalEnergy"] = (p["u"] / UV ** 2).astype("f4")
                    g["ElectronAbundance"] = p["xe"].astype("f4"); g["NeutralHydrogenAbundance"] = p["nha"].astype("f4")
                    g["GFM_CoolingRate"] = p["cool"].astype("f4")
                if pt == 4:
                    a_f = a_of_t(p["tform"])
                    g["StellarFormationTime"] = a_f.astype("f4")
                    g["GFM_InitialMass"] = (p["m_init"] * h / 1e10)
                    g["SNTime"] = p["sntime"].astype("f4"); g["BirthPos"] = p["bpos"].astype("f4")
                    g["BirthDensity"] = p["bd_phys"] * a_f ** 3 / dens_code
                if pt in (0, 4):
                    met = np.zeros((len(p["mass"]), 10), "f4"); met[:, 0] = p["XH"]; met[:, 4] = p["XO"]; met[:, 1] = 1 - p["XH"] - 2.5 * p["XO"]
                    met[:, 9] = 1.5 * p["XO"]; g["GFM_Metals"] = met; g["GFM_Metallicity"] = (2.5 * p["XO"]).astype("f4")
    # catalogue
    gd = os.path.join(OUT, f"groups_{s:03d}"); os.makedirs(gd, exist_ok=True)
    with h5py.File(os.path.join(gd, f"fof_subhalo_tab_{s:03d}.0.hdf5"), "w") as f:
        g = f.create_group("Header")
        for k, v in dict(BoxSize=L, HubbleParam=h, Omega0=0.307, Redshift=z, Time=a, Ngroups_ThisFile=len(grps), Ngroups_Total=len(grps),
                         Nsubgroups_ThisFile=len(subs), Nsubgroups_Total=len(subs), NumFiles=1).items():
            g.attrs[k] = v
        G = f.create_group("Group")
        G["GroupLenType"] = np.array([x["lentype"] for x in grps], "i4"); G["GroupLen"] = G["GroupLenType"][()].sum(1)
        G["GroupFirstSub"] = np.array([x["first"] for x in grps], "i4"); G["GroupNsubs"] = np.array([x["nsubs"] for x in grps], "i4")
        G["GroupPos"] = np.array([x["pos"] for x in grps]); G["GroupVel"] = np.array([x["vel"] for x in grps]) * a
        G["Group_R_Crit200"] = np.array([x["r200"] * kpc_code(a) for x in grps], "f4")
        G["Group_M_Crit200"] = np.array([x["m200"] * h / 1e10 for x in grps], "f4")
        S = f.create_group("Subhalo")
        lt = np.array([x["lentype"] for x in subs]); S["SubhaloLenType"] = lt.astype("i4"); S["SubhaloLen"] = lt.sum(1).astype("i4")
        S["SubhaloGrNr"] = np.array([x["grnr"] for x in subs], "i4")
        S["SubhaloPos"] = np.array([x["pos"] for x in subs]); S["SubhaloVel"] = np.array([x["vel"] for x in subs], "f4")
        mt = np.zeros((len(subs), 6)); hm = np.zeros((len(subs), 6)); mets = {0: np.zeros((len(subs), 10)), 4: np.zeros((len(subs), 10))}
        for i, x in enumerate(subs):
            for pt, p in x["parts"].items():
                mt[i, pt] = p["mass"].sum()
                r = np.linalg.norm(p["rel_host"], axis=1); hm[i, pt] = np.median(r) * kpc_code(a)
                if pt in (0, 4):
                    mets[pt][i, 0] = np.average(p["XH"], weights=p["mass"]); mets[pt][i, 4] = np.average(p["XO"], weights=p["mass"])
        S["SubhaloMassType"] = (mt * h / 1e10).astype("f4"); S["SubhaloMass"] = (mt.sum(1) * h / 1e10).astype("f4")
        S["SubhaloHalfmassRadType"] = hm.astype("f4"); S["SubhaloMassInHalfRadType"] = (0.5 * mt * h / 1e10).astype("f4"); S["SubhaloHalfmassRad"] = hm.max(1).astype("f4")
        S["SubhaloVmax"] = np.sqrt(4.3e-6 * mt.sum(1) / np.maximum(hm[:, 1] / kpc_code(a), 1e-3)).astype("f4")
        S["SubhaloVmaxRad"] = hm[:, 1].astype("f4"); S["SubhaloVelDisp"] = np.full(len(subs), 20.0, "f4")
        S["SubhaloIDMostbound"] = np.array([x["parts"][1]["ids"][0] for x in subs], "u8")
        S["SubhaloGasMetalFractions"] = mets[0].astype("f4"); S["SubhaloStarMetalFractions"] = mets[4].astype("f4")
    # truth relative to the main centre [physical kpc]; masses in "Msun" of 1.989e33 g as written
# (lyratools converts with MSUN = 1.98841e33 g, i.e. 3e-4 higher)
    rel = {pt: np.asarray(periodic(allp[pt]["pos"] - CEN[s]) / kpc_code(a)) for pt in allp if "pos" in allp[pt]}
    r = {pt: np.linalg.norm(v, axis=1) for pt, v in rel.items()}
    st = allp[4]; ok = st["m_init"] > 0; age = T[s] - st["tform"]
    m200 = {pt: allp[pt]["mass"][(r[pt] < R200[s]) & (ok if pt == 4 else True)].sum() for pt in r}   # inert stars excluded
    return dict(t=T[s], a=a, R200=R200[s], subs=truth_ids, Mstars_R200=float(st["mass"][ok & (r[4] < R200[s])].sum()),
                MDM_R200=float(m200[1]), Mgas_1kpc=float(allp[0]["mass"][r[0] < 1].sum()),
                Contam_R200=float(m200[2] / sum(m200.values())),
                SFR10=float(st["m_init"][ok & (r[4] < 0.1 * R200[s]) & (age < 10)].sum() / 1e7),
                Pos_ckpc=(CEN[s] * 1e3 / h).tolist())

def periodic(d):
    return (d + L / 2) % L - L / 2

truth = {s: write(s) for s in SNAPS}
with open(os.path.join(OUT, "..", "truth.json"), "w") as f:
    json.dump({str(k): v for k, v in truth.items()}, f, indent=1)
print("made 3-snapshot time-series test data in", OUT)
