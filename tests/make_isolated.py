# synthetic non-cosmological (isolated) galaxy for lyratools.timeseries: 3 snapshots, physical kpc /
# 1e10 Msun / km/s code units, Time in code time units, no group catalogues. HubbleParam = 0.7 in the
# header must be ignored. A truncated singular isothermal DM halo (analytic R200), an old PartType2
# disc and PartType4 inert stars from the initial conditions, a cold rotating gas disc, and stars
# forming at a constant rate; the whole galaxy moves with a bulk velocity.
import json, os
import h5py, numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data", "iso", "output")
UL, UM, UV = 3.085678e21, 1.989e43, 1e5
MYR = UL / UV / 3.15576e13                    # Myr per code time unit
mp, kB = 1.67262192e-24, 1.380649e-16
TIMES = {0: 0.1, 1: 0.2, 2: 0.3}              # code time
X0, V = np.array([400.0, 500.0, 600.0]), np.array([50.0, -20.0, 10.0])
M_T, R_T = 1e11, 300.0                        # halo mass [Msun] and truncation radius [kpc]
H0 = 70.0
rng = np.random.default_rng(11)

def iso_dirs(n):
    v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1)[:, None]

def disc(n, Rd, zd, vrot):
    R = rng.exponential(Rd, n); phi = rng.uniform(0, 2 * np.pi, n)
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), rng.normal(0, zd, n)])
    vel = np.column_stack([-vrot * np.sin(phi), vrot * np.cos(phi), np.zeros(n)]) + rng.normal(0, 10, (n, 3))
    return pos, vel

nh = 100000
halo_pos = iso_dirs(nh) * rng.uniform(0, R_T, nh)[:, None]      # dM/dr constant: M(<r) = M_T r / R_T
halo_vel = rng.normal(0, 80, (nh, 3))
old_pos, old_vel = disc(20000, 2.0, 0.2, 150.0)
inert_pos, inert_vel = disc(500, 1.0, 0.2, 100.0)
gas_pos, gas_vel = disc(20000, 2.0, 0.1, 150.0)
ns = 4000
R = 3.0 * np.sqrt(rng.uniform(0, 1, ns)); phi = rng.uniform(0, 2 * np.pi, ns)
new_pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), rng.normal(0, 0.1, ns)])
new_vel = np.column_stack([-150 * np.sin(phi), 150 * np.cos(phi), np.zeros(ns)])
tform = rng.uniform(0.0, max(TIMES.values()), ns)                # code time
m_init = np.where(rng.random(ns) < 0.1, 20.0, 4.0)
XH, XO_gas, XO_star = 0.75, 4e-3, 2e-3

def write(s):
    t = TIMES[s]
    c = X0 + V * t                                                # code length = kpc, velocity km/s
    born = tform <= t
    age = (t - tform[born]) * MYR
    # isolated LYRA runs store absolute code times of death (alive: > Time; dead: -time of death); lifetime 10 Myr
    t_death = tform[born] + 10.0 / MYR
    snt = np.where(m_init[born] < 8, -1.0, np.where(age > 10.0, -t_death, t_death))
    d = os.path.join(OUT, f"snapdir_{s:03d}"); os.makedirs(d, exist_ok=True)
    n = np.array([len(gas_pos), nh, len(old_pos), 0, born.sum() + len(inert_pos), 0])
    with h5py.File(os.path.join(d, f"snap_{s:03d}.0.hdf5"), "w") as f:
        g = f.create_group("Header")
        for k, v in dict(BoxSize=1000.0, HubbleParam=0.7, Omega0=0.0, OmegaLambda=0.0, OmegaBaryon=0.0, Redshift=0.0, Time=t,
                         UnitLength_in_cm=UL, UnitMass_in_g=UM, UnitVelocity_in_cm_per_s=UV, NumPart_ThisFile=n, NumPart_Total=n,
                         NumPart_Total_HighWord=np.zeros(6, int), MassTable=np.zeros(6), NumFilesPerSnapshot=1).items():
            g.attrs[k] = v
        nid = [1]
        def part(pt, pos, vel, mass):
            g = f.create_group(f"PartType{pt}")
            g["Coordinates"] = pos + c; g["Velocities"] = (vel + V).astype("f4")
            g["ParticleIDs"] = np.arange(nid[0], nid[0] + len(pos), dtype="u8"); nid[0] += len(pos)
            g["Masses"] = (np.broadcast_to(mass, len(pos)) / 1e10).astype("f4")
            return g
        g = part(0, gas_pos, gas_vel, 1e5)
        ng = len(gas_pos); nH = np.full(ng, 50.0); Tk = np.full(ng, 5e3); xe = np.full(ng, 0.01)
        mu = 4 / (1 + 3 * XH + 4 * XH * xe)
        g["Density"] = (nH * mp / XH / (UM / UL ** 3)).astype("f4")
        g["InternalEnergy"] = (Tk * kB / (2 / 3) / (mu * mp) / UV ** 2).astype("f4")
        g["ElectronAbundance"] = xe.astype("f4"); g["NeutralHydrogenAbundance"] = np.full(ng, 0.9, "f4")
        g["GFM_CoolingRate"] = np.full(ng, -1e-23, "f4")
        met = np.zeros((ng, 10), "f4"); met[:, 0] = XH; met[:, 4] = XO_gas; g["GFM_Metals"] = met
        part(1, halo_pos, halo_vel, M_T / nh)
        part(2, old_pos, old_vel, 1e5)
        pos4 = np.concatenate([new_pos[born], inert_pos]); vel4 = np.concatenate([new_vel[born], inert_vel])
        m4 = np.concatenate([0.9 * m_init[born], np.full(len(inert_pos), 1e4)])
        g = part(4, pos4, vel4, m4)
        g["GFM_InitialMass"] = np.concatenate([m_init[born], np.zeros(len(inert_pos))]) / 1e10
        g["GFM_StellarFormationTime"] = np.concatenate([tform[born], np.zeros(len(inert_pos))]).astype("f4")
        g["SNTime"] = np.concatenate([snt, np.full(len(inert_pos), -1.0)]).astype("f4")
        g["BirthDensity"] = np.full(len(m4), 100 * mp / XH / (UM / UL ** 3), "f4")
        met = np.zeros((len(m4), 10), "f4"); met[:, 0] = XH; met[:, 4] = XO_star; g["GFM_Metals"] = met
    rho_c = 3 * (H0 / 3.085678e19) ** 2 / (8 * np.pi * 6.6743e-8) * 3.085678e21 ** 3 / 1.98841e33
    m_bar = len(gas_pos) * 1e5 + len(old_pos) * 1e5 + np.sum(m4)          # baryons, all well inside R200
    rr = np.linspace(10, 200, 200001)
    r200 = rr[np.flatnonzero((M_T * rr / R_T + m_bar) / (4 / 3 * np.pi * rr ** 3) >= 200 * rho_c)[-1]]
    return dict(t=t * MYR, centre=c.tolist(), vel=V.tolist(), R200=r200,
                SFR10=float(np.sum(m_init[born][age < 10]) / 1e7),
                Mstars=float(np.sum(0.9 * m_init[born])), MIC=float(len(old_pos) * 1e5 + len(inert_pos) * 1e4),
                N_SN10=int(np.sum((m_init[born] >= 8) & (age > 10) & (age <= 20))))

truth = {s: write(s) for s in TIMES}
json.dump(truth, open(os.path.join(OUT, "..", "truth.json"), "w"), indent=1)
