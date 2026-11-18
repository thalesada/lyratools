"""Synthetic stand-ins for every external table, in the formats TTT.py read."""
import os, numpy as np, h5py
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data", "tables"); os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)

# Sukhbold (LYRA layout, 170 columns): M, E_SN, 0, 83 supernova element masses (H first), 0, 83 wind element masses
m = np.linspace(9, 120, 40)
esn = np.where((m > 22) & (m < 25), 0.0, 1e51 * (0.5 + m / 100))
def block(scale_h, scale_c, scale_n, scale_o):
    b = np.zeros((len(m), 83)); b[:, 0] = scale_h; b[:, 1] = 0.4 * scale_h; b[:, 5] = scale_c; b[:, 6] = scale_n; b[:, 7] = scale_o
    b[:, 9:] = 1e-6; return b
sn = block(0.4 * m, 0.01 * m, 0.002 * m, 0.03 * m)
wind = block(0.1 * m, 0.02 * (m > 40) * m + 1e-4, 1e-4 * m, 6.2e-3 * 0.14 * m)
np.savetxt(f"{D}/StellarEvolution_Sukhbold2015.txt", np.column_stack([m, esn, np.zeros_like(m), sn, np.zeros_like(m), wind]),
           header=f"{len(m)} 0 84", comments="")

# LC18 BR26 STEM tables (one per rotation velocity): header 'N_mass N_Z N_el', rows Z-major:
# X_O, M, E_SN, SN block[N_el], wind block[N_el] (column i = atomic number i; H=1, C=6, N=7, O=8)
lc_xo = [2.0472e-05, 2.0472e-04, 2.0472e-03, 5.7706e-03]; lc_m = np.array([8.0, 15.0, 30.0, 60.0, 120.0])
for kv, v in enumerate(["000", "150", "300"]):
    rows = []
    for kz, xo in enumerate(lc_xo):
        for mm in lc_m:
            sn = np.zeros(84); wd = np.zeros(84)
            sn[1] = 0.4 * mm; sn[6] = 0.01 * mm * (1 + kv); sn[7] = 1e-3 * mm * (1 + kz) * (1 + kv); sn[8] = 0.03 * mm * (1 + 0.1 * kz)
            wd[1] = 0.1 * mm * (1 + kz); wd[6] = 1e-3 * mm * (1 + kz); wd[7] = 1e-4 * mm * (1 + kv); wd[8] = xo * 0.1 * mm
            rows.append(np.concatenate([[xo, mm, 1e51], sn, wd]))
    np.savetxt(f"{D}/StellarEvolution_LC18_BR26_v{v}.txt", np.array(rows), header=f"{len(lc_m)} {len(lc_xo)} 84", comments="")

# Karakas (LYRA layout, 241 columns): Z, M, 121 zeros, 118 element masses (H at column 123); 6 Z x 30 masses, Z-major
Zs = np.array([1e-4, 1e-3, 2.8e-3, 7e-3, 0.014, 0.03]); ms = np.linspace(0.75, 8, 30)
rows = []
for Z in Zs:
    for mm in ms:
        el = np.zeros(118); el[0] = 0.7 * (mm - 0.6); el[1] = 0.28 * (mm - 0.6)
        el[5] = 2e-3 * mm; el[6] = (1e-3 + 0.05 * (mm > 4)) * mm * (1 + 0.0 * Z); el[7] = (Z / 0.014) * 5.8e-3 * mm + 1e-5
        rows.append(np.concatenate([[Z, mm], np.zeros(121), el]))
np.savetxt(f"{D}/StellarEvolution_Karakas.txt", np.array(rows), header="30 6 119", comments="")

# solar composition: name, mass fraction
sol = np.full(30, 1e-6); sol[0] = 0.7381; sol[1] = 0.2485; sol[7] = 5.7e-3; sol[25] = 1.29e-3; sol[11] = 7e-4
with open(f"{D}/solar_composition_asplund_ARAA09.txt", "w") as f:
    f.write("el X\n"); [f.write(f"{i} {v}\n") for i, v in enumerate(sol)]

# Rahmati et al. 2013 table: z, alpha1, alpha2, beta, xi, n0, f
np.savetxt(f"{D}/SelfShielding_Rahmati12", np.array([[0, -3.98, -1.09, 1.29, 1, -2.94, 0.01],
                                                     [5, -2.28, -0.84, 1.75, 1, -1.79, 0.03],
                                                     [10, -2.28, -0.84, 1.75, 1, -1.79, 0.03]]))

# PARSEC-like isochrones: three [M/H] files, ages 6.0..10.1, masses 0.1..100
for k, mh in enumerate([-2.0, -1.0, 0.0]):
    d = f"{D}/MH_{k}_x"; os.makedirs(d, exist_ok=True)
    names = ["#", "Zini", "MH", "logAge", "Mini"] + [f"c{i}" for i in range(5, 28)] + ["mbolmag", "Umag", "Bmag", "Vmag"]
    lines = [f"# header {i}\n" for i in range(13)] + [" ".join(names) + "\n"]
    for la in np.arange(6.0, 10.11, 0.5):
        for mi in np.geomspace(0.1, 100, 60):
            L = mi ** 3.5 * np.exp(-(10 ** la / 1e6) / (1e4 * mi ** -2.5))
            V = 4.8 - 2.5 * np.log10(max(L, 1e-12)) + 0.1 * mh
            row = [0.0152 * 10 ** mh, mh, la, mi] + [0.0] * 23 + [V - 0.1, V + 0.3, V + 0.2, V]
            lines.append(" ".join(f"{v:.6g}" for v in row) + "\n")
    open(f"{d}/agebin0.03.dat", "w").writelines(lines)

# Ploeckinger-like ion table
with h5py.File(f"{D}/UVB_dust0_CR0_G0_shield0.hdf5", "w") as f:
    f["ElementNamesShort"] = np.array([b"H", b"He", b"C", b"N", b"O", b"Ne", b"Mg", b"Si", b"S", b"Ca", b"Fe"])
    zb, tb, mb, db = np.linspace(0, 9, 4), np.linspace(1, 9, 9), np.linspace(-4, 0.5, 5), np.linspace(-8, 4, 7)
    for n, v in [("RedshiftBins", zb), ("TemperatureBins", tb), ("MetallicityBins", mb), ("DensityBins", db)]:
        f[f"TableBins/{n}"] = v
    f["Tdep/IonFractions/06magnesium"] = -np.ones((4, 9, 5, 7, 13))

# KS points, Behroozi relation, JWST zeropoints
np.savetxt(f"{D}/KS.txt", rng.uniform(0, 1, (11, 5)))
os.makedirs(f"{D}/release-sfh_z0_z8_052913/smmr", exist_ok=True)
lm = np.linspace(9, 14, 20)
np.savetxt(f"{D}/release-sfh_z0_z8_052913/smmr/c_smmr_then_z0.10_red_all_smf_m1p1s1_bolshoi_fullcosmos_ms.dat",
           np.column_stack([lm, -3 + 0.5 * (lm - 9), 0.1 * np.ones(20), 0.1 * np.ones(20)]))
os.makedirs(f"{D}/JWST_NirCam_filters", exist_ok=True)
with open(f"{D}/JWST_NirCam_filters/NRC_ZPs_1126pmap.txt", "w") as f:
    f.writelines([f"# h{i}\n" for i in range(6)])
    for det in ["A1", "B1"]:
        f.write(f" x | NRC{det}_F200W | x | x | 25.9{det[-1]} | 759.6 \n")
        f.write(f" x | NRC{det}_F444W | x | x | 27.8{det[-1]} | 280.4 \n")

# McConnachie-like FITS
from astropy.io import fits
cols = {}
for n in ["dmod", "dmod-", "dmod+", "rh", "rh-", "rh+", "Vmag", "Vmag-", "Vmag+", "Bmag", "Bmag-", "Bmag+",
          "[Fe/H]", "feh-", "feh+", "e=1-b/a", "e-", "e+", "sigma_s", "sigma_s-", "sigma_s+", "vrot_s", "vrot_s-", "vrot_s+"]:
    v = rng.uniform(0.1, 2, 5); v[0] = 999; cols[n] = v
cols["dmod"] = np.array([24.5, 20, 22, 18, 25.0])
fits.BinTableHDU.from_columns([fits.Column(name=k, format="D", array=v) for k, v in cols.items()]).writeto(
    f"{D}/NearbyGalaxies_Jan2021_PUBLIC.fits", overwrite=True)
print("tables written")
