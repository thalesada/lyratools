"""
Build LYRA/STEM-format stellar yield tables with metallicity-dependent yields.

Massive stars: Limongi & Chieffi (2018), set R, from the NuPyCEE tables (NuGrid; B. Cote),
               rotation 0, 150, 300 km/s, or the metallicity-dependent mix of Prantzos et al. (2018).
AGB stars:     FRUITY (Cristallo et al. 2011, 2015), completed with Karakas outside 1.3-6 Msun.

LYRA table format (inferred from StellarEvolution_Sukhbold2015.txt and StellarEvolution_Karakas.txt):
    header:  N_mass  N_Z  N_elements
    rows:    [Z if N_Z > 0]  M  E_SN  supernova block (N_elements: element 0 'nn', H, He, ...)
                                      wind block      (N_elements)
    masses are total ejected masses [Msun] per star; a failed explosion has E_SN = 0, an empty
    supernova block and its (pre-collapse) ejecta in the wind block.

Usage:
    python make_lyra_yield_tables.py --nupycee /path/to/NuPyCEE/yield_tables \
        --sukhbold StellarEvolution_Sukhbold2015.txt --karakas StellarEvolution_Karakas.txt --out tables/
"""
import argparse
import os
import re

import numpy as np

LC18_FILES = {
    "R000": "agb_and_massive_stars_K10_LC18_R000.txt",
    "R150": "agb_and_massive_stars_K10_LC18_R150.txt",
    "R300": "agb_and_massive_stars_K10_LC18_R300.txt",
    "Rmix": "agb_and_massive_stars_C15_LC18_R_mix.txt",
}
FRUITY_FILE = "agb_and_massive_stars_C15_N13_0_0_HNe.txt"
LC18_Z = [3.236e-5, 3.236e-4, 3.236e-3, 1.345e-2]      # native LC18 metallicities ([Fe/H] = -3, -2, -1, 0)
LC18_EXPLODE_MAX = 25.0                                # set R: stars above 25 Msun collapse to black holes
N_EL_MASSIVE, N_EL_AGB = 84, 119


# ------------------------------------------------------------------------------------------ reading
def read_nupycee(path):
    """{(M, Z): (element masses by atomic number [0..118], M_final)} from a NuPyCEE yield file."""
    txt = open(path).read()
    parts = re.split(r"H Table: \(M=([\d.]+),Z=([\d.eE+-]+)\)", txt)[1:]
    out = {}
    for i in range(0, len(parts), 3):
        M, Z, body = float(parts[i]), float(parts[i + 1]), parts[i + 2]
        el = np.zeros(N_EL_AGB)
        for line in body.splitlines():
            if line.startswith("&") and not line.startswith("&Isotopes"):
                p = [s.strip() for s in line[1:].split("&")]
                y, zat = float(p[1]), int(p[3])
                if y > 1e-29:
                    el[zat] += y
        mf = re.search(r"H Mfinal: ([\d.eE+-]+)", body)
        out[(M, Z)] = (el, float(mf.group(1)) if mf else np.nan)
    return out


def read_lyra(path):
    """Read a LYRA-format table: returns header (nM, nZ, nEl) and arrays Z, M, E, SN block, wind block."""
    with open(path) as f:
        nM, nZ, nEl = (int(v) for v in f.readline().split())
    d = np.atleast_2d(np.loadtxt(path, skiprows=1))
    off = 1 if nZ > 0 else 0
    Z = d[:, 0] if nZ > 0 else np.full(len(d), np.nan)
    M, E = d[:, off], d[:, off + 1]
    sn, wind = d[:, off + 2:off + 2 + nEl], d[:, off + 2 + nEl:off + 2 + 2 * nEl]
    return (nM, nZ, nEl), Z, M, E, sn, wind


def write_lyra(path, Z, M, E, sn, wind, n_el, comment=None):
    """Write a LYRA-format table (Z column only if more than one metallicity / Z given)."""
    Zs, Ms = np.unique(Z[np.isfinite(Z)]), np.unique(M)
    with_z = np.isfinite(Z).any()
    with open(path, "w") as f:
        f.write(f"{len(Ms)} {len(Zs) if with_z else 0} {n_el}\n")
        for i in range(len(M)):
            row = ([Z[i]] if with_z else []) + [M[i], E[i]] + list(sn[i][:n_el]) + list(wind[i][:n_el])
            f.write(" ".join(f"{v:.4e}" if j >= (3 if with_z else 2) else f"{v:g}" for j, v in enumerate(row)) + "\n")
    if comment:
        with open(path + ".README", "w") as f:
            f.write(comment.strip() + "\n")


# ------------------------------------------------------------------------------------------ massive stars
def build_massive(nupycee_dir, variant, sukhbold_path, energy_mode="mean"):
    """
    LC18 set R yields in LYRA format, one row per (Z, M), Z-major.
    Masses 9-12 Msun (below LC18's grid) use LC18's 13 Msun ejecta composition scaled by the total
    ejected mass of LYRA's Sukhbold models (so the composition stays at the right metallicity). E_SN: exploding stars (M <= 25) get the IMF-averaged
    energy of exploding Sukhbold stars in 9-25 Msun ('mean') ; stars above 25 Msun get 0.
    """
    tab = read_nupycee(os.path.join(nupycee_dir, LC18_FILES[variant]))
    zs_avail = sorted({z for (m, z) in tab if m > 8})
    zs = [z for z in LC18_Z if any(np.isclose(z, za, rtol=1e-3) for za in zs_avail)]
    zs = [min(zs_avail, key=lambda za: abs(za - z)) for z in zs]
    m_lc = sorted({m for (m, z) in tab if m > 8})

    (_, _, nS), _, mS, eS, snS, wS = read_lyra(sukhbold_path)
    totS = snS + wS
    def sukh(m):        # Sukhbold total ejecta per element at mass m (interpolated)
        return np.array([np.interp(m, mS, totS[:, k]) for k in range(nS)])
    expl = (eS > 0) & (mS <= LC18_EXPLODE_MAX)
    w = mS ** -2.3 * np.gradient(mS)
    e_expl = np.sum((w * eS)[expl]) / np.sum(w[expl])

    m_ext = [9.0, 10.0, 11.0, 12.0]
    # a grid point just above the explosion limit (collapse, winds of the 30 Msun model scaled by mass),
    # so that interpolation in mass does not blend an explosion with a collapse between 25 and 30 Msun
    m_edge = LC18_EXPLODE_MAX + 0.01
    masses = m_ext + sorted(m_lc + [m_edge])
    Z, M, E, SN, W = [], [], [], [], []
    for z in zs:
        y13 = tab[(13.0, z)][0][:N_EL_MASSIVE]
        s13 = sukh(13.0)[:N_EL_MASSIVE]
        for m in masses:
            if m < 13.0:
                # keep LC18's 13 Msun composition (correct metallicity), scale by Sukhbold's ejected mass
                y = y13 * sukh(m).sum() / s13.sum()
            elif np.isclose(m, m_edge):
                y = tab[(30.0, z)][0][:N_EL_MASSIVE] * m / 30.0
            else:
                y = tab[(m, z)][0][:N_EL_MASSIVE]
            explode = m <= LC18_EXPLODE_MAX
            Z.append(z); M.append(m); E.append(e_expl if explode else 0.0)
            SN.append(y if explode else np.zeros(N_EL_MASSIVE))
            W.append(np.zeros(N_EL_MASSIVE) if explode else y)
    return np.array(Z), np.array(M), np.array(E), np.array(SN), np.array(W), e_expl


# ------------------------------------------------------------------------------------------ AGB stars
def build_agb(nupycee_dir, karakas_path, high_mass="karakas"):
    """
    FRUITY AGB yields on FRUITY's metallicity grid, in Karakas format. Mass grid: FRUITY's own
    models (1.3-6 Msun), extended with Karakas's grid points outside that range. Below 1.3 Msun: Karakas. Above 6 Msun: Karakas ('karakas') or FRUITY's 6 Msun model scaled
    by the ejected mass of the Karakas models ('scaled'). Karakas is interpolated in log Z
    (clamped to its grid) to FRUITY's metallicities.
    """
    fr = read_nupycee(os.path.join(nupycee_dir, FRUITY_FILE))
    fr = {k: v for k, v in fr.items() if k[0] <= 8 and k[1] <= 0.02}
    zf = sorted({z for (m, z) in fr}); mf = sorted({m for (m, z) in fr})
    # repair implausible entries: an intermediate-mass model (>= 4 Msun) leaving a remnant below
    # 0.6 Msun gets the remnant mass of the nearest metallicity, keeping its ejecta composition
    # (affects the 6 Msun, Z = 1e-4 model: stated final mass 0.43 Msun vs ~1.0 Msun at other Z)
    repaired = []
    for (m, z), (el, mfin) in list(fr.items()):
        rem = m - el.sum()
        if m >= 4 and rem < 0.6:
            j = zf.index(z); zn = zf[j + 1] if j + 1 < len(zf) else zf[j - 1]
            rem_n = m - fr[(m, zn)][0].sum()
            fr[(m, z)] = (el * (m - rem_n) / el.sum(), rem_n)
            repaired.append(f"M={m}, Z={z}: remnant {rem:.2f} -> {rem_n:.2f} Msun")
    build_agb.repaired = repaired
    (_, _, nK), ZK, MK, _, _, wK = read_lyra(karakas_path)
    zk, mk = np.unique(ZK), np.unique(MK)
    kar = np.zeros((len(zk), len(mk), nK))
    for i in range(len(ZK)):
        kar[np.searchsorted(zk, ZK[i]), np.searchsorted(mk, MK[i])] = wK[i]

    def kar_at(z):      # Karakas ejecta on its mass grid at metallicity z (log-Z interpolation, clamped)
        lz, lg = np.log10(np.clip(z, zk[0], zk[-1])), np.log10(zk)
        j = int(np.clip(np.searchsorted(lg, lz) - 1, 0, len(lg) - 2))
        f = (lz - lg[j]) / (lg[j + 1] - lg[j])
        return (1 - f) * kar[j] + f * kar[j + 1]

    # mass grid: Karakas below and above FRUITY's range, FRUITY's native models in between
    grid = [m for m in mk if m < mf[0]] + list(mf) + [m for m in mk if m > mf[-1]]
    i6 = int(np.argmin(abs(mk - mf[-1])))
    Z, M, E, SN, W = [], [], [], [], []
    for z in zf:
        k = kar_at(z)
        for m in grid:
            if m < mf[0] or (m > mf[-1] and high_mass == "karakas"):
                y = k[np.searchsorted(mk, m)]
            elif m <= mf[-1]:
                y = fr[(m, z)][0][:nK]
            else:
                y = fr[(mf[-1], z)][0][:nK] * k[np.searchsorted(mk, m)].sum() / k[i6].sum()
            Z.append(z); M.append(m); E.append(0.0); SN.append(np.zeros(nK)); W.append(y)
    return np.array(Z), np.array(M), np.array(E), np.array(SN), np.array(W)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nupycee", required=True, help="NuPyCEE yield_tables directory")
    ap.add_argument("--sukhbold", required=True)
    ap.add_argument("--karakas", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for variant in LC18_FILES:
        Z, M, E, SN, W, e_expl = build_massive(a.nupycee, variant, a.sukhbold)
        name = f"StellarEvolution_LC18_{variant}"
        write_lyra(os.path.join(a.out, name + ".txt"), Z, M, E, SN, W, N_EL_MASSIVE, comment=f"""
Limongi & Chieffi (2018) massive-star yields, set R, rotation {variant}, in LYRA/STEM format.
Source: NuPyCEE yield_tables/{LC18_FILES[variant]} (NuGrid; prepared by B. Cote); native LC18 metallicities only.
Header: N_mass N_Z N_elements. Rows: Z M E_SN [84 supernova element masses] [84 wind element masses] (element 0 = 'nn').
Set R: stars <= 25 Msun explode (total ejecta, incl. pre-supernova winds, in the supernova block);
stars > 25 Msun collapse to black holes (E_SN = 0, ejecta = winds in the wind block), as in LYRA's Sukhbold table.
A grid point at 25.01 Msun (collapse) keeps mass interpolation from blending explosions and collapses.
E_SN of exploding stars: IMF-averaged energy of exploding Sukhbold stars with M <= 25 Msun ({e_expl:.3e} erg).
9-12 Msun: LC18 13 Msun ejecta composition, scaled by the total ejected mass of the Sukhbold models (extrapolation).
NOTE: LYRA must read the massive-star table with a metallicity dimension (N_Z > 0), as it does for the AGB table.""")
        # single-metallicity versions in the exact Sukhbold layout (no Z column), readable without code changes
        for z in np.unique(Z):
            s = Z == z
            write_lyra(os.path.join(a.out, f"{name}_Z{z:.3g}.txt"), np.full(s.sum(), np.nan), M[s], E[s], SN[s], W[s], N_EL_MASSIVE)

    for hm in ("karakas", "scaled"):
        Z, M, E, SN, W = build_agb(a.nupycee, a.karakas, hm)
        write_lyra(os.path.join(a.out, f"StellarEvolution_FRUITY_{hm}6to8.txt"), Z, M, E, SN, W, N_EL_AGB, comment=f"""
FRUITY AGB yields (Cristallo et al. 2011, 2015) in LYRA/STEM (Karakas) format, 0.75-8 Msun (FRUITY's native masses, Karakas grid points outside 1.3-6 Msun).
Source: NuPyCEE yield_tables/{FRUITY_FILE}; FRUITY metallicities 1e-4 to 0.02.
Below 1.3 Msun: Karakas (interpolated in log Z). Above 6 Msun: {'Karakas' if hm == 'karakas' else 'FRUITY 6 Msun model scaled by the Karakas ejected mass'}.
Header: N_mass N_Z N_elements. Rows: Z M E_SN(=0) [119 zeros] [119 element masses] (element 0 = 'nn').
Repaired source entries (ejecta rescaled to the remnant mass of the neighbouring metallicity): {'; '.join(build_agb.repaired) or 'none'}.""")
    print("written to", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
