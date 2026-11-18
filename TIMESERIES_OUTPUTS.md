# Time-series outputs: reference for analysis code

Everything an analysis pipeline needs to read the outputs of `scripts/lyra_timeseries.py`:
file layout, indexing, units, the exact definition of every quantity, and the caveats.
Definitions refer to the code in `src/lyratools/timeseries/` (`main.py`, `subhalos.py`, `combine.py`).

## 1. Files in `--out`

| file | content | use for analysis |
|---|---|---|
| `timeseries_<sim>.h5` | everything below, stacked over snapshots | **yes, the only file needed** |
| `main_track.h5` | Step 0 main-progenitor track (same as `/MainTrack`) | no |
| `snaps/snap_XXX.h5` | per-snapshot `/Main` and `/Sub` (inputs of the combine step) | only if the combined file is missing |
| `config.json` | the configuration used | it is also in the file attributes |
| `logs/` | per-snapshot logs | debugging |

`<sim>` is `--sim`, or the parent directory of `output/`.

## 2. Conventions

- **Run type.** `f.attrs["cosmological"]` is 1 for cosmological runs and 0 for isolated runs. Isolated runs have
  `a` = 1 and `z` = 0 at every snapshot, times since t = 0 of the simulation, comoving = physical (`*_ckpc` equal the
  physical values), and no Hubble flow. If they had no group catalogues, `/Sub` is absent and `/MainTrack` comes
  from particles (see section 4).

- **Snapshot axis.** Every time-series array has the snapshot as its first axis (or the second axis for
  `/Sub/Tracks`), ordered as `/Header/Snap` (increasing; only snapshots on the main track). An index `k`
  is a position in `/Header/Snap`, **not** a snapshot number.
- **Units:** physical kpc, Msun, Myr, km/s, yr⁻¹ rates; energies in **1e51 erg**. Most `/Main` and
  `/Sub/Catalogue` datasets carry a `units` attribute; `/MainTrack` and `/Sub/Tracks` do not (see tables).
- **Time:** `/Header/t_Myr` is cosmic time since the Big Bang from the header scale factor (lyratools
  `TimeConversion` with the file cosmology); for isolated runs, `Time` × UnitLength/UnitVelocity in Myr. Use `/MainTrack/t_Myr` (float64) for differences of close times.
- **Centre and frame:** the main progenitor's `SubhaloPos` (centre) and `SubhaloVel` (velocity frame).
  `/Main` velocities are peculiar, with **no Hubble flow**; `/Sub` velocities **include** it.
- **"gal"** = sphere of radius `gal_R200 × R200` = 0.1 R200 around the centre. R200 = `Group_R_Crit200` of the
  FoF group containing the main progenitor (the host's R200 if the progenitor is a satellite).
- **Stars** exclude `GFM_InitialMass == 0` (inert initial-condition stars) and wind particles (formation time ≤ 0).
  SFRs come from stellar ages × initial masses, never from the gas `StarFormationRate`.
- **Metallicity:** only 12 + log(O/H) by number from the GFM_Metals H and O mass fractions. No total Z or Fe.
  "MW" (reservoir) values are ratios of summed O and H masses, not means of per-particle logs.
- **Missing values:** NaN for floats (empty selection, quantity undefined); −1 for integers.
  `Times/Cool/*` is `inf` when the gas is net heating.
- **Precision:** stored floats are float32 (computed in float64). IDs are uint64; subhalo indices int64.
- **Booleans** are stored as int8 (`astype(bool)`).
- **Projections:** datasets with a trailing axis of length 3 named `...Proj`, `...LOS...` or `SFR10_inRe_proj` are
  projections along the box x, y, z axes; the `_faceon` sibling is along `RotAxis`.

### File attributes (`f.attrs`)

| attr | meaning |
|---|---|
| `sim` | simulation name |
| `config` | JSON string of the full configuration (`json.loads`) |
| `config_hash` | 16-hex hash of `config`; identical for files that are comparable |
| `n_snap`, `n_tracks` | lengths of the snapshot and track axes (`n_tracks` = 0 without `/Sub`) |
| `cosmological` | 1 cosmological, 0 isolated |
| `main_desc_mismatches` | number of snapshots where the subhalo-descendant link of the main progenitor disagreed with Step 0 (0 expected) |
| `lyratools_version`, `lyratools_git`, ... | provenance |

The default configuration (`timeseries/config.py`) fixes the apertures, windows and thresholds quoted below;
read the actual values from `config`, since a run may override them.

## 3. `/Header` — (Nsnap,)

`Snap` (int64), `a`, `z`, `t_Myr` (float32).

## 4. `/MainTrack` — Step 0 main progenitor, (Nsnap, ...)

Followed backwards from FoF group `--halo` at the last snapshot by the 1000 most-bound DM IDs
(rank-weighted, SubLink-like). No `units` attributes:

| dataset | shape | units / meaning |
|---|---|---|
| `Snap` | (N,) | snapshot number |
| `a`, `z`, `t_Myr` | (N,) float64 | scale factor, redshift, Myr |
| `SubhaloID` | (N,) | global SUBFIND subhalo index in that snapshot's catalogue |
| `GroupID` | (N,) | FoF group index |
| `is_central` | (N,) bool | progenitor is its group's first subhalo |
| `Pos_code` | (N,3) float64 | SubhaloPos, code units (ckpc/h × unit length) |
| `Pos_ckpc` | (N,3) | comoving kpc (box coordinates) |
| `Vel` | (N,3) | SubhaloVel, peculiar km/s |
| `M200c` | (N,) | Msun, `Group_M_Crit200` |
| `R200c` | (N,) | **physical** kpc |
| `R200c_ckpc` | (N,) | comoving kpc |
| `SubhaloMassType` | (N,6) | Msun per particle type |
| `SubhaloLenType` | (N,6) | particle counts per type |
| `SeedFraction` | (N,) | rank-weighted fraction of the later snapshot's seed IDs found in this progenitor (1 = clean link; low values flag a doubtful step, e.g. a major merger) |

### Without group catalogues (`run.center = "shrink"`, the default for isolated runs without SUBFIND)

The centre comes from a shrinking sphere (centre of mass within r, r × 0.9 per step, until fewer than 1000
particles) on the stars (PartType4, plus PartType2/3 when they count as stars), started at the centre of mass at
the last snapshot and at the later snapshot's centre (within 20 kpc) for earlier ones. R200 is the largest radius
where the mean enclosed density of all particles is ≥ 200 ρ_crit, with ρ_crit from `run.H0_kms_Mpc` (default 70)
for isolated runs, or `run.R200_kpc` if set. M200c is the mass within it. `Vel` is the mass-weighted velocity of all
particles within 0.1 R200. `SubhaloMassType` and `SubhaloLenType` are the masses and counts within R200;
`SubhaloID` and `GroupID` are −1, `is_central` false and `SeedFraction` NaN.

## 5. `/Main` — main-progenitor quantities, (Nsnap, ...)

Particles of all types within 1.5 R200 of the centre. Column order of the aperture arrays (`Apertures_kpc`,
shape (N, 9), physical kpc at each snapshot): **0.04, 0.1, 0.2, 0.5, 1, 2 kpc, then 0.1, 0.5, 1 R200**.
Spheres, r < aperture.

### Masses
| dataset | shape | definition |
|---|---|---|
| `Apertures_kpc` | (N,9) | aperture radii, kpc |
| `Mass/Gas`, `Mass/DM`, `Mass/Stars` | (N,9) | enclosed current mass, Msun |
| `Mass/StarsInit` | (N,9) | enclosed initial stellar mass |
| `Mass/HI` | (N,9) | Σ X_H × NeutralHydrogenAbundance × m (no H₂ correction) |
| `Mass/GasO`, `Mass/StarsO` | (N,9) | oxygen mass in gas / stars |
| `Mtrue_inRe/{DM,Gas,Stars}` | (N,) | mass within the 3D stellar half-mass radius `Rhalf/Stars3D` |
| `Contam_R200` | (N,) | mass fraction of PartType2+3 within R200; NaN unless `run.pt23` is `contamination` (zooms) |
| `Mass/StarsIC` | (N,9) | only when `run.pt23` = `stars` (isolated default): stars from the initial conditions, PartType2/3 plus PartType4 with GFM_InitialMass = 0. They have no ages or abundances and are in no other stellar quantity |

### Star formation (gal stars unless noted)
| dataset | shape | definition |
|---|---|---|
| `SFR/{5,10,20,50,100,200}` | (N,) | Σ m_init(age < w) / w, Msun/yr |
| `SFR_R200/{...}` | (N,) | same, all stars within R200 |
| `SFH_recent` | (N,300) | Msun/yr in 1-Myr age bins; bin i = ages [i, i+1) Myr |
| `MassFormedSincePrev` | (N,) | Σ m_init of gal stars formed after the previous snapshot's time; NaN at k = 0 |
| `Age/MassWeighted`, `Age/Median` | (N,) | current-mass-weighted mean / median age, Myr |
| `SFR10_inRe_proj` (+`_faceon`) | (N,3) | SFR10 of stars within the projected half-mass radius |
| `KS/Sigma_gas` | (N,) | gas mass in a face-on cylinder of radius and half-height 0.1 R200 / area, Msun/pc² |
| `KS/Sigma_SFR10`, `KS/Sigma_SFR100` | (N,) | same cylinder, Msun/yr/kpc² |
| `BirthDensity/Young10` | (N,3) | 16/50/84th percentiles of n_H at birth of stars younger than 10 Myr, cm⁻³ |

### Sizes, kinematics, shapes
| dataset | shape | definition |
|---|---|---|
| `Rhalf/Stars3D` | (N,) | 3D half (current) mass radius of gal stars ("Re" elsewhere) |
| `Rhalf/Young10`, `Rhalf/Young100` | (N,) | 3D half initial-mass radius of stars < 10 / 100 Myr |
| `Rhalf/Gas`, `Rhalf/ColdGas` | (N,) | gal gas; cold = T < 1e4 K |
| `Rhalf/StarsProj` (+`_faceon`) | (N,3) | projected half-mass radius |
| `RotAxis` | (N,3) | unit angular-momentum vector (box frame) defining face-on |
| `RotAxisSource` | (N,) int | 0 cold gas, 1 all gas, 2 stars (all in gal), −1 none (face-on values NaN) |
| `Sigma/StarsLOS_Re` (+`_faceon`) | (N,3) | mass-weighted LOS dispersion of stars within projected Re, km/s |
| `Sigma/Young10LOS` (+`_faceon`) | (N,3) | stars < 10 Myr within projected R < 1 kpc |
| `Sigma/ColdGasLOS_1kpc` (+`_faceon`) | (N,3) | cold gas within projected R < 1 kpc |
| `VoverSigma/{Stars,ColdGas}` | (N,) | mean v_φ / (3D dispersion about it / √3), face-on frame |
| `AxisRatios/{Stars,DM}/{0.1kpc,R200}` | (N,2) | b/a, c/a from the mass-weighted second-moment tensor (not reduced, not iterative); NaN if < 10 particles |

### Oxygen abundance, 12 + log(O/H)
| dataset | definition |
|---|---|
| `OH/GasMW/1kpc` | all gas r < 1 kpc |
| `OH/GasMW/0.1R200` | all gal gas |
| `OH/ColdGasMW/1kpc` | gas r < 1 kpc with T < 1e4 K and n_H > 10 cm⁻³ |
| `OH/StarsMW_Re` | stars within `Rhalf/Stars3D` |
| `OH/Young10/mean`, `/std` | m_init-weighted mean and std of per-star values, stars < 10 Myr |
| `OH/Gradient/per_kpc` | linear fit of binned O/H of cold dense gal gas vs face-on projected R, 10 bins in 0.1–1 kpc (3D r if no axis); NaN with < 3 filled bins |
| `OH/Gradient/per_Re` | slope × face-on projected Re (3D Re if no axis) |

### ISM phases (gal gas)
| dataset | definition |
|---|---|
| `Phases/{Mass,Volume}/{CNM,WNM,WIM,HIM}` | T < 1e2; 1e2–1e4; 1e4–10^5.5; ≥ 10^5.5 K. Volume = Σ m/ρ, kpc³ |
| `Phases/nH_hist` (N,36) | gas mass per log n_H bin; edges in attribute `log_nH_edges` (−4 to 5, step 0.25) |

### Gas flows: `Flows/{Out,In}/{Mass,OMass,Energy}/{shell}/{phase}`, (N,)
- shells: `0.04kpc, 0.2kpc, 0.5kpc, 1kpc, 0.25R200, 0.5R200, 1R200`; each a spherical shell of thickness
  0.2 × radius centred on it.
- phases: `all`, `cold` (T < 1e4), `warm` (1e4–1e5), `hot` (≥ 1e5 K).
- rate = Σ m |v_r| / Δr over cells with v_r > 0 (Out) or < 0 (In). Mass and OMass in Msun/yr;
  Energy (kinetic in the progenitor frame + thermal) in 1e51 erg/yr.
- `Flows/Out/vr_p50`, `vr_p90` (km/s): mass-weighted median / 90th percentile of v_r of outflowing gas.
- v_r is peculiar (no Hubble flow), so at 1 R200 the inflow/outflow split is relative to the halo, not
  the expanding background.

### Supernovae and timescales
| dataset | definition |
|---|---|
| `SN/N_10Myr`, `SN/N_50Myr` (int) | gal stars of m_init > 8 Msun whose death time (from `SNTime`) falls in the last 10 / 50 Myr. `SNTime` is a signed lifetime in Gyr in cosmological runs and a signed absolute code time of death in isolated LYRA runs; the convention of isolated snapshots is detected per snapshot (`stellar_evolution.sntime_convention`) and converted |
| `SN/E_10Myr`, `SN/E_50Myr` | their energy (Sukhbold tables; failed SNe 1e49 erg), 1e51 erg. Divide by the window for a rate |
| `Times/Dyn/{0.1kpc,1kpc,R200}` | 1/√(4πGρ̄) of all matter within r, Myr |
| `Times/Cool/{...}` | Σ thermal energy / net cooling rate of the gas within r, Myr; `inf` if net heating, NaN if no cooling rates |

## 6. `/Sub/Catalogue/snap_XXX` — subhaloes around the main progenitor, one group per snapshot

Rows are the **selected** subhaloes: SubhaloPos within `R_sel_R200 × R200` of the main progenitor, no
PartType2/3 members, and ≥ `min_dm` DM or ≥ `min_stars` star particles (defaults 3, 100, 50; check
`config["sub"]`). The main progenitor itself is always a row (`is_main == 1`).

| dataset | shape | units / meaning |
|---|---|---|
| `SubhaloID` | (n,) | global SUBFIND index in this snapshot |
| `GrNr`, `is_central`, `is_main` | (n,) | FoF group; is first subhalo of its group; is the main progenitor |
| `Pos` | (n,3) | kpc physical, relative to the main progenitor (box axes, periodic) |
| `Vel` | (n,3) | km/s, peculiar difference **+ Hubble flow** |
| `dist`, `v_rad` | (n,) | kpc; km/s (radial, + Hubble flow; > 0 moving away) |
| `inside_R200` | (n,) int8 | dist < R200 of the main progenitor |
| `Mass`, `MassType` | (n,), (n,6) | Msun (SUBFIND bound masses) |
| `LenType` | (n,6) | particle counts |
| `HalfmassRadType`, `VmaxRad` | (n,6), (n,) | kpc |
| `Vmax`, `VelDisp` | (n,) | km/s |
| `OH_gas_cat`, `OH_star_cat` | (n,) | 12+log(O/H) from SUBFIND's `Subhalo{Gas,Star}MetalFractions` (NaN if absent) |
| `N_stars` | (n,) | member stars used below |
| `M_init` | (n,) | Σ initial mass of member stars, Msun |
| `M_formed_{10,50,100}` | (n,) | initial mass formed in the last 10/50/100 Myr, Msun |
| `AgeMedian`, `AgeP10`, `AgeP90` | (n,) | initial-mass-weighted age percentiles, Myr |
| `OH_star_mean`, `OH_star_std` | (n,) | current-mass-weighted mean/std of per-star 12+log(O/H) |
| `f_born_in_main` | (n,) | initial-mass fraction of member stars whose `BirthPos` lies within 0.1 R200 of the main track at their formation time. Only stars formed between the first and last tracked snapshot count; NaN if none |
| `DescSubhaloID` | (n,) | descendant's global SUBFIND index in the **next** snapshot of `/Header/Snap`, or −1 (none / last snapshot) |
| `DescShare` | (n,) | rank-weighted share of the tracking IDs found in that descendant; NaN at the last snapshot |
| `TrackIDs`, `TrackRank`, `TrackType`, `TrackOffsets` | ragged | the up-to-100 most-bound DM (type 1) then star (type 4) IDs of each row: row i owns `TrackIDs[TrackOffsets[i]:TrackOffsets[i+1]]` |

Within a snapshot, a subhalo's row is found by `np.flatnonzero(SubhaloID == sid)`. The rows do not line up
between snapshots; use `/Sub/Tracks` for histories.

## 7. `/Sub/Tracks` — subhalo histories, (Ntr, Nsnap, ...)

One row per track, built by following `DescSubhaloID` from snapshot to snapshot. Column k = `/Header/Snap[k]`;
values are NaN (SubhaloID −1) where the track does not exist. No `units` attributes; units are those of the
catalogue column of the same name.

| dataset | shape | meaning |
|---|---|---|
| `SubhaloID` | (Ntr,N) | catalogue index at each snapshot, −1 if absent; links back to `/Sub/Catalogue` |
| `dist`, `v_rad`, `Mass`, `Vmax`, `AgeMedian`, `OH_star_mean`, `OH_star_std`, `M_formed_10`, `DescShare` | (Ntr,N) | catalogue columns |
| `MassType` | (Ntr,N,6) | catalogue column |
| `Rhalf_star` | (Ntr,N) | `HalfmassRadType[:, 4]`, kpc |
| `first_snap`, `last_snap` | (Ntr,) | snapshot **numbers** of the first and last appearance |
| `fate` | (Ntr,) bytes | `survives`, `merged_main`, `merged_other`, `lost` (`.astype(str)`) |
| `merged_into` | (Ntr,) | track index it merged into, −1 otherwise |
| `is_main` | (Ntr,) int8 | the main-progenitor track (exactly one, follows `/MainTrack`) |
| `M_peak`, `snap_peak` | (Ntr,) | maximum `Mass` along the track and its snapshot number |
| `Vmax_peak` | (Ntr,) | maximum `Vmax` |
| `snap_infall` | (Ntr,) | first snapshot number with `inside_R200`, −1 if never |
| `f_born_in_main` | (Ntr,) | value at the last snapshot where it is defined |

Linking rules: when several rows share a descendant, the main progenitor (otherwise the most massive by `Mass`)
continues the track and the others end with fate `merged_*`. A track never restarts.

## 8. Caveats for the analysis

1. **`lost` is not "disrupted".** A track ends as `lost` when the descendant is not found **or** when the
   descendant exists but fails the selection in the next snapshot: it left R_sel × R200, dropped below
   `min_dm`/`min_stars`, or acquired low-resolution particles. To separate stripping from leaving the
   volume, look at `dist`, `Mass` and `DescShare` at `last_snap`.
2. **Tracks start at selection, not at formation.** `first_snap` is the first snapshot in which the subhalo
   was selected (inside R_sel × R200, enough particles). Subhaloes that leave and re-enter, or briefly drop
   below the threshold, appear as two tracks: the first `lost`, the second new.
3. **Survivors.** `survives` means the track reaches the last snapshot of the run. The main track always survives.
4. **SUBFIND switching.** Near pericentre the main progenitor can briefly lose mass to a satellite; check
   `/MainTrack/SeedFraction` and `is_central`, and `f.attrs["main_desc_mismatches"]`.
5. **Satellite-host R200.** If `/MainTrack/is_central` is ever false, R200, apertures in R200 and "gal" use
   the host group's R200 at those snapshots.
6. **`SFH_recent` windows overlap** between snapshots (300 Myr window, snapshot spacing ~10–40 Myr). Do not
   sum them; for a full SFH use the last snapshot's (bias: stars now in gal, including accreted ones) or
   `SFR/*` per snapshot.
7. **Stellar quantities in subhaloes** use SUBFIND member stars (bound), while `/Main` uses spheres.
   Do not compare `MassType[:, 4]` of the main row with `Mass/Stars` directly.
8. **Velocity frames differ:** `/Main` velocities are peculiar about SubhaloVel; `/Sub` velocities include
   the Hubble flow.
9. **Resolution.** Subhaloes near `min_dm` (100 DM) have noisy `Vmax`, `Mass` and descendant links; consider a
   stricter cut in analysis (e.g. `M_peak` above a few hundred particle masses) for rates and mass functions.
10. **Contamination.** Check `Contam_R200` before using a snapshot; values ≳ 1e−3 are suspicious.
    For isolated runs it is NaN; there, sizes, shapes and dispersions describe the stars formed in the run only,
    and the initial disc/bulge appears only in `Mass/StarsIC`.
11. **Not in the files:** gas SFRs, total metallicity, Fe, H₂, radial profiles, maps, particle-level data.
    These need the snapshots.

## 9. Loading recipe

```python
import json, h5py, numpy as np

def load_timeseries(path):
    """Nested dict of numpy arrays; Sub/Catalogue kept as {snap: {field: array}}."""
    out = {}
    with h5py.File(path, "r") as f:
        out["attrs"] = dict(f.attrs)
        out["config"] = json.loads(f.attrs["config"])
        for grp in ("Header", "MainTrack", "Main", "Sub/Tracks"):
            d = {}
            f[grp].visititems(lambda n, o: d.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
            out[grp] = d
        out["Sub/Tracks"]["fate"] = out["Sub/Tracks"]["fate"].astype(str)
        out["Sub/Catalogue"] = {int(k[5:]): {n: v[()] for n, v in g.items()}
                                for k, g in f["Sub/Catalogue"].items()}
    return out

ts = load_timeseries("timeseries_zhalo1.h5")
t, z = ts["Header"]["t_Myr"], ts["Header"]["z"]
mstar_gal = ts["Main"]["Mass/Stars"][:, 6]               # column 6 = 0.1 R200
sfr10 = ts["Main"]["SFR/10"]
T = ts["Sub/Tracks"]
lum = np.nanmax(T["MassType"][..., 4], axis=1) > 0       # tracks that ever had stars
accreted = (T["fate"] == "merged_main") & lum
```

Units of a `/Main` dataset: `f["Main"][name].attrs["units"]`.
