# lyratools

Analysis tools for LYRA / AREPO simulations. Snapshots and group catalogues are read with h5py
(only the needed index ranges and columns); scida is optional (`ds = lt.load_scida(path)`, then `lt.add_fields(ds)` for scida datasets).
A port of `TTT.py` (loadmodules) to an installable package.

```bash
pip install -e .                      # from a clone of the repository
pip install -e ".[scida]"             # optional: lyratools fields on scida datasets (scida + dask)

python scripts/fetch_data.py ~/lyra_data   # public data tables; lists the others and where to get them
export LYRATOOLS_DATA=~/lyra_data
```

Python >= 3.10; dependencies numpy, scipy, astropy, h5py, matplotlib. For Jupyter, register the environment
as a kernel (`python -m ipykernel install --user --name lyra`). License: MIT (`LICENSE`); how to cite:
`CITATION.cff`; changes: `CHANGELOG.md`.

```python
import lyratools as lt

lt.set_data_dir("/path/to/lyra_data")       # tables: yields, isochrones, ... (see "Data tables")
halo = lt.load_halo("output/snapdir_039", halo_id=0)   # FoF group as saved: catalogue centre, velocities as stored, physical units
# catalogue: groups_039 next to snapdir_039, or groupdir="..."; a scida dataset is also accepted (only its paths are used)
# halo = lt.load_halo(snapdir, halo_id=0, rest_frame="group")   # velocities in the frame of the catalogue GroupVel

t, sfr = lt.sfh.sfh(halo.stars.tform, halo.stars.m_init, dt=10)
rates = lt.flows.spherical_flow_rate(halo.gas.pos, halo.gas.vel, halo.gas.mass, radius=2, thickness=0.2)
```

## Conventions

Lengths **kpc**, velocities **km/s**, masses **Msun**, times **Myr** (TTT used Gyr), rates per **yr**,
metallicities as mass fractions unless the name says otherwise (`Z_Zsun`). Functions take plain arrays
in these units (pint quantities from scida are also converted) and return plain numpy arrays.

## Fields on a region

`load_halo` returns a `Region`; `halo.gas`, `halo.stars`, `halo.dm` are `Particles` (dicts that also allow
attribute access).

**Always in memory, plain numpy in lyratools units:** `pos` [kpc, relative to the centre], `vel` [km/s], `mass`
[Msun], `r` [kpc], `vr` [km/s]; stars also `tform`, `age` [Myr], `m_init` [Msun] and, with `SNTime`, `t_death` [Myr].
Precision `dtype=np.float32` by default (`dtype=np.float64` doubles the memory). Wind particles (formation time <= 0)
are dropped from `halo.stars`.

**Preloaded snapshot fields** (numpy, as stored, i.e. code units): `DEFAULT_FIELDS` in `region.py`, or exactly what you
list with `load_halo(snap, 0, fields={"gas": [...], "stars": [...], "dm": [...]})` (PartTypeN keys also work). Coordinates,
Velocities and Masses are always read; types not listed are not loaded. `GFM_Metals` and `BirthPos` are not preloaded
by default (read on first access; derived `nH`/`Temperature` read only the hydrogen column).

**Anything else is read on first access** (`halo.gas["Temperature"]`, `halo.stars.BirthDensity`, `halo.gas["nH"]`):
FoF members are one contiguous slice of each dataset (offsets from `GroupLenType`), so only that slice is read with
h5py, never the full snapshot. Box selections and later selections (`within`, `select`) keep track of their particles'
indices, so lazy fields stay aligned. Derived fields (`lyratools.fields`: `nH`, `Fe_H`, `Age`, ...) and, with a
catalogue, `SubhaloID` / `GroupID` (-1: not a member) are computed on first access. Fetched fields are kept;
`p.load("A", "B", ...)` reads several; `p.column("GFM_Metals", 4)` reads one column only. A missing name raises
`KeyError` listing what is in memory, the raw snapshot fields and the derived fields (`p.available()` lists them all).
`name in p` only reports what is already in memory. Fields read later (BirthPos, Velocities, Coordinates) are as
stored: `recenter`/`rotate` act on `pos` and `vel` only.

**Numpy in lyratools units:** `halo.gas["Density"]` is the stored code value; `halo.gas.values("Density")` -> g/cm^3,
`values("nH")` -> cm^-3, `values("InternalEnergy")` -> erg/g, `values("BirthDensity", "Msun/kpc**3")`,
`values("pos", "pc")` (any astropy unit string; only a scalar factor is computed). Code units come from the header's
UnitLength/Mass/Velocity with the comoving `a` and `h` factors removed (physical); `p.unit_of(name)` gives the unit.
Fields with unknown units (e.g. `GFM_StellarFormationTime`) are returned as stored.

**Metadata:** `halo.z`, `halo.a`, `halo.h`, `halo.t_now` [Myr], `halo.r200` [kpc], `halo.m200` [Msun] (physical, from
the catalogue; also `R200`, `M200`), `halo.halo_id`, `halo.center_code` (centre used, code units), `halo.center` (shift
applied since, kpc), `halo.center_kpc` (current centre, physical kpc in box coordinates).

**Selections:** `halo.within(r_kpc)` and `halo.within_r200(f)` return views of the region with every particle type
masked (metadata shared, arrays copied); `halo.select({"gas": mask, ...})` or `halo.select(lambda p: ...)` in general.

**Stars:** `halo.stars.alive(m_min=4)`: initial mass >= m_min and still alive by LYRA's `SNTime` (> 0: SN to come).
Stars with `SNTime = -1` (never a SN), and every star if the snapshot has no `SNTime` (with a warning), are judged from
the Raiteri et al. (1996) metallicity-dependent lifetimes.

**Temperature:** LYRA stores `Temperature`; the stored field is always used (a derived field never replaces a stored
one). Otherwise it is derived from `InternalEnergy` and `ElectronAbundance` with mu = 4 / (1 + 3X + 4X x_e),
which covers atomic and ionised H and He but not molecular gas.

**Abundances by number** (gas and stars, from `GFM_Metals`): `12logOH`; `[X/H]` as `X_H` and `[X/O]` as `X_O` for
C, N, Mg, Si, Mn, Fe (and He_H), relative to Asplund et al. (2009) (mass-fraction ratio over the solar one, which is the
number ratio over the solar number ratio). For reservoirs use ratios of summed masses, as the time-series code does:
`lt.chemistry.twelve_log_oh_reservoir(metals, mass)`, `lt.chemistry.abundance_ratio_reservoir(metals, mass, "N", "O")`.

## Isolated runs

`halo = lt.load_isolated("output/snapdir_102")` loads every particle of an isolated (non-cosmological) snapshot and
converts with the header's UnitLength/Mass/Velocity: positions kpc, velocities km/s, masses Msun, `Time` and
`StellarFormationTime` in code time units UnitLength/UnitVelocity -> `halo.t_now`, `halo.stars.tform` [Myr]. As in the
time-series code, a = 1, z = 0 and the header's HubbleParam is ignored (h = 1; pass `h=` to override). Positions are
centred on the median position of the inert initial-condition stars (`GFM_InitialMass = 0`), which are excluded from
`halo.stars` and kept as `halo.inert`. `values()` and all derived fields work as for cosmological runs. `SNTime` in the isolated FUV runs holds absolute code times (alive: time of death > `Time`; dead:
-time of death), not lifetimes in Gyr: `load_isolated` detects this (`halo.sntime_convention`) for `stars.t_death`, and the time-series code converts it to lifetimes for its SN counts and energies (before this fix it read death times as lifetimes and undercounted SNe in isolated runs, e.g. 12 instead of 22 in the last 10 Myr of isolated_fuv snapshot 102; rerun affected isolated time series with --redo).

## FUV-run snapshot fields

Fields written by LYRA runs with FUV heating (`lt.fields.FIELD_REGISTRY`): `FUV_G0` (Habing), `FUV_PEHeatingRate` (erg cm^3 s^-1),
`G0TableDebugBypass` (earlier `G0TableBypass`), and the derived `net_true`. In the old scheme `GFM_CoolingRate`
excludes the photoelectric heating, so `net_true = GFM_CoolingRate + FUV_PEHeatingRate`; with the G0 table it is part
of `GFM_CoolingRate` (except in bypassed cells), so `net_true = GFM_CoolingRate`. The scheme is read from the
snapshot's Config (`VIRGODWARF_G0_COOLING_TABLE`).

## Layout

| module | contents |
|---|---|
| `fields` | derived fields (numpy, computed on first access on a Region; `add_fields(ds)` registers them as scida recipes): gas `nH`, `Temperature` (only if not stored), `Pressure_k`, `CoolingTime`, `SelfShieldingFactor`, `net_true`; stars `FormationTime`, `Age`, `SNState`, `DeathTime`, `ESN`; both `Z_Zsun`, `12logOH` and `[X/Y]` ratios (`Fe_H`, `O_Fe`, `N_O`, `C_H`, ...); `FIELD_REGISTRY` (units and meaning of the FUV-run fields) |
| `region` | `load_halo(snap, halo_id, fields=..., center='group'|'subhalo'|coords, rest_frame=None|'group'|'subhalo'|'particles', selection='fof'|'box', dtype=..., groupdir=...)` loads the FoF group as saved (catalogue centre, no recomputation); `load_isolated(path)` for isolated runs; `open_snapshot`; `Region` (recenter, face_on, rotate, remove_bulk_velocity, within, within_r200, select, close); `Particles` (lazy fields, `values`, `unit_of`, `column`, `load`, `available`, `has`, `alive`); `shrinking_sphere` (explicit, optional) |
| `sfh` | `sfh`, `sfr_trailing`, `find_bursts`, `burstiness_stats`, `compute_ks`, `compute_ks_grid`, `sfr_from_density`, `phase_diagram` |
| `flows` | `spherical_flow_rate`, `disk_flow_rate`, `loading_factors`, `loading_factors_isolated` |
| `profiles` | `density_profile`, `rotation_curve`, `surface_brightness_profile`, `half_mass_radius`, `half_light_radius`, `ellipticity`, `radii` |
| `photometry` | `magnitudes` (PARSEC, resolved + unresolved stars), `add_magnitude_fields`, `luminosity`, `total_magnitude`, `jwst_zeropoint`, `imf_truncated` |
| `stellar_evolution` | `sn_energy`, `sn_mass_return`, `sn_metal_return`, `agb_returns`, `sn_energies`, LYRA SNTime helpers `sn_state`, `death_time`, `died_within`, `sntime_convention`, `sntime_as_lifetime` (absolute code times of death in isolated runs -> lifetimes) |
| `chemistry` | `abundance_ratio`, `ion_fraction`, `self_shielding_factor`, `cooling_time`, `cooling_time_within`, `dynamical_time_within` |
| `models` | `nfw`, `einasto`, `sersic`, `powerlaw`, `schechter`, KS relations, abundance matching, `relaxation_time`, `evaporation_factor`, `rho_crit`, `jeans_temperature`, `mag_at_sb` |
| `substructure` | `luminous_subhalos`, `per_subhalo`, `exsitu_stars`, `birth_radius`, `birth_positions_ckpc` |
| `timeseries` | main-progenitor track, per-snapshot main-galaxy quantities and substructure table, combined time series; driven by `scripts/lyra_timeseries.py` (see below) |
| `evolution` | `SubfindEvolution` (main halo across all group catalogs, `halo_track()`) |
| `cosmology` | `TimeConversion` (from the snapshot's cosmology), `snapshot_table`, `snap_at_z`, `z_at_snap` |
| `logs` | `SNLogData`, `PopIIILogData`, `CPUData`, `nparticles_ic`, `nparticles_timebins`, `start_logfile` |
| `observations` | `McConnachie2012`, `ks_sigma_figgs`, `ks_sigma_things` |
| `storage` | `add_to_storage`, `storage_path`, `write_to_file`, `add_to_file`, `add_dataset` |
| `voronoi` | mesh-aware images like `get_Aslice`: `column_density`, `projection`, `weighted_projection`, `cell_slice` (with cell-boundary `contours`), general `voronoi_grid`, optional gradient reconstruction |
| `darkmatter` | halo structure and core formation: `radial_profile`, `power_radius` (convergence), `local_slope`, `inner_slope`, `fit_profile` (NFW, gNFW, coreNFW, Burkert), `dicintio14_shape`, `core_formation_energy`, `velocity_profiles`, `spherical_potential`, `apocentres`, `match_ids` |
| `yields` | post-processing correction of C, N, O for metallicity-dependent yields: `YieldCorrection` (massive stars: Sukhbold table with metallicity-scaled birth composition, secondary N and winds; AGB: Karakas rescaled to FRUITY), `correct_particles`, `polluter_metallicity`, `plateau_diagnostic`, `number_ratio`, `twelve_log_oh` |
| `plotting` | `use_style("ttt" or "lyra")`, `add_zaxis`, `add_scale_bar`, `particle_image` |
| `diagnostics` | `grav_timestep` |

## Migration from TTT.py

| TTT.py | lyratools |
|---|---|
| `load_halo(sim, snap)` | `halo = lt.load_halo(snapdir, halo_id)` (reads only the FoF group, like `loadonlyhalo`) |
| `load_snapshot`, `calc_halo_indexes`, `get_subhalo_index`, `s.calc_sf_indizes` | `halo.stars["SubhaloID"]`, `["GroupID"]` per particle (from the catalogue offsets) |
| `set_time_table`, `t_at_a`, `a_at_t`, `z_at_t`, `t_at_z` | `tc = halo.tconv` (or `lt.TimeConversion.from_params(h, Om0, Ob0)`); `tc.t_at_a(a)` etc. (Myr) |
| `snap_at_z`, `z_at_snap` | `lt.cosmology.snapshot_table(output_dir)` + `snap_at_z` / `z_at_snap` |
| `get_temp`, `s.nH` | fields `Temperature`, `nH` |
| `get_age_in_Gyr`, `get_lookback_age` | fields `FormationTime`, `Age`; `halo.stars.tform`, `.age` |
| `check_sfr` (defined twice) | `sfh.sfr_from_density` |
| `SFH`, `SFH_in_r` | `sfh.sfh` (select stars with a mask first) |
| `phase`, `pressure_density` | `sfh.phase_diagram` with fields `nH`, `Temperature`, `Pressure_k` |
| `compute_KS`, `compute_KS_grid`, `KS_relation_*`, `ks_sigma_*` | `sfh.compute_ks`, `sfh.compute_ks_grid`, `models.ks_relation_*`, `observations.ks_sigma_*` |
| `DiskFlowRate`, `SphericalFlowRate` | `flows.disk_flow_rate`, `flows.spherical_flow_rate` (return dicts) |
| `get_loading_factor`, `get_loading_factors_non_cosmological` | `flows.loading_factors`, `flows.loading_factors_isolated` |
| `ESN`, `get_sn_energies`, `recent_sn`, `recent_agb` | `stellar_evolution.sn_energy`, `sn_energies` / field `ESN`, `died_within(massive=True/False)` |
| `Meject_SNII`, `Metaleject_SNII`, `get_SNII_returns`, `get_AGB_returns` | `sn_mass_return`, `sn_metal_return`, `dead_massive`, `dead_agb`, `agb_returns` |
| `get_abundance_ratio` | `chemistry.abundance_ratio` / fields `Fe_H`, `O_Fe`, ... |
| `get_ion_fraction`, `self_shielding_factor` | `chemistry.ion_fraction`, `chemistry.self_shielding_factor` / field |
| `get_coolingtime`, `coolingtime_inr`, `dynamicaltime_inr` | `chemistry.cooling_time` / field `CoolingTime`, `cooling_time_within`, `dynamical_time_within` |
| `rotcurve`, `density_profile`, `SB_profile` | `profiles.rotation_curve`, `profiles.density_profile`, `profiles.surface_brightness_profile` |
| `get_half_mass_radius`, `get_half_light_radius`, `get_ellipticity`, `get_radii` | `profiles.*` |
| `shrinking_sphere` | `lt.shrinking_sphere` (arrays) |
| `get_magnitudes`, `get_luminosities`, `get_total_magnitude`, `get_luminosity_in_solar`, `get_jwst_zeropoint` | `photometry.magnitudes` / `add_magnitude_fields`, `luminosity`, `total_magnitude`, `jwst_zeropoint` |
| `init_parsec_cube`, `init_integrated_cube`, `IMF_for_photometrics` | `photometry.parsec_cube`, `integrated_cube`, `imf_truncated` (cached, no globals) |
| `NFW`, `Einasto`, `Sersic`, `powerlaw`, `Schechter`, `*_abundance_matching`, `relaxation_time`, `rho_crit`, `jeansT`, `mag_at_sb` | `models.*` |
| `get_luminous_subhalos`, `get_subhalo_age`, `get_subhalo_gima`, `get_subhalo_stellar_indexes` | `substructure.luminous_subhalos`, `substructure.per_subhalo` |
| `get_evaporation_factor` | `models.evaporation_factor` with per-subhalo masses/ages from `per_subhalo` |
| `get_birth_radius`, `get_exsitu_stars` | `substructure.birth_radius`, `substructure.exsitu_stars` with `SubfindEvolution.halo_track()` |
| `SubfindEvolution`, `SNLogData`, `PopIIILogData`, `CPUData`, `McConnachie2012` | same names in `evolution`, `logs`, `observations` |
| `get_start_logfile`, `get_nparticles_ic`, `get_nparticles`, `get_floats_from_string` | `logs.start_logfile`, `nparticles_ic`, `nparticles_timebins`, `floats_from_string` |
| `add_to_storage`, `add_to_file`, `write_to_file`, `add_dataset` | `storage.*` (`add_to_storage` takes a path; `storage_path` builds the old one) |
| `add_zaxis`, `add_scale_bar`, `particle_image`, rcParams on import | `plotting.add_zaxis`, `add_scale_bar`, `particle_image`, `lt.use_style("ttt")` |
| `gadget_snap.get_Aslice(proj=True)` | `voronoi.projection` / `column_density` / `weighted_projection` (no compiled extension needed) |
| `gadget_snap.get_Aslice(proj=False)` | `voronoi.cell_slice` (returns `neighbours` and `contours`) |
| `get_grav_timestep` | `diagnostics.grav_timestep` |

## Behaviour changes and fixes

* **Units:** no hard-coded code units or `* 1e10` factors; code units come from the snapshot header.
* **No global state:** tables are cached per file path; `set_data_dir` replaces hard-coded paths.
* **Times in Myr** instead of Gyr, computed from the snapshot's cosmology.
* `get_coolingtime` divided seconds by 60·60·365 (missing a factor 24); `cooling_time` returns Myr.
* `coolingtime_inr` summed specific energies; `cooling_time_within` uses total thermal energy / total net cooling rate.
* `get_birth_radius` returned squared distances that were compared to radii; `birth_radius` returns distances.
* `relaxation_time` mixed code masses into the Spitzer formula; now Msun, pc in, Gyr out.
* `rho_crit` returned Omega_m x rho_crit; `models.rho_crit` and `models.rho_mean_matter` are separate.
* `get_evaporation_factor` called helpers with arguments they did not accept; replaced by `per_subhalo` + `models.evaporation_factor`.
* `IMF_for_photometrics` ignored its `mass_limit`; `imf_truncated` uses it (default 4 Msun as before).
* `scipy.interpolate.interp2d` and `scipy.integrate.simps` (removed from SciPy) replaced by `RegularGridInterpolator` and `simpson`.
* Log parsers use `float()` instead of `eval()`.
* New: `photometry.magnitudes(..., clip_to_table=True)` gives very metal-poor resolved stars the magnitudes at the
  lowest isochrone metallicity instead of the fill value 35 (the old behaviour, still the default).

## Time series of a zoom or isolated galaxy (`scripts/lyra_timeseries.py`)

```bash
python scripts/lyra_timeseries.py --snapdir output --groupdir output --out ts [--halo 0] [--snaps a:b] \
    [--nproc auto] [--config cfg.json] [--redo] [--dry-run] [--combine-only] [--cleanup]
```

* **Step 0** follows the main progenitor of FoF `--halo` (its GroupFirstSub at the last snapshot) back in time with its
  1000 most-bound DM IDs, rank-weighted as in SubLink; cached in `ts/main_track.h5`. Checks that subhalo particles are
  ordered by binding energy (else ranks by distance) and cross-checks against `SubfindEvolution`.
* **Step 1** writes `ts/snaps/snap_XXX.h5` per snapshot, one spawned process each (latest first; `--nproc auto` from
  memory): `/Main` (masses in apertures, SFRs from ages, sizes, shapes, kinematics, O/H, ISM phases, flows, SNe,
  times; projected quantities for x, y, z and `_faceon`) and `/Sub` (subhaloes within `R_sel` x R200, member-star
  properties, most-bound IDs and descendant in the next snapshot). Logs in `ts/logs/`.
* **Step 2** refuses to run unless every snapshot of the track has a complete file with the current config hash, then
  writes `ts/timeseries_<sim>.h5`: `/Header`, stacked `/Main`, `/MainTrack`, `/Sub/Catalogue/snap_XXX`, `/Sub/Tracks`.
* All parameters are in `lyratools.timeseries.DEFAULT_CONFIG` (override with `--config`); `--dry-run` prints how many
  subhaloes each `R_sel` / minimum-DM choice selects. Every dataset has a `units` attribute; energies are in 1e51 erg.
* Particle data are read with h5py by index range (only the selected rows and the H, O columns of `GFM_Metals`).
* **Isolated (non-cosmological) runs** are detected from `Parameters/ComovingIntegrationOn`, else from the header
  (`Time` > 1, or `Redshift` = 0 with `Omega0` = 0): a = 1, z = 0, h = 1 (header `HubbleParam` ignored), times in
  code time units, no Hubble flow. Without group catalogues Step 0 centres by a shrinking sphere on the stars, takes
  R200 from the enclosed mean density (200 rho_crit with `run.H0_kms_Mpc`, default 70; or a fixed `run.R200_kpc`)
  and the velocity frame within 0.1 R200, and `/Sub` is skipped. PartType2/3 count as initial-condition stars
  (`/Main/Mass/StarsIC`, with inert PartType4) instead of contamination. Settings in the `run` section of the config;
  see `TIMESERIES_OUTPUTS.md`.

## Correcting C/N/O for metallicity-dependent yields

LYRA's massive-star ejecta come from solar-metallicity Sukhbold models for every star, and its
AGB ejecta from Karakas. In the yields alone this gives log C/O ~ 0 (90% of massive-star carbon
is from solar-metallicity Wolf-Rayet winds) and log N/O ~ -0.7 (primary N from hot-bottom
burning in 4-8 Msun Karakas AGB stars), about 0.6-1 dex above observed dwarfs.
`lyratools.yields` corrects existing snapshots without rerunning: oxygen traces each cell's
massive-star ejecta, so their C and N can be swapped for metallicity-dependent values and the
AGB residual rescaled to FRUITY (Cristallo et al. 2011, 2015; table shipped in `data/`).

```python
from lyratools import yields
corr = yields.YieldCorrection.from_files(alpha=0.8, high_mass="karakas")   # uses lt.set_data_dir tables
out = yields.correct_particles(halo.gas, corr, agb_age_myr=500)              # z_eff defaults to the cell's own O
out["log_NO"], out["log_CO"], out["12logOH"]                                  # corrected; '_sim' suffix: uncorrected

# better estimate of the polluters' metallicity (instead of the cell's own, an upper bound):
zp = yields.polluter_metallicity(halo.gas.pos, halo.stars.pos, z_birth_O, halo.stars.m_init,
                                 halo.stars.t_death, halo.t_now, window_myr=50, radius=0.3,
                                 default=yields.oxygen_metallicity(o_frac))
out = yields.correct_particles(halo.gas, corr, z_eff=zp, agb_age_myr=500)
```

Brackets to report: `alpha` 0.7-0.85 (wind mass loss ~ Z^alpha), `high_mass` 'karakas' vs
'scaled' (FRUITY stops at 6 Msun), `z_eff` cell vs polluters, `secondary_n`, and the IMF upper
mass (`massive_range`; LYRA's value matters because the most massive stars' carbon is wind).
At 0.1 Zsun after 1 Gyr the corrected log N/O is -1.3 to -1.6 (observed -1.5 to -1.3); log C/O
drops by 0.3-0.5 dex to about -0.2 to -0.4, still above observed values (-1.0 to -0.4): the wind
scaling does not capture how weaker winds change core masses and explodability.
`YieldCorrection.from_files(high_mass_table="LC18")` replaces the scaled Sukhbold yields by the
metallicity-dependent LC18 + BR26 tables (`StellarEvolution_LC18_BR26_v{000,150,300}.txt` in the data
directory, or `lc18_tables=[...]`), mixed with the Prantzos et al. (2018) rotation weights and
interpolated in log X_O between the four LC18 bins (clamped outside); `alpha`/`secondary_n` are not
used. At 0.1 Zsun after 1 Gyr it gives log C/O = -0.42 and log N/O = -1.62 ('karakas') / -1.73 ('scaled').
The IMF-averaged LC18 oxygen yield is about twice Sukhbold's, so corrected O (and O/H) rise by ~0.3 dex.
Use `out["negative_N"]` / `out["negative_C"]` to find cells where the decomposition fails
(enriched by too few stars for IMF-averaged yields to apply); results are meaningful as averages
over regions or populations, not cell by cell.

## Data tables

`lt.set_data_dir(path)` (or `LYRATOOLS_DATA`) should contain the files listed in `lt.config.DATA_FILES`, with
the same names as before (e.g. `StellarEvolution_Sukhbold2015.txt`, `MH_?_*/agebin0.03*` isochrones,
`JWST_NirCam_filters/...`). Files kept elsewhere: `lt.set_data_file("mcconnachie", "/path/to/NearbyGalaxies_Jan2021_PUBLIC.fits")`.

Each entry records the reference to cite, where the file comes from and, for public tables that may be
redistributed, a download URL; `lt.config.describe()` prints them, and a missing file raises an error that says
where to get it. `python scripts/fetch_data.py DIR` downloads the public ones (McConnachie 2012 catalogue,
Behroozi et al. 2013 release; with `--large` also the 2.2 GB Ploeckinger & Schaye 2020 table) and lists the rest
(LYRA input tables, PARSEC isochrones from the CMD web interface, JWST zeropoints from JDox).

## Tests

`python tests/run_tests.py` builds synthetic tables, a synthetic LYRA snapshot, a small two-file zoom snapshot with a
two-group catalogue (`tests/make_zoom.py`, used by `tests/test_region.py` for region field access and unit
conversions, plus `load_isolated` on the synthetic isolated run) and a 3-snapshot zoom sequence, and checks every module; it exits non-zero if any check fails.
GitHub Actions (`.github/workflows/tests.yml`) runs the tests on Python 3.10-3.12 with and
without scida.
If scida is installed, `test_all.py` and `test_region.py` also check the optional scida helpers (`add_fields`,
`add_magnitude_fields`, `load_halo` on a scida dataset); scida builds the cache of a multi-file snapshot in a process
pool, so scripts that `load` multi-file snapshots with scida need an `if __name__ == "__main__":` guard.

scida's default unit file for cosmological AREPO runs assumes `code_length = ckpc/h`; LYRA runs use Mpc
(`UnitLength_in_cm = 3.086e24`), so with a plain `scida.load` lengths and densities are wrong by powers of 1000.
`lt.load_scida(path, catalog=None)` loads the snapshot and its catalogue with the code units of the header, and
`lt.add_fields(ds)` refuses a dataset whose `code_length` disagrees with the header.
