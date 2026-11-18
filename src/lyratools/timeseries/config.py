"""
Configuration of the time-series pipeline: one nested dict, stored as JSON in every output file
together with its hash. Override entries with a JSON file (--config), e.g. {"sub": {"R_sel_R200": 5}}.
"""
import copy
import hashlib
import json
import os
import subprocess

DEFAULT_CONFIG = {
    "schema": 1,
    "track": {
        "halo": 0,                  # FoF group at the final snapshot; its GroupFirstSub is the target
        "N_seed": 1000,             # most-bound DM IDs used as seeds
        "rank_alpha": 1.0,          # seed weight (rank + 1)^-alpha, as in SubLink
        "search_R200": 3.0,         # search radius in R200 (comoving) of the later snapshot
        "min_search_ckpc": 10.0,    # floor of the search radius (groups with R200 = 0)
        "min_dm": 100,              # stop when the progenitor has fewer DM particles
        "check_n_subhalos": 20,     # subhaloes used for the binding-order check
    },
    "main": {
        "load_R200": 1.5,
        "apertures_kpc": [0.04, 0.1, 0.2, 0.5, 1.0, 2.0],
        "apertures_R200": [0.1, 0.5, 1.0],
        "gal_R200": 0.1,
        "T_cold": 1e4,              # K, cold gas (rotation axis, dispersions, V/sigma, Rhalf)
        "nH_cold": 10.0,            # cm^-3, additional cut for cold-gas O/H
        "sfh_window_Myr": 300.0,
        "sfh_bin_Myr": 1.0,
        "sfr_windows_Myr": [5, 10, 20, 50, 100, 200],
        "young_Myr": [10, 100],
        "proj_radius_kpc": 1.0,     # aperture for Young10 and cold-gas LOS dispersions
        "axis_radii_kpc": [0.1],    # plus R200
        "gradient_kpc": [0.1, 1.0],
        "gradient_nbins": 10,
        "phase_T": [1e2, 1e4, 10 ** 5.5],
        "nH_hist": [-4.0, 5.0, 0.25],
        "flow_shells_kpc": [0.04, 0.2, 0.5, 1.0],
        "flow_shells_R200": [0.25, 0.5, 1.0],
        "flow_thickness": 0.2,      # shell thickness / radius (old script: fixed 0.1 R200, 0.1 kpc, 0.01 kpc)
        "flow_phase_T": [1e4, 1e5],
        "sn_windows_Myr": [10, 50],
        "min_particles_shape": 10,
    },
    # run type and centring; defaults keep the hashes of earlier cosmological runs (see config_hash)
    "run": {
        "cosmological": "auto",     # true / false / "auto" (Parameters/ComovingIntegrationOn, else header)
        "center": "auto",           # "subfind" (needs catalogues), "shrink" (particles only), "auto"
        "center_on": "stars",       # shrinking sphere on "stars" (+ IC stars), "dm" or "all"
        "center_guess_kpc": None,   # start of the shrinking sphere at the last snapshot (default: centre of mass)
        "shrink_factor": 0.9,
        "shrink_min_n": 1000,
        "shrink_r0_kpc": 20.0,      # first radius at earlier snapshots, around the later centre
        "R200_kpc": "auto",         # physical kpc, or "auto": mean density 200 rho_crit(H0)
        "H0_kms_Mpc": 70.0,         # for R200 in isolated runs (cosmological runs: header cosmology)
        "R200_search_kpc": 1000.0,
        "vel_R200": 0.1,            # velocity frame: mass-weighted velocity of all particles within this
        "pt23": "auto",             # PartType2/3: "contamination" (zooms), "stars" (isolated ICs), "ignore"
    },
    "sub": {
        "R_sel_R200": 3.0,          # placeholder: set from --dry-run
        "margin_R200": 0.5,         # extra search radius for descendants
        "min_dm": 100,              # placeholder: set from --dry-run
        "min_stars": 50,
        "N_track": 100,
        "rank_alpha": 1.0,
        "formed_windows_Myr": [10, 50, 100],
        "born_in_main_R200": 0.1,
        "dry_run_R_sel": [2, 3, 5],
        "dry_run_min_dm": [50, 100, 300],
    },
}


def merge(base, override):
    """Recursively update a copy of `base` with `override` (unknown keys are an error)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k not in out:
            raise KeyError(f"unknown config key '{k}'")
        out[k] = merge(out[k], v) if isinstance(out[k], dict) and isinstance(v, dict) else v
    return out


def load_config(path=None, **sections):
    cfg = DEFAULT_CONFIG
    if path:
        with open(path) as f:
            cfg = merge(cfg, json.load(f))
    for k, v in sections.items():
        cfg = merge(cfg, {k: v})
    return cfg


def to_json(cfg):
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def hashable(cfg):
    """cfg without the "run" section when it has its defaults, so earlier files keep their hashes."""
    if cfg.get("run") == DEFAULT_CONFIG["run"]:
        cfg = {k: v for k, v in cfg.items() if k != "run"}
    return cfg


def config_hash(cfg):
    return hashlib.sha256(to_json(hashable(cfg)).encode()).hexdigest()[:16]


def provenance():
    """lyratools version and git hash (or 'unknown' outside a git checkout)."""
    from .. import __version__
    try:
        git = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        git = "unknown"
    return {"lyratools_version": __version__, "lyratools_git": git}
