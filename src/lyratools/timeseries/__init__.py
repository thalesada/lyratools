"""
Time series of a LYRA zoom: the main progenitor and its substructure at every snapshot (Tier 1).

    python scripts/lyra_timeseries.py --snapdir D --groupdir D --out D [--halo 0] [--snaps a:b]
        [--nproc auto] [--config cfg.json] [--redo] [--dry-run] [--combine-only] [--cleanup]

Step 0 (track):   main progenitor followed back by its most-bound DM IDs -> <out>/main_track.h5
Step 1 (per snap): /Main quantities and the /Sub table -> <out>/snaps/snap_XXX.h5 (parallel)
Step 2 (combine):  <out>/timeseries_<sim>.h5 with stacked /Main, /MainTrack, /Sub/Catalogue, /Sub/Tracks

Units: physical kpc, Msun, Myr, km/s (rates per yr, energies in erg). All parameters are in
config.DEFAULT_CONFIG; the configuration is stored as JSON with its hash in every output file.
"""
from . import combine, config, io, main, snapshot, subhalos, track
from .config import DEFAULT_CONFIG, config_hash, load_config
