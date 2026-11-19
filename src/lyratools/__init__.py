"""
lyratools: analysis tools for LYRA / AREPO simulations (snapshots read with h5py).

    import lyratools as lt

    lt.set_data_dir("/path/to/lyra_data")      # stellar yields, isochrones, ... (see lt.config.DATA_FILES)
    halo = lt.load_halo("output/snapdir_039", halo_id=0)   # FoF group as saved, centred on the catalogue GroupPos
    halo.gas["nH"], halo.stars["Fe_H"]          # derived fields (lyratools.fields), computed on first access
    t, sfr = lt.sfh.sfh(halo.stars.tform, halo.stars.m_init, dt=10)

Conventions: lengths kpc, velocities km/s, masses Msun, times Myr, rates per yr,
metallicities as mass fractions unless a name says otherwise (e.g. Z_Zsun).
"""
from . import (chemistry, config, constants, cosmology, darkmatter, diagnostics, evolution, fields, flows, logs, models,
               observations, photometry, plotting, profiles, region, sfh, stellar_evolution, storage, substructure, timeseries,
               voronoi, yields)
from .config import set_data_dir, set_data_file
from .cosmology import TimeConversion
from .fields import add_fields, load_scida
from .plotting import use_style
from .region import Region, load_halo, load_isolated, open_snapshot, shrinking_sphere

from ._release import LATER_RELEASE

# modules intended for a later public release (lyratools/_release.py): imported when present
for _name in ("fuv", "cooling"):
    try:
        globals()[_name] = __import__(f"{__name__}.{_name}", fromlist=[_name])
    except ModuleNotFoundError as _e:
        if _e.name != f"{__name__}.{_name}":
            raise
del _name

from ._version import __version__
