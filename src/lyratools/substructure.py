"""Satellites and star clusters from the SUBFIND catalog, and ex-situ stars from birth positions."""
import numpy as np

from .units import as_array


def luminous_subhalos(ds, fof=0, min_n_particles=32, min_n_stars=1, min_half_rad_kpc=0.01,
                      min_stellar_half_rad_kpc=0.004, r200_fraction=1.0):
    """
    Boolean mask over all subhalos: resolved, luminous subhalos of FoF group `fof` within
    r200_fraction x R200 of its centre (periodic distances). Radii thresholds in physical kpc.
    ds: a Region from load_halo (its catalogue is used), a lyratools.timeseries.io.Catalogue, or a scida dataset.
    """
    cat = _catalogue(ds)
    if cat is not None:
        kpc = cat.kpc_per_code
        d = np.asarray(cat["SubhaloPos"], float) - np.asarray(cat["GroupPos"][fof], float)
        if cat.box > 0:
            d = (d + cat.box / 2) % cat.box - cat.box / 2
        dist = np.linalg.norm(d, axis=1) * kpc
        r200 = float(cat["Group_R_Crit200"][fof]) * kpc
        lentype = np.asarray(cat["SubhaloLenType"]).reshape(-1, 6)
        return ((np.asarray(cat["SubhaloGrNr"]) == fof) & (dist < r200_fraction * r200) & (lentype[:, 4] > min_n_stars)
                & (np.asarray(cat["SubhaloLen"]) > min_n_particles)
                & (np.asarray(cat["SubhaloHalfmassRad"]) * kpc > min_half_rad_kpc)
                & (np.asarray(cat["SubhaloHalfmassRadType"])[:, 4] * kpc > min_stellar_half_rad_kpc))
    sub, grp = ds.data["Subhalo"], ds.data["Group"]
    box = float(np.atleast_1d(as_array(ds.boxsize))[0])
    grnr = as_array(sub["SubhaloGrNr"])
    d = as_array(sub["SubhaloPos"]) - as_array(grp["GroupPos"][fof])
    d = (d + box / 2) % box - box / 2
    kpc = float(ds.ureg("code_length").to("kpc").magnitude)
    dist = np.linalg.norm(d, axis=1) * kpc
    r200 = float(as_array(grp["Group_R_Crit200"][fof], "kpc"))
    lentype = as_array(sub["SubhaloLenType"])
    hmr = as_array(sub["SubhaloHalfmassRad"], "kpc")
    hmr_star = as_array(sub["SubhaloHalfmassRadType"], "kpc")[:, 4]
    return ((grnr == fof) & (dist < r200_fraction * r200) & (lentype[:, 4] > min_n_stars)
            & (as_array(sub["SubhaloLen"]) > min_n_particles)
            & (hmr > min_half_rad_kpc) & (hmr_star > min_stellar_half_rad_kpc))


def per_subhalo(subhalo_id, values, n_subhalos, weights=None, stat="sum"):
    """
    Per-subhalo statistic of particle values using each particle's SubhaloID
    (unbound particles must have IDs outside 0..n_subhalos-1).
    stat: 'sum', 'mean' or 'std' (weighted if weights given).
    """
    sid = np.asarray(subhalo_id)
    ok = np.isfinite(sid) & (sid >= 0) & (sid < n_subhalos)
    v = np.asarray(values, float)[ok]
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)[ok]
    sid = sid[ok].astype(np.int64)
    if stat == "sum":
        return np.bincount(sid, weights=v * (w if weights is not None else 1), minlength=n_subhalos)
    sw = np.bincount(sid, weights=w, minlength=n_subhalos)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(sid, weights=w * v, minlength=n_subhalos) / sw
        if stat == "mean":
            return mean
        var = np.bincount(sid, weights=w * (v - mean[sid]) ** 2, minlength=n_subhalos) / sw
        return np.sqrt(var)


def birth_radius(birth_pos_ckpc, birth_time_myr, halo_time_myr, halo_pos_ckpc):
    """
    Comoving distance [ckpc] of each star's birth position from the main halo centre,
    interpolated along the halo track (times in Myr, positions in comoving kpc).
    """
    c = np.column_stack([np.interp(birth_time_myr, halo_time_myr, halo_pos_ckpc[:, k]) for k in range(3)])
    return np.linalg.norm(np.asarray(birth_pos_ckpc) - c, axis=1)


def exsitu_stars(birth_pos_ckpc, birth_time_myr, halo_time_myr, halo_pos_ckpc, halo_rvir_ckpc, return_radius=False):
    """True for stars born outside the (interpolated) virial radius of the main halo."""
    brad = birth_radius(birth_pos_ckpc, birth_time_myr, halo_time_myr, halo_pos_ckpc)
    rvir = np.interp(birth_time_myr, halo_time_myr, halo_rvir_ckpc)
    return (brad > rvir, brad) if return_radius else brad > rvir


def _catalogue(obj):
    from .timeseries.io import Catalogue
    if isinstance(obj, Catalogue):
        return obj
    return getattr(obj, "cat", None) if hasattr(obj, "sn") else None


def birth_positions_ckpc(ds, birth_pos):
    """Convert BirthPos (code length, comoving) to comoving kpc without h. ds: a Region (e.g. from load_halo,
    with birth_pos = halo.stars["BirthPos"]) or a scida dataset."""
    if hasattr(ds, "sn"):
        return np.asarray(birth_pos, np.float64) * ds.sn.ckpc_per_code
    h = float(ds.header["HubbleParam"])
    a = float(ds.header["Time"])
    kpc_phys = float(ds.ureg("code_length").to("kpc").magnitude)   # physical kpc at this snapshot
    return as_array(birth_pos) * kpc_phys / a                       # comoving kpc
