"""
Optional: the lyratools derived fields as scida field recipes, for working on a scida dataset directly
(lyratools itself reads with h5py; see lyratools.fields for the same fields on numpy arrays).

    import lyratools
    ds = lyratools.load_scida("output/snapdir_042")   # scida.load with the header's code units
    lyratools.add_fields(ds)          # merge into ds (call right after load)
    ds.data["PartType0"]["nH"], ds.data["PartType4"]["Age"], ds.data["PartType4"]["Fe_H"]

Fields are lazy: computed only when accessed, chunk by chunk.

Gas (PartType0):  nH [cm^-3], Temperature [K], Pressure_k [K cm^-3], Z_Zsun,
                  CoolingTime [Myr], SelfShieldingFactor, element ratios X_Y,
                  net_true [erg cm^3 s^-1] (FUV runs)
Stars (PartType4): FormationTime [Myr], Age [Myr], SNState (0 none, 1 pending, 2 exploded),
                  DeathTime [Myr], ESN [erg], Z_Zsun, element ratios X_Y
Both: 12logOH (12 + log O/H by number), X_H and X_O ([X/H], [X/O] relative to
      Asplund et al. 2009; mass-fraction ratios over the solar ones, i.e. by number).

Fields stored in the snapshot always win over a lyratools recipe of the same name: LYRA
writes Temperature, and add_fields does not replace it with the derived one.
"""
import glob
import os

import dask.array as da
import numpy as np
from scida.fields import FieldContainer

from . import chemistry, stellar_evolution
from .constants import (BOLTZMANN, GAMMA_MINUS1, HYDROGEN_MASSFRAC, PROTONMASS, ZSOLAR, SN_MIN_MASS)
from .units import magnitude
from .fields import FIELD_REGISTRY, net_true_rate, uses_g0_table  # noqa: F401

fielddefs = FieldContainer(containers=["PartType0", "PartType4"])

RATIOS = [("Fe", "H"), ("O", "Fe"), ("Mg", "Fe"), ("C", "Fe"), ("N", "Fe"), ("Si", "Fe"), ("Mn", "Fe"), ("O", "H")]
RATIOS += [(el, "H") for el in ("He", "C", "N", "Mg", "Si", "Mn")]
RATIOS += [(el, "O") for el in ("C", "N", "Mg", "Si", "Mn", "Fe")]


def _cgs(q, unit):
    return magnitude(q.to(unit)) if hasattr(q, "to") else q


def _hydrogen_fraction(arrs):
    if "GFM_Metals" in arrs:
        return magnitude(arrs["GFM_Metals"])[:, 0]
    return HYDROGEN_MASSFRAC


def _formation_time_myr(arrs, snap):
    key = "StellarFormationTime" if "StellarFormationTime" in arrs else "GFM_StellarFormationTime"   # LYRA or GFM name
    a = magnitude(arrs[key])
    agrid = np.geomspace(1e-4, 1.0, 8000)
    tgrid = snap.cosmology.age(1.0 / agrid - 1.0).to("Myr").value
    return da.where(a > 0, da.map_blocks(np.interp, a, agrid, tgrid, dtype=float), np.nan)


def _t_now_myr(snap):
    return float(snap.cosmology.age(float(snap.header["Redshift"])).to("Myr").value)


# ------------------------------------------------------------------ gas
@fielddefs.register_field("PartType0")
def nH(arrs, snap=None, **kwargs):
    """Hydrogen number density [cm^-3]."""
    rho = _cgs(arrs["Density"], "g/cm**3")
    return _hydrogen_fraction(arrs) * rho / PROTONMASS * snap.ureg("cm**-3")


@fielddefs.register_field("PartType0")
def Temperature(arrs, snap=None, **kwargs):
    """
    Gas temperature [K] from internal energy and electron abundance, with
    mu = 4 / (1 + 3X + 4X x_e) (atomic H and He; molecular gas is not covered).
    Only used when the snapshot does not store Temperature (LYRA does).
    """
    u = _cgs(arrs["InternalEnergy"], "cm**2/s**2")
    X = _hydrogen_fraction(arrs)
    xe = magnitude(arrs["ElectronAbundance"]) if "ElectronAbundance" in arrs else 1.0 + (1 - X) / (4 * X)   # fully ionised H, singly ionised He
    mu = 4.0 / (1.0 + 3.0 * X + 4.0 * X * xe)
    return GAMMA_MINUS1 * u / BOLTZMANN * mu * PROTONMASS * snap.ureg.K


@fielddefs.register_field("PartType0")
def Pressure_k(arrs, snap=None, **kwargs):
    """Thermal pressure divided by Boltzmann's constant [K cm^-3]."""
    rho = _cgs(arrs["Density"], "g/cm**3")
    u = _cgs(arrs["InternalEnergy"], "cm**2/s**2")
    return GAMMA_MINUS1 * rho * u / BOLTZMANN * snap.ureg("K/cm**3")


@fielddefs.register_field("PartType0")
def CoolingTime(arrs, snap=None, **kwargs):
    """Cooling time [Myr] from GFM_CoolingRate; 0 where gas is heated."""
    u = _cgs(arrs["InternalEnergy"], "cm**2/s**2")
    rho = _cgs(arrs["Density"], "g/cm**3")
    lam = magnitude(arrs["GFM_CoolingRate"])
    X = _hydrogen_fraction(arrs)
    return da.map_blocks(chemistry.cooling_time, u, rho, lam, X, dtype=float) * snap.ureg.Myr


@fielddefs.register_field("PartType0")
def SelfShieldingFactor(arrs, snap=None, **kwargs):
    """Rahmati et al. (2013) shielded fraction of the UV background at the snapshot redshift."""
    z = float(snap.header["Redshift"])
    rho = _cgs(arrs["Density"], "g/cm**3")
    nh = _hydrogen_fraction(arrs) * rho / PROTONMASS
    return da.map_blocks(chemistry.self_shielding_factor, nh, z, dtype=float)


# ------------------------------------------------------------------ FUV runs (VIRGODWARF_PE_HEATING)
@fielddefs.register_field("PartType0")
def net_true(arrs, snap=None, **kwargs):
    """Net heating - cooling rate / n_H^2 [erg cm^3 s^-1] including photoelectric heating (FUV runs)."""
    lam = magnitude(arrs["GFM_CoolingRate"])
    pe = magnitude(arrs["FUV_PEHeatingRate"])
    g0t = uses_g0_table(snap)
    bypass = None
    if g0t:
        for name in ("G0TableDebugBypass", "G0TableBypass"):
            try:
                bypass = magnitude(arrs[name])
                break
            except KeyError:
                continue
    out = net_true_rate(lam, pe, g0t, bypass)
    return out * snap.ureg("erg*cm**3/s") if hasattr(snap, "ureg") else out


# ------------------------------------------------------------------ stars
@fielddefs.register_field("PartType4")
def FormationTime(arrs, snap=None, **kwargs):
    """Cosmic time of formation [Myr]; NaN for wind/invalid particles."""
    return _formation_time_myr(arrs, snap) * snap.ureg.Myr


@fielddefs.register_field("PartType4")
def Age(arrs, snap=None, **kwargs):
    """Stellar age at the snapshot time [Myr]."""
    return (_t_now_myr(snap) - _formation_time_myr(arrs, snap)) * snap.ureg.Myr


@fielddefs.register_field("PartType4")
def SNState(arrs, snap=None, **kwargs):
    """LYRA SNTime convention: 0 never a SN, 1 SN pending, 2 exploded."""
    s = magnitude(arrs["SNTime"])
    return da.where(s == -1, 0, da.where(s > 0, 1, 2))


@fielddefs.register_field("PartType4")
def DeathTime(arrs, snap=None, **kwargs):
    """Cosmic time [Myr] at which the star died (formation time + |SNTime|); NaN if alive."""
    s = magnitude(arrs["SNTime"])
    return da.map_blocks(stellar_evolution.death_time, _formation_time_myr(arrs, snap), s, dtype=float) * snap.ureg.Myr


@fielddefs.register_field("PartType4")
def ESN(arrs, snap=None, **kwargs):
    """Energy [erg] injected by each star's SN (0 if none yet; failed SNe 1e49 erg)."""
    m = _cgs(arrs["GFM_InitialMass"], "Msun")
    s = magnitude(arrs["SNTime"])
    return da.map_blocks(stellar_evolution.sn_energies, m, s, dtype=float) * snap.ureg.erg


# ------------------------------------------------------------------ both
def _register_common():
    for container in ["PartType0", "PartType4"]:
        def Z_Zsun(arrs, snap=None, **kwargs):
            return magnitude(arrs["GFM_Metallicity"]) / ZSOLAR
        fielddefs.register_field(container, name="Z_Zsun", description="metallicity in solar units")(Z_Zsun)

        for el1, el2 in RATIOS:
            def ratio(arrs, snap=None, el1=el1, el2=el2, **kwargs):
                met = magnitude(arrs["GFM_Metals"])
                return da.map_blocks(chemistry.abundance_ratio, met, el1, el2, drop_axis=1, dtype=float)
            fielddefs.register_field(container, name=f"{el1}_{el2}", description=f"[{el1}/{el2}]")(ratio)

        def oh12(arrs, snap=None, **kwargs):
            met = magnitude(arrs["GFM_Metals"])
            return da.map_blocks(chemistry.twelve_log_oh_metals, met, drop_axis=1, dtype=float)
        fielddefs.register_field(container, name="12logOH", description="12 + log10(O/H) by number")(oh12)


_register_common()


def _stored_names(ds, container):
    try:
        return set(ds.file[container].keys())
    except Exception:
        cont = ds.data[container]
        return {k for k, r in getattr(cont, "_fieldrecipes", {}).items()
                if "h5path" in (getattr(getattr(r, "func", None), "keywords", None) or {})}


def _first_hdf5(path):
    path = os.fspath(path)
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.hdf5")))
        if not files:
            raise FileNotFoundError(f"no .hdf5 files in {path}")
        return files[0]
    return path


def _is_cosmological(header, config=None):
    """scida's rule (CosmologyMixin.validate): Config/ComovingIntegrationOn = 1, else Time = 1/(1+z)."""
    if config is not None and int(np.ravel(config.get("ComovingIntegrationOn", 0))[0]) == 1:
        return True
    if "Time" in header and "Redshift" in header:
        return bool(np.isclose(float(header["Time"]), 1 / (1 + float(header["Redshift"]))))
    return False


def code_unit_hints(header, config=None):
    """
    scida unit definitions (a `unitfile` dict) for the code units in a snapshot header.

    scida's default unit file for cosmological AREPO runs assumes code_length = ckpc/h; LYRA runs
    use UnitLength_in_cm = 1 Mpc, so with the defaults every length-dependent field is off by
    powers of 1000 (densities by 1e9). These definitions follow the header instead
    (comoving and /h for cosmological runs, as scida decides it).
    """
    ul, um, uv = (float(header[k]) for k in ("UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s"))
    a_h, per_h = (" * a / h", " / h") if _is_cosmological(header, config) else ("", "")
    return {"units": {"code_length": f"{ul!r} * cm{a_h}", "code_mass": f"{um!r} * g{per_h}",
                      "code_velocity": f"{uv!r} * cm / s", "code_time": f"{ul / uv!r} * s"}}


def load_scida(path, catalog=None, units=True, **kwargs):
    """
    scida.load for a LYRA snapshot with the code units taken from its header (see code_unit_hints).

    catalog: groups_XXX path, None to let scida find it, or "none". Extra keyword arguments go to
    scida.load. The catalogue is loaded with the same unit definitions and unit registry as the
    snapshot (scida's own catalogue loading would apply its default unit file to it), then attached
    the way scida attaches it (ds.catalog, merged Group/Subhalo data, catalogue IDs).
    """
    import h5py
    from scida import load
    with h5py.File(_first_hdf5(path), "r") as f:
        header = dict(f["Header"].attrs)
        config = dict(f["Config"].attrs) if "Config" in f else None
    hints = code_unit_hints(header, config)
    ds = load(path, units=units, unitfile=hints, catalog="none", **kwargs)
    if catalog == "none":
        return ds
    ds.catalog = None if catalog is None else os.fspath(catalog)
    if ds.catalog is None:
        ds.discover_catalog()
    if not ds.catalog:
        return ds
    kw = {k: kwargs[k] for k in ("overwrite", "overwrite_cache") if k in kwargs}
    cat = load(ds.catalog, units=units, unitfile=hints, ureg=getattr(ds, "ureg", None),
               fileprefix=ds._get_fileprefix(ds.catalog),              # as in scida's load_catalog
               nonvirtual_datasets=["Group/GroupFirstSub", "Group/GroupLenType", "Group/GroupNsubs",
                                    "Subhalo/SubhaloGrNr", "Subhalo/SubhaloGroupNr", "Subhalo/SubhaloLenType"],
               **kw)
    z_cat, z_snap = cat.header.get("Redshift"), ds.header.get("Redshift")
    if z_cat is not None and z_snap is not None and not np.isclose(z_cat, z_snap):
        raise ValueError(f"redshift mismatch between snapshot and catalogue: {z_snap:.2f} vs {z_cat:.2f}")
    ds.catalog = cat
    ds.merge_data(cat)
    if "Group" in cat.data and len(cat.data["Group"].keys()) > 0:
        ds.add_catalogIDs()
    ds.merge_hints(cat)
    return ds


def check_code_units(ds):
    """Raise ValueError if the scida dataset's code_length disagrees with UnitLength_in_cm of its header."""
    hdr, ureg = getattr(ds, "header", {}), getattr(ds, "ureg", None)
    if ureg is None or "UnitLength_in_cm" not in hdr:
        return
    try:
        got = (1 * ureg.code_length).to("cm").magnitude
    except Exception:                                # no code_length defined: nothing to compare
        return
    want = float(hdr["UnitLength_in_cm"])
    if "cosmology" in getattr(ds, "_mixins", ()):
        want *= float(hdr.get("Time", 1)) / float(hdr.get("HubbleParam", 1))
    if not np.isclose(got, want, rtol=1e-3):
        raise ValueError(
            f"scida's code_length is {got:.4g} cm but the snapshot header gives {want:.4g} cm "
            f"(UnitLength_in_cm = {hdr['UnitLength_in_cm']:.4g}): scida's default unit file assumes ckpc/h. "
            "Load with lyratools.load_scida(path) (header units), or pass "
            "unitfile=lyratools.scida_fields.code_unit_hints(header) and catalog='none' to scida.load.")


def add_fields(ds, overwrite=True):
    """
    Merge the lyratools derived fields into a scida dataset (call before accessing them).
    Fields stored in the snapshot (e.g. LYRA's Temperature) are kept, never replaced by a recipe.
    Raises ValueError if the dataset's code units disagree with its header (see load_scida).
    """
    check_code_units(ds)
    keep = {}
    for c in ("PartType0", "PartType4"):
        if c not in ds.data:
            continue
        cont = ds.data[c]
        stored = _stored_names(ds, c) & set(fielddefs[c]._fieldrecipes)
        keep[c] = {k: (cont._fieldrecipes.get(k), cont._fields.get(k)) for k in stored}
    ds.data.merge(fielddefs, overwrite=overwrite)
    for c, saved in keep.items():
        cont = ds.data[c]
        for k, (rec, fld) in saved.items():
            if rec is not None:
                cont._fieldrecipes[k] = rec
            if fld is not None:
                cont._fields[k] = fld
            elif k in cont._fields:
                del cont._fields[k]
    return ds
