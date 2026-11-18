"""
Load the particles of a halo, or of a whole isolated run, into memory with h5py: physical units,
centred, plain numpy.

    import lyratools as lt
    halo = lt.load_halo("output/snapdir_039", halo_id=0)          # catalogue groups_039 found next to it
    halo = lt.load_halo("output/snapdir_039", 0, groupdir="output/groups_039")
    halo.gas.pos, halo.gas.mass, halo.stars.tform, halo.stars["GFM_Metallicity"]
    halo.gas["Temperature"], halo.gas.values("nH")                 # read or derived on first access, then kept
    inner = halo.within_r200(0.1)                                  # masked view of all particle types
    iso = lt.load_isolated("output/snapdir_102")                   # non-cosmological run, all particles

By default only the FoF group's own particles are read: snapshots are sorted by group, so these are
one contiguous slice of each field. selection="box" instead loads everything in a box around the
halo (including unbound gas and neighbours), which scans the coordinates of all particles.

Standard arrays on every particle set (plain numpy, precision `dtype`, default float32):
    pos [kpc], vel [km/s], mass [Msun], r [kpc], vr [km/s]
Stars additionally get: m_init [Msun], tform [Myr], age [Myr], and, if SNTime exists, t_death [Myr].
p["Name"] is a snapshot field as stored (code units) or a derived field (lyratools.fields);
p.values("Name") is in lyratools units, physical (kpc, km/s, Msun, Myr, g/cm^3, erg/g, K, cm^-3,
Msun/yr, erg cm^3/s), or in `unit` (any astropy unit string, e.g. values("Density", "Msun/pc**3")).
Fields not read up front are read on first access, only for this region's particles, and kept.
"""
import copy
import glob
import os
import re
import warnings

import numpy as np

from . import fields as F
from .stellar_evolution import death_time
from .timeseries import io as tio

DEFAULT_FIELDS = {
    "PartType0": ["Velocities", "Masses", "Density", "InternalEnergy", "ElectronAbundance", "GFM_Metallicity",
                  "StarFormationRate"],
    "PartType1": ["Velocities", "Masses"],
    "PartType4": ["Velocities", "Masses", "GFM_InitialMass", "StellarFormationTime", "GFM_Metallicity", "SNTime"],
}
ALIASES = {"PartType0": "gas", "PartType1": "dm", "PartType4": "stars"}
PARTTYPE = {v: k for k, v in ALIASES.items()}

# units of the standard arrays
STANDARD_UNITS = dict(pos="kpc", vel="km/s", mass="Msun", r="kpc", vr="km/s",
                      tform="Myr", age="Myr", m_init="Msun", t_death="Myr")
# snapshot fields converted by values(): kind of code unit, or a fixed unit for fields stored in it
FIELD_KIND = dict(Coordinates="length", BirthPos="length", CenterOfMass="length",
                  Velocities="velocity", BirthVel="velocity",
                  Masses="mass", GFM_InitialMass="mass",
                  Density="density", BirthDensity="density",
                  InternalEnergy="energy")
KIND_UNIT = dict(length="kpc", velocity="km/s", mass="Msun", density="g/cm**3", energy="erg/g")
STORED_UNITS = dict(StarFormationRate="Msun/yr", Temperature="K", GFM_Metallicity="", GFM_Metals="",
                    ElectronAbundance="", NeutralHydrogenAbundance="",
                    **{k: v["units"] for k, v in F.FIELD_REGISTRY.items() if v["units"] is not None})

# Field names that differ between AREPO/GFM and LYRA snapshots. Either name can be requested; the
# field is read from whichever exists and stored under the canonical name.
FIELD_ALIASES = {
    "GFM_StellarFormationTime": "StellarFormationTime",   # LYRA writes StellarFormationTime
    "StellarFormationTime": "GFM_StellarFormationTime",
    "G0TableDebugBypass": "G0TableBypass",                # the first G0-table version wrote G0TableBypass
    "G0TableBypass": "G0TableDebugBypass",
}
CANONICAL_NAME = {"StellarFormationTime": "GFM_StellarFormationTime"}


def unit_factor(from_unit, to_unit):
    """Factor converting values in `from_unit` to `to_unit` (astropy unit strings)."""
    import astropy.units as u
    return float((1.0 * u.Unit(from_unit)).to(u.Unit(to_unit)).value)


class _Source:
    """Where a Particles set lives in the snapshot: global indices start:stop of particle type pt,
    optionally restricted to positions `sub` (relative to start) after selections."""

    def __init__(self, region, pt, start, stop, sub=None):
        self.region, self.pt, self.n = region, pt, int(pt[-1])
        self.start, self.stop, self.sub = int(start), int(stop), sub

    def restricted(self, idx):
        out = copy.copy(self)
        idx = np.asarray(idx)
        out.sub = idx if self.sub is None else self.sub[idx]
        return out

    def raw_names(self):
        return self.region._raw_names(self.n)

    def resolve(self, name):
        """Name of the stored dataset holding `name` (directly or by alias), or None."""
        raw = self.raw_names()
        if name in raw:
            return name
        alt = FIELD_ALIASES.get(name)
        return alt if alt in raw else None

    def global_index(self):
        if self.sub is None:
            return np.arange(self.start, self.stop, dtype=np.int64)
        return self.start + np.asarray(self.sub, np.int64)

    def membership(self, key):
        """SubhaloID / GroupID of each particle from the catalogue offsets (-1: not a member)."""
        cat = self.region.cat
        goff, soff = cat.offsets()
        if key == "GroupID":
            off, ln = goff[:, self.n], np.asarray(cat["GroupLenType"], np.int64).reshape(-1, 6)[:, self.n]
        else:
            off, ln = soff[:, self.n], np.asarray(cat["SubhaloLenType"], np.int64).reshape(-1, 6)[:, self.n]
        i = self.global_index()
        if len(off) == 0:
            return np.full(len(i), -1, np.int64)
        j = np.searchsorted(off, i, side="right") - 1
        jc = np.clip(j, 0, len(off) - 1)
        return np.where((j >= 0) & (i < off[jc] + ln[jc]), j, -1).astype(np.int64)

    def read(self, stored, cols=None):
        sn = self.region.sn
        if self.sub is None:
            return sn.read_ranges(self.n, stored, [(self.start, self.stop)], cols)
        idx = self.start + np.asarray(self.sub, np.int64)
        if len(idx) > 1 and np.any(np.diff(idx) < 0):
            order = np.argsort(idx, kind="stable")
            vals = sn.read_indices(self.n, stored, idx[order], cols)
            out = np.empty_like(vals)
            out[order] = vals
            return out
        return sn.read_indices(self.n, stored, idx, cols)


class Particles(dict):
    """
    Dictionary of numpy arrays for one particle type, also accessible as attributes.

    Keys that are not in memory are read from the snapshot on first access (stored fields, in code
    units) or computed (derived fields of lyratools.fields), and kept. `name in p` only reports what is
    already in memory; `p.available()` lists everything, `p.has(name)` whether a name can be accessed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = None

    def __reduce__(self):
        # pickles hold the arrays in memory only (not the open snapshot)
        return (Particles, (dict(self),))

    def __copy__(self):
        out = Particles(dict(self))
        out._source = getattr(self, "_source", None)
        return out

    # -- lazy access
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            pass
        val = self._fetch(key)
        dict.__setitem__(self, CANONICAL_NAME.get(key, key), val)
        return val

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(str(e.args[0]) if e.args else key) from None

    def __len__(self):
        return len(dict.__getitem__(self, "pos")) if dict.__contains__(self, "pos") else 0

    def _names(self):
        """(raw, derived) names available for this particle type."""
        src = getattr(self, "_source", None)
        if src is None:
            return [], []
        raw = src.raw_names()
        extra = {"SubhaloID", "GroupID"} if src.region.cat is not None else set()
        return sorted(raw), sorted((set(F.DERIVED.get(src.pt, {})) | extra) - set(raw))

    def available(self):
        """All names that can be accessed: in memory, raw snapshot fields and derived fields."""
        raw, derived = self._names()
        return sorted(set(self.keys()) | set(raw) | set(derived))

    def has(self, name):
        if dict.__contains__(self, name):
            return True
        src = getattr(self, "_source", None)
        return src is not None and (src.resolve(name) is not None or F.derived(src.pt, name) is not None
                                    or (name in ("SubhaloID", "GroupID") and src.region.cat is not None))

    def _missing(self, key, extra=""):
        raw, derived = self._names()
        src = getattr(self, "_source", None)
        where = ALIASES.get(src.pt, src.pt) if src is not None else "this particle set"
        return KeyError(f"'{key}' is not a field of {where}{extra}. In memory: {sorted(self.keys())}; "
                        f"raw (snapshot): {raw}; derived: {derived}")

    def _fetch(self, key):
        src = getattr(self, "_source", None)
        if src is None:
            raise self._missing(key)
        stored = src.resolve(key)
        if stored is not None:
            return src.read(stored)
        d = F.derived(src.pt, key)
        if d is not None:
            return np.asarray(d[0](self, src.region))
        if key in ("SubhaloID", "GroupID") and src.region.cat is not None:
            return src.membership(key)
        raise self._missing(key)

    def column(self, name, j):
        """Column j of a 2D field (e.g. GFM_Metals): from memory if loaded, else only that column is read."""
        if dict.__contains__(self, name):
            return np.asarray(dict.__getitem__(self, name))[:, j]
        src = getattr(self, "_source", None)
        stored = src.resolve(name) if src is not None else None
        if stored is None:
            raise self._missing(name)
        return src.read(stored, cols=[j])[:, 0]

    def load(self, *names):
        """Read several fields now (otherwise each is read on first access); returns self."""
        for n in names:
            self[n]
        return self

    # -- numpy access in lyratools units
    def unit_of(self, name):
        """lyratools unit (string) of values(name); None if unknown (returned as stored)."""
        name = CANONICAL_NAME.get(name, name)
        if name in STANDARD_UNITS:
            return STANDARD_UNITS[name]
        src = getattr(self, "_source", None)
        stored = src.resolve(name) if src is not None else (name if dict.__contains__(self, name) else None)
        if stored is None and src is not None and F.derived(src.pt, name) is not None:
            return F.derived(src.pt, name)[1]
        stored = CANONICAL_NAME.get(stored, stored)
        if stored in FIELD_KIND:
            return KIND_UNIT[FIELD_KIND[stored]]
        if stored == "GFM_StellarFormationTime":
            return None
        return STORED_UNITS.get(stored)

    def values(self, name=None, unit=None):
        """
        Field `name` as a numpy array in lyratools units (physical: kpc, km/s, Msun, Myr, g/cm^3,
        erg/g, K, cm^-3, Msun/yr, erg cm^3/s), or in `unit` (astropy unit string).
        Fields with unknown units are returned as stored.
        Without a name, the dict's values() view (for compatibility with dict code).
        """
        if name is None:
            return dict.values(self)
        arr = np.asarray(self[name])
        u0 = self.unit_of(name)
        src = getattr(self, "_source", None)
        stored = src.resolve(name) if src is not None else None
        stored = CANONICAL_NAME.get(stored, stored)
        if stored in FIELD_KIND and name not in STANDARD_UNITS:
            arr = arr.astype(np.float64) * src.region.code_factor(FIELD_KIND[stored])
        if unit is None or unit == u0:
            return arr
        if u0 is None:
            raise ValueError(f"'{name}' has no known units; cannot convert to {unit}")
        return arr * unit_factor(u0, unit)

    def get_as(self, key, unit=None):
        """A field as a plain numpy array: as stored, or converted to `unit` (see values)."""
        return np.asarray(self[key]) if unit is None else self.values(key, unit)

    # -- selections
    def select(self, mask):
        """New Particles with only the masked entries (boolean mask or integer indices)."""
        n = len(self)
        mask = np.asarray(mask)
        idx = np.flatnonzero(mask) if mask.dtype == bool else mask
        out = Particles()
        for k, v in self.items():
            out[k] = v[mask] if hasattr(v, "__len__") and not np.isscalar(v) and len(v) == n else v
        src = getattr(self, "_source", None)
        out._source = src.restricted(idx) if src is not None else None
        return out

    # -- stars
    def alive(self, m_min=4.0):
        """
        Boolean mask of stars that are still alive and have an initial mass >= m_min [Msun].

        From LYRA's SNTime (sn_state == 1: explosion still to come). Stars with SNTime = -1
        (never a SN) and stars of snapshots without SNTime are judged from metallicity-dependent
        lifetimes (lyratools.yields.lifetime_myr, Raiteri et al. 1996) and the stellar age, with a warning
        when SNTime is missing.
        """
        from . import stellar_evolution as se
        from .yields import lifetime_myr
        m = np.asarray(self["m_init"], np.float64)
        sel = m >= m_min
        age = np.asarray(self["age"], np.float64)
        Z = np.asarray(self["GFM_Metallicity"], np.float64) if self.has("GFM_Metallicity") else None
        if self.has("SNTime"):
            state = se.sn_state(np.asarray(self["SNTime"]))
            alive = state == 1
            none = state == 0
            if none.any():
                alive[none] = age[none] < _lifetime(lifetime_myr, m[none], None if Z is None else Z[none])
        else:
            warnings.warn("no SNTime field: alive() uses metallicity-dependent stellar lifetimes")
            alive = age < _lifetime(lifetime_myr, m, Z)
        return sel & alive


def _lifetime(fn, m, Z):
    return fn(m) if Z is None else fn(m, Z)


def normalize_fields(fields):
    """Accept {'gas': [...], 'stars': [...], 'dm': [...]} as well as PartTypeN keys."""
    if fields is None:
        return DEFAULT_FIELDS
    out = {}
    for k, names in fields.items():
        pt = PARTTYPE.get(k, k)
        out[pt] = [names] if isinstance(names, str) else list(names)
    return out


def _inplace_periodic(x, box):
    """Minimum-image wrap in place; skipped when no separation exceeds half the box (one cheap pass)."""
    if box > 0 and len(x) and (x.max() > 0.5 * box or x.min() < -0.5 * box):
        x += 0.5 * box
        np.mod(x, box, out=x)
        x -= 0.5 * box
    return x


class Region:
    """
    Particles of several types, centred and in physical units: the members of FoF group `fof_halo`,
    all particles in a box of full `width` (code length) around `center_code`, or (neither given) all
    particles of the snapshot.

    Metadata: a, z, h, t_now [Myr], cosmological, center_code (centre used, code units), center (current
    centre in kpc relative to center_code), center_kpc (physical kpc in box coordinates), and for haloes
    from load_halo: halo_id, r200 [kpc], m200 [Msun] (also R200, M200). sn is the open
    lyratools.timeseries.io.Snapshot (cat the Catalogue, if any).
    """
    R200 = None
    M200 = None
    halo_id = None

    def __init__(self, sn, center_code, width=None, fields=None, fof_halo=None, cat=None, dtype=np.float32):
        self.sn, self.cat = sn, cat
        self.header = dict(sn._file(0)["Header"].attrs)
        self.cosmological = sn.cosmological
        self.a, self.z, self.h = sn.a, sn.z, sn.h
        self.time_code = sn.time_code
        self.t_now = sn.t_now
        self.tconv = sn.tc if sn.cosmological else None
        self.kpc_per_code = sn.kpc_per_code
        self.msun_per_code = sn.msun_per_code
        self.center_code = np.asarray(center_code, dtype=np.float64)
        self.dtype = dtype
        self.center = np.zeros(3)          # current centre in kpc relative to center_code
        self.bulk_velocity = np.zeros(3)
        self.config = sn.config()
        self.g0_table = "VIRGODWARF_G0_COOLING_TABLE" in self.config
        self._raw = {}
        if fof_halo is not None:
            goff, _ = cat.offsets()
            glen = np.asarray(cat["GroupLenType"], np.int64).reshape(-1, 6)

        self.parts = {}
        for pt, names in normalize_fields(fields).items():
            n = int(pt[-1])
            if sn.total(n) == 0:
                continue
            rel = None
            if fof_halo is not None:
                start = int(goff[fof_halo, n])
                src = _Source(self, pt, start, start + int(glen[fof_halo, n]))
            elif width is not None:
                idx, rel = sn.scan_box(n, self.center_code, 0.5 * np.asarray(width, np.float64))
                start = int(idx[0]) if len(idx) else 0
                src = _Source(self, pt, start, int(idx[-1]) + 1 if len(idx) else 0, idx - start)
            else:
                src = _Source(self, pt, 0, sn.total(n))
            p = Particles()
            p._source = src
            if rel is None:
                rel = src.read("Coordinates").astype(np.float64)
                rel -= self.center_code
                _inplace_periodic(rel, sn.box)
            rel *= self.kpc_per_code
            p["pos"] = rel.astype(dtype, copy=False)
            del rel
            p["vel"] = self._scaled(src.read("Velocities"), self.code_factor("velocity"))
            if src.resolve("Masses") is not None:
                p["mass"] = self._scaled(src.read("Masses"), self.msun_per_code)
            else:
                p["mass"] = np.full(len(p["pos"]), sn.masstable[n] * self.msun_per_code, dtype)
            if len(p["mass"]) and not np.any(p["mass"] > 0):
                raise ValueError(f"{pt}: all particle masses are zero (MassTable entry is 0 and no 'Masses' dataset)")
            for name in names:
                if name in ("Coordinates", "Velocities", "Masses"):
                    continue
                stored = src.resolve(name)
                if stored is not None:
                    dict.__setitem__(p, CANONICAL_NAME.get(name, name), src.read(stored))
            self.parts[pt] = p
        self._update_radii()

    # -- construction helpers
    def _raw_names(self, n):
        if n not in self._raw:
            self._raw[n] = self.sn.field_names(n)
        return self._raw[n]

    def code_factor(self, kind):
        """Code-to-lyratools factor (physical, at this snapshot) for a FIELD_KIND."""
        sn = self.sn
        return dict(length=sn.kpc_per_code, velocity=sn.velocity_cgs / 1e5 * np.sqrt(sn.a), mass=sn.msun_per_code,
                    density=sn.density_cgs_comoving / sn.a ** 3, energy=sn.velocity_cgs ** 2)[kind]

    def _scaled(self, raw, factor):
        out = raw.astype(self.dtype)          # always a copy
        out *= self.dtype(factor)
        return out

    def _add_stellar_times(self, pt="PartType4", sntime_convention="auto"):
        """tform, age, m_init, t_death of the stars; drops wind particles (formation time <= 0) of cosmological runs."""
        p = self.parts.get(pt)
        if p is None or not p.has("GFM_StellarFormationTime"):
            return
        t_code = np.asarray(p["GFM_StellarFormationTime"], np.float64)
        if self.cosmological:
            valid = t_code > 0                # negative/zero: wind particles or invalid
            if not valid.all():
                p = p.select(valid)
                t_code = t_code[valid]
        p["tform"] = self.sn.t_myr(t_code)
        p["age"] = self.t_now - p["tform"]
        p["m_init"] = (np.asarray(p["GFM_InitialMass"], np.float64) * self.msun_per_code if p.has("GFM_InitialMass")
                       else p["mass"].astype(np.float64))
        if p.has("SNTime"):
            s = np.asarray(p["SNTime"], np.float64)
            conv = "lifetime_gyr"
            if not self.cosmological:
                conv = _sntime_convention(s, t_code, self.time_code) if sntime_convention == "auto" else sntime_convention
            self.sntime_convention = conv
            if conv == "absolute":
                dead = (s < 0) & (s != -1)
                p["t_death"] = np.where(dead, -s * self.sn.myr_per_code_time, np.nan)
            else:
                p["t_death"] = death_time(p["tform"], s)
        self.parts[pt] = p

    def _update_radii(self):
        for p in self.parts.values():
            pos, vel = p["pos"], p["vel"]
            p["r"] = np.sqrt(np.einsum("ij,ij->i", pos, pos)).astype(pos.dtype)
            with np.errstate(invalid="ignore", divide="ignore"):
                p["vr"] = (np.einsum("ij,ij->i", pos, vel) / np.where(p["r"] > 0, p["r"], np.inf)).astype(pos.dtype)

    # -- access
    def __getitem__(self, key):
        return self.parts[PARTTYPE.get(key, key)]

    def __getattr__(self, key):
        for pt, alias in ALIASES.items():
            if key == alias:
                return self.__dict__.get("parts", {}).get(pt, Particles())
        if key == "inert":
            return self.__dict__.get("parts", {}).get("PartType4_inert", Particles())
        raise AttributeError(key)

    @property
    def r200(self):
        """R200c [physical kpc] from the catalogue (None if not a catalogue halo)."""
        return self.R200

    @property
    def m200(self):
        """M200c [Msun] from the catalogue (None if not a catalogue halo)."""
        return self.M200

    @property
    def center_kpc(self):
        """Current centre in physical kpc, in the simulation's coordinate frame."""
        return self.center_code * self.kpc_per_code + self.center

    def all_types(self, keys=("pos", "mass", "vel", "r")):
        """Concatenate the standard arrays over all particle types."""
        return {k: np.concatenate([p[k] for p in self.parts.values() if len(p)]) for k in keys}

    # -- selections
    def select(self, masks):
        """
        A view of the region with only the masked particles: `masks` is {parttype or alias: mask}
        or a function p -> mask applied to every particle type. Metadata is shared; arrays are copies.
        """
        view = object.__new__(type(self))
        view.__dict__.update(self.__dict__)
        if callable(masks):
            view.parts = {pt: p.select(masks(p)) for pt, p in self.parts.items()}
        else:
            masks = {PARTTYPE.get(k, k): v for k, v in masks.items()}
            view.parts = {pt: (p.select(masks[pt]) if pt in masks else p) for pt, p in self.parts.items()}
        return view

    def within(self, r_kpc):
        """View with the particles of every type within r_kpc [physical kpc] of the current centre."""
        return self.select(lambda p: p["r"] < r_kpc)

    def within_r200(self, f=1.0):
        """View with the particles within f x R200 of the current centre."""
        if self.R200 is None:
            raise ValueError("R200 unknown for this region (use load_halo, or within(r_kpc))")
        return self.within(f * self.R200)

    # -- transformations
    def recenter(self, center_kpc=None, velocity_kms=None):
        """Shift positions to a new centre [kpc] and/or remove a velocity [km/s]."""
        for p in self.parts.values():
            if center_kpc is not None:
                p["pos"] -= np.asarray(center_kpc, dtype=p["pos"].dtype)          # in place: no extra copy
            if velocity_kms is not None:
                p["vel"] -= np.asarray(velocity_kms, dtype=p["vel"].dtype)
        if center_kpc is not None:
            self.center = self.center + center_kpc
        if velocity_kms is not None:
            self.bulk_velocity = self.bulk_velocity + velocity_kms
        self._update_radii()

    def remove_bulk_velocity(self, radius_kpc, parttype="PartType4"):
        """Remove the mass-weighted mean velocity of `parttype` within radius_kpc."""
        p = self[parttype]
        sel = p["r"] < radius_kpc
        if sel.sum() == 0:
            return
        w = p["mass"][sel].astype(np.float64)
        self.recenter(velocity_kms=(p["vel"][sel].astype(np.float64) * w[:, None]).sum(0) / w.sum())

    def rotate(self, matrix):
        """Apply a 3x3 rotation matrix to positions and velocities (fields read from the snapshot
        later, such as BirthPos or Velocities, stay in simulation coordinates)."""
        m = np.asarray(matrix)
        for p in self.parts.values():
            p["pos"] = (p["pos"] @ m.T.astype(p["pos"].dtype))
            p["vel"] = (p["vel"] @ m.T.astype(p["vel"].dtype))
        self._update_radii()

    def face_on(self, radius_kpc, parttype="PartType4"):
        """Rotate so the angular momentum of `parttype` within radius_kpc points along z."""
        p = self[parttype]
        sel = p["r"] < radius_kpc
        L = np.sum(np.cross(p["pos"][sel].astype(np.float64), p["vel"][sel]) * p["mass"][sel, None], axis=0)
        zhat = L / np.linalg.norm(L)
        xhat = np.cross([0, 1, 0] if abs(zhat[1]) < 0.9 else [1, 0, 0], zhat)
        xhat /= np.linalg.norm(xhat)
        self.rotate(np.vstack([xhat, np.cross(zhat, xhat), zhat]))

    def close(self):
        """Close the snapshot files (fields not yet in memory can then no longer be read)."""
        self.sn.close()


def shrinking_sphere(pos, mass, center=(0, 0, 0), maxrad=None, minpart=1000, shrink=0.975,
                     max_particles=2_000_000, seed=42):
    """
    Centre of mass found by iteratively shrinking a sphere [same units as pos] until fewer than
    `minpart` particles remain inside (Power et al. 2003).

    Memory-light for very large particle sets: distances are computed one coordinate at a time,
    only particles inside the starting sphere are kept (a random subsample of at most
    `max_particles` of them, which leaves the centre unaffected within Poisson noise), and each
    iteration works on the particles of the previous sphere only.
    """
    center = np.asarray(center, dtype=np.float64)
    pos = np.asarray(pos)
    mass = np.asarray(mass)

    def r2_of(p, c):
        r2 = np.zeros(len(p), dtype=np.float64)
        for k in range(3):
            r2 += (p[:, k].astype(np.float64) - c[k]) ** 2
        return r2

    r2 = r2_of(pos, center)
    rmax = np.sqrt(r2.max()) if len(r2) else 0.0
    radius = rmax if (maxrad is None or not np.isfinite(maxrad) or maxrad <= 0) else min(maxrad, rmax)
    sel = np.flatnonzero(r2 < radius ** 2)
    del r2
    if len(sel) == 0:
        return center
    subsampled = max_particles is not None and len(sel) > max_particles
    if subsampled:
        sel = np.random.default_rng(seed).choice(sel, max_particles, replace=False)

    def shrink_loop(P, M, center, radius):
        while True:
            inside = r2_of(P, center) < radius ** 2
            if int(inside.sum()) <= minpart or M[inside].sum() <= 0:
                break
            P, M = P[inside], M[inside]
            center = (P * M[:, None]).sum(0) / M.sum()
            radius *= shrink
        return center, radius

    center, radius = shrink_loop(pos[sel].astype(np.float64), mass[sel].astype(np.float64), center, radius)
    if subsampled:
        # refine on the full particle set near the centre: the subsample's last sphere is wider
        # than the full set's would be, so continue shrinking with all particles inside 4x its radius
        near = np.flatnonzero(r2_of(pos, center) < (4 * radius) ** 2)
        if len(near) > minpart:
            center, radius = shrink_loop(pos[near].astype(np.float64), mass[near].astype(np.float64), center, 4 * radius)
    return center


# ------------------------------------------------------------------ opening snapshots
def _files_of(path):
    """Snapshot or catalogue files of one output: a directory of chunks, one chunk file (-> all its
    siblings), or a single file."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.hdf5")), key=tio._chunk_number)
        if not files:
            raise FileNotFoundError(f"no *.hdf5 files in {path}")
        return files
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    m = re.fullmatch(r"(.*)\.(\d+)\.hdf5", os.path.basename(path))
    if m:
        return sorted(glob.glob(os.path.join(os.path.dirname(path), f"{glob.escape(m.group(1))}.*.hdf5")), key=tio._chunk_number)
    return [path]


def _guess_groupdir(snap):
    """groups_XXX next to snapdir_XXX, or fof_subhalo_tab_XXX.hdf5 next to snap_XXX.hdf5; None if absent."""
    snap = os.path.abspath(os.path.expanduser(str(snap)))
    if os.path.isdir(snap):
        m, parent = re.search(r"_(\d+)$", os.path.basename(snap)), os.path.dirname(snap)
    else:
        d = os.path.dirname(snap)
        m = re.search(r"_(\d+)(?:\.\d+)?\.hdf5$", os.path.basename(snap))
        parent = os.path.dirname(d) if os.path.basename(d).startswith("snapdir_") else d
    if not m:
        return None
    num = m.group(1)
    for cand in (f"groups_{num}", f"fof_subhalo_tab_{num}.hdf5", f"fof_tab_{num}.hdf5"):
        if os.path.exists(os.path.join(parent, cand)):
            return os.path.join(parent, cand)
    return None


def open_snapshot(snap, groupdir=None, cosmological="auto", need_catalogue=False):
    """
    (Snapshot, Catalogue or None) for `snap`: a snapdir_XXX directory, a snapshot file (any chunk) or a
    scida dataset (its paths are used; nothing is read through scida). The catalogue is `groupdir`
    (groups_XXX directory or file), else the one next to the snapshot, else the scida dataset's.
    """
    if not isinstance(snap, (str, os.PathLike)):          # scida dataset: reuse its paths
        ds = snap
        if groupdir is None and getattr(ds, "catalog", None) is not None:
            groupdir = getattr(ds.catalog, "path", None) or getattr(ds.catalog, "location", None)
        snap = getattr(ds, "path", None) or getattr(ds, "location", None)
        if snap is None:
            raise TypeError("pass a snapshot path (directory or file) or a scida dataset")
    sn = tio.Snapshot(_files_of(snap), cosmological=cosmological)
    if groupdir is None:
        groupdir = _guess_groupdir(snap)
    cat = None
    if groupdir is not None:
        cat = tio.Catalogue(_files_of(groupdir), units=sn.units, cosmological=sn.cosmological)
    elif need_catalogue:
        raise FileNotFoundError(f"no group catalogue found for {snap}; pass groupdir=")
    return sn, cat


def load_halo(snap, halo_id=0, fields=None, center="group", rest_frame=None, selection="fof", size_r200=2.5,
              dtype=np.float32, groupdir=None):
    """
    Load FoF group `halo_id` as AREPO saved it: no recentring computations, no velocity changes.

    snap:       snapdir_XXX directory, snapshot file, or a scida dataset (only its paths are used).
    groupdir:   groups_XXX directory or catalogue file (default: next to the snapshot).
    Positions are relative to the group-catalogue centre, velocities are as stored (peculiar km/s),
    masses in Msun. Adds attributes M200 [Msun], R200 [kpc] and halo_id.

    center:     "group" (default): the catalogue's GroupPos;
                "subhalo": SubhaloPos of the group's central subhalo (GroupFirstSub);
                or explicit coordinates [code units].
    rest_frame: None (default): velocities as stored;
                "group" / "subhalo": subtract the catalogue's GroupVel / SubhaloVel;
                "particles": subtract the mass-weighted mean velocity of all loaded particles.
    selection:  "fof" (default) reads only the group's own particles; "box" reads everything in a
                box of size_r200 x R200 around the centre (includes unbound gas; scans all coordinates).
    fields:     None (default: DEFAULT_FIELDS) or {"gas": [...], "stars": [...], "dm": [...]} (PartTypeN keys
                also accepted) to choose what is read up front; Coordinates, Velocities and Masses are always
                read. Types not listed are not loaded. Any other field is read on first access (halo.gas["X"]).
    dtype:      precision of pos, vel, mass, r, vr (float32 default; np.float64 doubles the memory).
    A shrinking-sphere centre, if wanted, is available separately:
        c = shrinking_sphere(halo.dm.pos, halo.dm.mass, maxrad=...); halo.recenter(center_kpc=c)
    """
    sn, cat = open_snapshot(snap, groupdir, need_catalogue=True)
    if isinstance(center, str):
        if center == "group":
            center_code = np.asarray(cat["GroupPos"][halo_id], np.float64)
        elif center == "subhalo":
            first = int(cat["GroupFirstSub"][halo_id])
            if first < 0:
                raise ValueError(f"group {halo_id} has no subhaloes; use center='group'")
            center_code = np.asarray(cat["SubhaloPos"][first], np.float64)
        else:
            raise ValueError("center must be 'group', 'subhalo' or coordinates in code units")
    else:
        center_code = np.asarray(center, dtype=np.float64)

    fields = {k: list(v) for k, v in normalize_fields(fields).items()}
    if "PartType4" in fields:      # needed for tform, age, m_init
        fields["PartType4"] = list(dict.fromkeys(fields["PartType4"] + ["GFM_InitialMass", "StellarFormationTime"]))
    r200_code = float(cat["Group_R_Crit200"][halo_id])
    if selection == "fof":
        reg = Region(sn, center_code, fields=fields, fof_halo=halo_id, cat=cat, dtype=dtype)
    elif selection == "box":
        reg = Region(sn, center_code, size_r200 * r200_code, fields=fields, cat=cat, dtype=dtype)
    else:
        raise ValueError("selection must be 'fof' or 'box'")
    reg._add_stellar_times()
    reg.R200 = r200_code * cat.kpc_per_code
    reg.M200 = float(cat["Group_M_Crit200"][halo_id]) * cat.msun_per_code
    reg.halo_id = halo_id

    vunit = sn.velocity_cgs / 1e5
    if rest_frame is None:
        pass
    elif rest_frame == "group":            # GroupVel is stored as peculiar velocity x a
        reg.recenter(velocity_kms=np.asarray(cat["GroupVel"][halo_id], np.float64) * vunit / sn.a)
    elif rest_frame == "subhalo":          # SubhaloVel: peculiar
        first = int(cat["GroupFirstSub"][halo_id])
        reg.recenter(velocity_kms=np.asarray(cat["SubhaloVel"][first], np.float64) * vunit)
    elif rest_frame == "particles":
        msum, mv = 0.0, np.zeros(3)
        for p in reg.parts.values():
            w = p["mass"].astype(np.float64)
            msum += w.sum()
            mv += (p["vel"].astype(np.float64) * w[:, None]).sum(0)
        reg.recenter(velocity_kms=mv / msum)
    else:
        raise ValueError("rest_frame must be None, 'group', 'subhalo' or 'particles'")
    return reg


# ------------------------------------------------------------------ isolated (non-cosmological) runs
def _sntime_convention(sntime, aform_code, time_code):
    """See lyratools.stellar_evolution.sntime_convention."""
    from .stellar_evolution import sntime_convention
    return sntime_convention(sntime, aform_code, time_code)


def load_isolated(path, fields=None, center="inert", h=None, dtype=np.float32, sntime_convention="auto"):
    """
    Load all particles of an isolated (non-cosmological) LYRA/AREPO snapshot, converted with the header's
    code units:

        halo = lt.load_isolated("output/snapdir_102")
        halo.stars.tform, halo.stars.age, halo.t_now     # Myr, from code time (UnitLength / UnitVelocity)
        halo.gas.values("Density")                       # g/cm^3; also nH, Temperature on request
        halo.inert                                       # initial-condition stars (GFM_InitialMass = 0)

    Positions are centred on the median position of the inert stars (or `center` in code units); those stars
    are excluded from halo.stars. The header's HubbleParam is ignored (h = 1) unless h is given, as in the
    time-series code. sntime_convention: 'auto' (default) detects whether SNTime holds absolute code times of
    death (the isolated FUV runs) or lifetimes in Gyr (cosmological convention); used for stars.t_death.
    """
    sn = tio.Snapshot(_files_of(path), cosmological=False)
    if h is not None:
        sn.h = float(h)
        sn._set_units(sn.units)
    if isinstance(center, str):
        if center != "inert":
            raise ValueError("center must be 'inert' or coordinates in code units")
        mi = sn.read_ranges(4, "GFM_InitialMass", [(0, sn.total(4))]) if sn.has(4, "GFM_InitialMass") else np.zeros(0)
        if not np.any(mi == 0):
            raise ValueError("no inert stars (GFM_InitialMass = 0) to centre on; pass center=coordinates")
        inert = np.flatnonzero(mi == 0)
        center = np.median(sn.read_indices(4, "Coordinates", inert), axis=0)
    fields = {k: list(v) for k, v in normalize_fields(fields).items()}
    if "PartType4" in fields:
        fields["PartType4"] = list(dict.fromkeys(fields["PartType4"] + ["GFM_InitialMass", "StellarFormationTime"]))
    reg = Region(sn, center, fields=fields, dtype=dtype)
    reg.myr_per_code = sn.myr_per_code_time
    p = reg.parts.get("PartType4")
    if p is not None:
        mi = np.asarray(p["GFM_InitialMass"], np.float64)
        reg.parts["PartType4_inert"] = p.select(mi == 0)
        reg.parts["PartType4"] = p.select(mi > 0)
        reg._add_stellar_times(sntime_convention=sntime_convention)
    return reg
