"""
Reading AREPO group catalogues and snapshot particles with h5py.

Only the needed index ranges, fields and GFM_Metals columns are read: a FoF/subhalo slice is a
contiguous range, and a spherical selection scans the coordinates once and then reads the other
fields only where selected particles are. lyratools.region (load_halo, load_isolated) uses the same reader.

Units follow AREPO's cosmological conventions, with unit factors from the file headers:
    Coordinates, BirthPos, SubhaloPos, GroupPos, radii: comoving code length (/h)
    Velocities: x sqrt(a) (peculiar = stored * sqrt(a)); SubhaloVel: peculiar km/s; GroupVel: x a
    masses: code mass / h; Density, BirthDensity: comoving code density (h^2 / a^3)
Non-cosmological (isolated) runs have a = 1, z = 0 and h = 1: code units are physical, Time and
formation times are code times (UnitLength / UnitVelocity), and there is no Hubble flow.
"""
import glob
import os
import re

import h5py
import numpy as np

from ..constants import KPC, MSUN

DEFAULT_UNITS = dict(UnitLength_in_cm=3.085678e24, UnitMass_in_g=1.989e43, UnitVelocity_in_cm_per_s=1e5)
FORMATION_TIME = ("StellarFormationTime", "GFM_StellarFormationTime")    # LYRA name first


# ------------------------------------------------------------------ files
def _chunk_number(path):
    m = re.search(r"\.(\d+)\.hdf5$", path)
    return int(m.group(1)) if m else -1


def _numbered_files(base, dir_prefix, file_prefixes):
    """
    {number: [files]} for <dir_prefix>_XXX/*.hdf5 (chunks, whatever their names) or
    <file_prefix>_XXX.hdf5 single files directly in base. base may also be one <dir_prefix>_XXX itself.
    """
    out = {}
    base = os.path.abspath(os.path.expanduser(base))
    if re.fullmatch(rf"{dir_prefix}_(\d+)", os.path.basename(base)):
        base = os.path.dirname(base)
    for d in glob.glob(os.path.join(base, f"{dir_prefix}_*")):
        m = re.fullmatch(rf"{dir_prefix}_(\d+)", os.path.basename(d))
        if m and os.path.isdir(d):
            files = sorted(glob.glob(os.path.join(d, "*.hdf5")), key=_chunk_number)
            if files:
                out[int(m.group(1))] = files
    for f in glob.glob(os.path.join(base, "*.hdf5")):
        m = re.fullmatch(rf"(?:{'|'.join(file_prefixes)})_(\d+)\.hdf5", os.path.basename(f))
        if m:
            out.setdefault(int(m.group(1)), [f])
    return out


def describe_dir(base, found):
    """One line for error messages: what a directory holds and which numbers were recognised."""
    try:
        names = sorted(os.listdir(os.path.expanduser(base)))
    except OSError as e:
        return f"{base}: {e.strerror}"
    shown = ", ".join(names[:6]) + (", ..." if len(names) > 6 else "")
    nums = sorted(found)
    rec = f"{len(nums)} recognised ({nums[0]}..{nums[-1]})" if nums else "none recognised"
    return f"{base}: {len(names)} entries [{shown}], {rec}"


def snapshot_files(snapdir):
    """{snap: [files]} of the snapshots in snapdir (snapdir_XXX/*.hdf5, or snap_XXX.hdf5 / snapshot_XXX.hdf5)."""
    return _numbered_files(snapdir, "snapdir", ("snap", "snapshot"))


def catalogue_files(groupdir):
    """{snap: [files]} of the group catalogues in groupdir (groups_XXX/*.hdf5, or fof_subhalo_tab_XXX.hdf5)."""
    return _numbered_files(groupdir, "groups", ("fof_subhalo_tab", "fof_tab", "groups"))


def _read_units(files):
    with h5py.File(files[0], "r") as f:
        for grp in ("Header", "Parameters"):
            if grp in f and all(k in f[grp].attrs for k in DEFAULT_UNITS):
                return {k: float(f[grp].attrs[k]) for k in DEFAULT_UNITS}
    return None


def periodic(d, box):
    """Minimum-image separation vectors (same units as box); unchanged if box <= 0."""
    if not box > 0:
        return d
    return (d + 0.5 * box) % box - 0.5 * box


def detect_cosmological(header, params=None):
    """(cosmological, reason) from Parameters/ComovingIntegrationOn, else from the header."""
    if params is not None and "ComovingIntegrationOn" in params:
        return bool(int(params["ComovingIntegrationOn"])), "Parameters/ComovingIntegrationOn"
    t, z, om = float(header["Time"]), float(header.get("Redshift", 0.0)), float(header.get("Omega0", 0.0))
    if t > 1.0 + 1e-6:
        return False, f"Time = {t:g} > 1 is not a scale factor"
    if z > 1e-8:
        return True, f"Redshift = {z:g}"
    return om > 0, f"Redshift = 0 and Omega0 = {om:g}"


def resolve_cosmological(setting, files):
    """setting 'auto', True or False -> (bool, reason), reading the first file when 'auto'."""
    if setting in (True, False):
        return bool(setting), "set in the configuration"
    with h5py.File(files[0], "r") as f:
        return detect_cosmological(dict(f["Header"].attrs), dict(f["Parameters"].attrs) if "Parameters" in f else None)


class _Units:
    """Code-to-physical factors at scale factor a (lengths: physical kpc; ckpc: comoving kpc)."""

    def _set_header(self, hd, cosmological, files):
        """a, z, h, box and the run type; for isolated runs a = 1, z = 0, h = 1 and time_code = Time."""
        if cosmological is None or cosmological == "auto":
            cosmological = resolve_cosmological("auto", files)[0]
        self.cosmological = bool(cosmological)
        self.box = float(np.atleast_1d(hd.get("BoxSize", 0.0))[0])
        self.time_code = float(hd["Time"])
        if self.cosmological:
            self.a, self.z, self.h = self.time_code, float(hd["Redshift"]), float(hd["HubbleParam"])
        else:
            self.a, self.z, self.h = 1.0, 0.0, 1.0

    def _set_units(self, units):
        self.units = units
        UL, UM, UV = units["UnitLength_in_cm"], units["UnitMass_in_g"], units["UnitVelocity_in_cm_per_s"]
        self.ckpc_per_code = UL / KPC / self.h
        self.kpc_per_code = self.ckpc_per_code * self.a
        self.msun_per_code = UM / MSUN / self.h
        self.velocity_cgs = UV
        self.density_cgs_comoving = UM * self.h ** 2 / UL ** 3     # times a^-3 for physical
        self.myr_per_code_time = UL / UV / 3.15576e13                # isolated runs (Julian Myr)


# ------------------------------------------------------------------ catalogue
class Catalogue(_Units):
    """
    A SUBFIND group catalogue, read lazily field by field (raw code values, concatenated over files).

        cat = Catalogue(files)
        cat["SubhaloPos"], cat.nsubs, cat.offsets()
    """

    def __init__(self, files, units=None, cosmological=None):
        self.files = list(files)
        with h5py.File(self.files[0], "r") as f:
            hd = dict(f["Header"].attrs)
        self._set_header(hd, cosmological, self.files)
        self.ngroups = int(hd.get("Ngroups_Total", 0))
        self.nsubs = int(hd.get("Nsubgroups_Total", hd.get("Nsubhalos_Total", 0)))
        self.units_from_file = _read_units(self.files)
        self._set_units(self.units_from_file or units or DEFAULT_UNITS)
        self._cache = {}

    def has(self, name):
        grp = "Group" if name.startswith("Group") else "Subhalo"
        for fn in self.files:
            with h5py.File(fn, "r") as f:
                if grp in f and name in f[grp]:
                    return True
        return False

    def __getitem__(self, name):
        if name not in self._cache:
            grp = "Group" if name.startswith("Group") else "Subhalo"
            parts = []
            for fn in self.files:
                with h5py.File(fn, "r") as f:
                    if grp in f and name in f[grp]:
                        parts.append(f[grp][name][()])
            n = self.ngroups if grp == "Group" else self.nsubs
            if not parts:
                if n == 0:
                    return np.zeros(0)
                raise KeyError(f"{name} not in catalogue {self.files[0]}")
            self._cache[name] = np.concatenate(parts)
        return self._cache[name]

    def offsets(self):
        """(group offsets (Ngroups, 6), subhalo offsets (Nsubs, 6)): first particle index of each type."""
        if "_offsets" not in self._cache:
            glt = np.asarray(self["GroupLenType"], np.int64).reshape(-1, 6)
            goff = np.zeros_like(glt)
            goff[1:] = np.cumsum(glt, axis=0)[:-1]
            if self.nsubs:
                slt = np.asarray(self["SubhaloLenType"], np.int64)
                cs = np.zeros_like(slt)
                cs[1:] = np.cumsum(slt, axis=0)[:-1]
                grnr = np.asarray(self["SubhaloGrNr"], np.int64)
                first = np.asarray(self["GroupFirstSub"], np.int64)[grnr]
                soff = goff[grnr] + cs - cs[first]          # subhaloes of a group are contiguous from GroupFirstSub
            else:
                soff = np.zeros((0, 6), np.int64)
            self._cache["_offsets"] = (goff, soff)
        return self._cache["_offsets"]

    def subhalo_distance(self, center_code):
        """Periodic distance [code length] of every SubhaloPos from center_code."""
        d = periodic(np.asarray(self["SubhaloPos"], np.float64) - center_code, self.box)
        return np.sqrt(np.einsum("ij,ij->i", d, d))


# ------------------------------------------------------------------ snapshot
class Snapshot(_Units):
    """
    A (multi-file) snapshot. Particles are addressed by global index per type.

        with Snapshot(files) as sn:
            idx, rel = sn.scan_sphere(0, center_code, radius_code)
            rho = sn.read_indices(0, "Density", idx)
            ids = sn.read_ranges(1, "ParticleIDs", [(start, stop)])
    """

    def __init__(self, files, cosmological=None, scan_chunk=1 << 22, max_gap=4096):
        self.files = list(files)
        self._h = {}
        hd = dict(self._file(0)["Header"].attrs)
        self._set_header(hd, cosmological, self.files)
        self.omega0 = float(hd.get("Omega0", 0.307))
        self.omega_b = float(hd.get("OmegaBaryon", 0.048))
        self.masstable = np.asarray(hd.get("MassTable", np.zeros(6)), np.float64)
        self._set_units(_read_units(self.files) or DEFAULT_UNITS)
        self.counts = np.array([np.asarray(self._file(i)["Header"].attrs["NumPart_ThisFile"], np.int64)
                                for i in range(len(self.files))]).reshape(len(self.files), -1)[:, :6]
        self.first = np.zeros_like(self.counts)
        self.first[1:] = np.cumsum(self.counts, axis=0)[:-1]
        self.scan_chunk, self.max_gap = scan_chunk, max_gap
        self._tc = None

    # -- clock: times in Myr since the Big Bang (cosmological) or since t = 0 (isolated)
    @property
    def tc(self):
        if self._tc is None:
            from ..cosmology import TimeConversion
            self._tc = TimeConversion.from_params(h=self.h, Om0=self.omega0, Ob0=self.omega_b)
        return self._tc

    def t_myr(self, time_code):
        """Myr of a header or formation time (scale factor if cosmological, code time otherwise)."""
        x = np.asarray(time_code, np.float64)
        return self.tc.t_at_a(x) if self.cosmological else x * self.myr_per_code_time

    @property
    def t_now(self):
        return float(self.t_myr(self.time_code))

    @property
    def hubble_kms_kpc(self):
        """H(z) [km/s/kpc]; 0 for isolated runs."""
        return float(self.tc.cosmology.H(self.z).to("km/(s kpc)").value) if self.cosmological else 0.0

    def formation_scale_factor(self, time_code):
        """Scale factor at formation (1 for isolated runs), for comoving birth quantities."""
        return np.asarray(time_code, np.float64) if self.cosmological else np.ones(np.shape(time_code))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for f in self._h.values():
            f.close()
        self._h = {}

    def _file(self, i):
        if i not in self._h:
            self._h[i] = h5py.File(self.files[i], "r")
        return self._h[i]

    def total(self, pt):
        return int(self.counts[:, pt].sum())

    def _dataset_info(self, pt, field):
        """(dtype, trailing shape) of a field, or None if it does not exist."""
        for i in range(len(self.files)):
            g = self._file(i).get(f"PartType{pt}")
            if g is not None and field in g:
                return g[field].dtype, g[field].shape[1:]
        return None

    def has(self, pt, field):
        return self._dataset_info(pt, field) is not None

    def field_names(self, pt):
        """Names of the datasets stored for PartType{pt} (from the first file that has the type)."""
        for i in range(len(self.files)):
            g = self._file(i).get(f"PartType{pt}")
            if g is not None:
                return sorted(g.keys())
        return []

    def config(self):
        """Compile-time options from the Config group ({} if absent)."""
        f = self._file(0)
        return {k: v for k, v in f["Config"].attrs.items()} if "Config" in f else {}

    def scan_box(self, pt, center_code, half_width_code):
        """As scan_sphere, for the periodic cube |x_i - c_i| < half_width_code."""
        center_code = np.asarray(center_code, np.float64)
        hw = np.broadcast_to(np.asarray(half_width_code, np.float64), (3,))
        idx, rel = [], []
        for i in range(len(self.files)):
            n = int(self.counts[i, pt])
            if n == 0:
                continue
            d = self._file(i)[f"PartType{pt}"]["Coordinates"]
            for a in range(0, n, self.scan_chunk):
                b = min(n, a + self.scan_chunk)
                x = periodic(d[a:b].astype(np.float64) - center_code, self.box)
                sel = np.flatnonzero(np.all(np.abs(x) < hw, axis=1))
                idx.append(sel + a + self.first[i, pt])
                rel.append(x[sel])
        if not idx:
            return np.zeros(0, np.int64), np.zeros((0, 3))
        return np.concatenate(idx).astype(np.int64), np.concatenate(rel)

    def field_name(self, pt, names):
        """First of several alternative field names that exists (e.g. formation-time names)."""
        for n in names:
            if self.has(pt, n):
                return n
        return None

    def read_ranges(self, pt, field, ranges, cols=None):
        """Values of `field` for sorted global index ranges [(start, stop), ...], concatenated in order."""
        info = self._dataset_info(pt, field)
        if info is None:
            raise KeyError(f"PartType{pt}/{field} not in {self.files[0]}")
        dtype, tail = info
        if cols is not None and len(tail):
            tail = (len(cols),)
        merged = []
        for a, b in ranges:
            if b <= a:
                continue
            if merged and a == merged[-1][1]:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        out = []
        for a, b in merged:
            for i in range(len(self.files)):
                lo, hi = max(a, self.first[i, pt]), min(b, self.first[i, pt] + self.counts[i, pt])
                if hi <= lo:
                    continue
                d = self._file(i)[f"PartType{pt}"][field]
                sl = slice(int(lo - self.first[i, pt]), int(hi - self.first[i, pt]))
                out.append(d[sl] if (cols is None or d.ndim == 1) else d[sl, list(cols)])
        return np.concatenate(out) if out else np.zeros((0,) + tuple(tail), dtype)

    def read_indices(self, pt, field, idx, cols=None):
        """Values of `field` at sorted global indices; nearby indices are read as one block."""
        idx = np.asarray(idx, np.int64)
        if len(idx) == 0:
            return self.read_ranges(pt, field, [], cols)
        breaks = np.flatnonzero(np.diff(idx) > self.max_gap) + 1
        out = []
        for run in np.split(idx, breaks):
            block = self.read_ranges(pt, field, [(int(run[0]), int(run[-1]) + 1)], cols)
            out.append(block[run - run[0]])
        return np.concatenate(out)

    def masses(self, pt, idx=None, ranges=None):
        """Masses [Msun] (from the MassTable if the type has no Masses dataset)."""
        if self.has(pt, "Masses"):
            raw = self.read_indices(pt, "Masses", idx) if idx is not None else self.read_ranges(pt, "Masses", ranges)
            return raw.astype(np.float64) * self.msun_per_code
        n = len(idx) if idx is not None else int(sum(b - a for a, b in ranges))
        if self.masstable[pt] <= 0:
            raise ValueError(f"PartType{pt}: no Masses dataset and MassTable entry 0")
        return np.full(n, self.masstable[pt] * self.msun_per_code)

    def scan_sphere(self, pt, center_code, radius_code):
        """
        Global indices of the particles within radius_code (periodic) of center_code, and their
        separation vectors [code length, float64]. Coordinates are read once, chunk by chunk.
        """
        center_code = np.asarray(center_code, np.float64)
        idx, rel = [], []
        r2max = radius_code ** 2
        for i in range(len(self.files)):
            n = int(self.counts[i, pt])
            if n == 0:
                continue
            d = self._file(i)[f"PartType{pt}"]["Coordinates"]
            for a in range(0, n, self.scan_chunk):
                b = min(n, a + self.scan_chunk)
                x = periodic(d[a:b].astype(np.float64) - center_code, self.box)
                sel = np.flatnonzero(np.einsum("ij,ij->i", x, x) < r2max)
                idx.append(sel + a + self.first[i, pt])
                rel.append(x[sel])
        if not idx:
            return np.zeros(0, np.int64), np.zeros((0, 3))
        return np.concatenate(idx).astype(np.int64), np.concatenate(rel)


# ------------------------------------------------------------------ subhalo particles
def subhalo_ranges(cat, subs, pt):
    """Index ranges and lengths of the type-pt particles of the given subhaloes (in the given order)."""
    _, soff = cat.offsets()
    subs = np.asarray(subs, np.int64)
    ln = np.asarray(cat["SubhaloLenType"], np.int64)[subs, pt] if len(subs) else np.zeros(0, np.int64)
    st = soff[subs, pt] if len(subs) else np.zeros(0, np.int64)
    return [(int(a), int(a + n)) for a, n in zip(st, ln)], ln


def binding_order_ok(sn, cat, n_check=20):
    """
    Check that particles within subhaloes are ordered by binding energy: for the n_check largest
    subhaloes, SubhaloIDMostbound must be the first particle of one of the types.
    Returns (ok, list of failing subhalo indices).
    """
    if cat.nsubs == 0:
        return True, []
    _, soff = cat.offsets()
    lt = np.asarray(cat["SubhaloLenType"], np.int64)
    mb = np.asarray(cat["SubhaloIDMostbound"])
    bad = []
    for s in np.argsort(-lt.sum(axis=1), kind="stable")[:n_check]:
        firsts = [sn.read_ranges(pt, "ParticleIDs", [(int(soff[s, pt]), int(soff[s, pt]) + 1)])[0]
                  for pt in range(6) if lt[s, pt] > 0 and sn.has(pt, "ParticleIDs")]
        if mb[s] not in firsts:
            bad.append(int(s))
    return not bad, bad


def most_bound_ids(sn, cat, sub, pt, n, mode="binding"):
    """
    IDs of the (up to) n most-bound type-pt particles of subhalo `sub`, most bound first.
    mode 'binding': the first particles of the subhalo (AREPO orders them by binding energy);
    'distance': ranked by distance to SubhaloPos instead.
    """
    (rng,), (ln,) = subhalo_ranges(cat, [sub], pt)
    if ln == 0:
        return np.zeros(0, np.uint64)
    if mode == "binding":
        return sn.read_ranges(pt, "ParticleIDs", [(rng[0], rng[0] + min(n, int(ln)))])
    ids = sn.read_ranges(pt, "ParticleIDs", [rng])
    x = periodic(sn.read_ranges(pt, "Coordinates", [rng]).astype(np.float64) - cat["SubhaloPos"][sub], sn.box)
    return ids[np.argsort(np.einsum("ij,ij->i", x, x), kind="stable")[:n]]


def match_sorted(sorted_keys, values):
    """Positions of values in sorted_keys and a mask of exact matches (searchsorted checked)."""
    if len(sorted_keys) == 0:
        return np.zeros(len(values), np.int64), np.zeros(len(values), bool)
    pos = np.minimum(np.searchsorted(sorted_keys, values), len(sorted_keys) - 1)
    return pos, sorted_keys[pos] == values
