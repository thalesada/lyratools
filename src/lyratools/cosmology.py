"""
Conversions between scale factor, redshift and cosmic time, computed from the
simulation's own cosmology instead of precomputed conversion tables.

    tc = TimeConversion.from_dataset(ds)      # scida dataset
    tc.t_at_a(0.1)                            # Myr
"""
import glob
import os

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM


class TimeConversion:
    """Fast, vectorised scale factor <-> cosmic time conversions (times in Myr)."""

    def __init__(self, cosmology, a_min=1e-4, n=8000):
        self.cosmology = cosmology
        self._a = np.geomspace(a_min, 1.0, n)
        self._t = cosmology.age(1.0 / self._a - 1.0).to("Myr").value

    @classmethod
    def from_dataset(cls, ds):
        return cls(ds.cosmology)

    @classmethod
    def from_params(cls, h=0.6777, Om0=0.307, Ob0=0.048):
        return cls(FlatLambdaCDM(H0=100 * h, Om0=Om0, Ob0=Ob0))

    def t_at_a(self, a):
        """Cosmic time [Myr] at scale factor a."""
        return np.interp(a, self._a, self._t)

    def a_at_t(self, t):
        """Scale factor at cosmic time t [Myr]."""
        return np.interp(t, self._t, self._a)

    def z_at_t(self, t):
        return 1.0 / self.a_at_t(t) - 1.0

    def t_at_z(self, z):
        return self.t_at_a(1.0 / (1.0 + np.asarray(z, dtype=float)))

    def lookback(self, t, t_now=None):
        """Lookback time [Myr] from t_now (default: z = 0)."""
        t_now = self._t[-1] if t_now is None else t_now
        return t_now - np.asarray(t)


def snapshot_table(output_dir):
    """
    Snapshot numbers with scale factor, redshift and file path, read from the
    headers in an AREPO output directory (replaces the old conversion tables).
    Returns a structured numpy array sorted by snapshot number.
    """
    entries = []
    for p in sorted(glob.glob(os.path.join(output_dir, "snapdir_*")) + glob.glob(os.path.join(output_dir, "snap_*.hdf5"))):
        f0 = sorted(glob.glob(os.path.join(p, "*.hdf5")))[0] if os.path.isdir(p) else p
        try:
            num = int(os.path.basename(p).split("_")[1].split(".")[0])
            with h5py.File(f0, "r") as f:
                a = float(f["Header"].attrs["Time"])
            entries.append((num, a, 1.0 / a - 1.0, p))
        except Exception:
            continue
    tab = np.array(entries, dtype=[("snap", int), ("a", float), ("z", float), ("path", "U512")])
    return np.sort(tab, order="snap")


def snap_at_z(table, z):
    """Snapshot number closest in redshift to z, and its actual redshift."""
    i = int(np.argmin(np.abs(table["z"] - z)))
    return int(table["snap"][i]), float(table["z"][i])


def z_at_snap(table, snap):
    return float(table["z"][np.searchsorted(table["snap"], snap)])
