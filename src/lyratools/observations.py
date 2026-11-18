"""Observational comparison data."""
import numpy as np

from . import config


def ks_sigma_figgs():
    """(Sigma_gas [Msun/pc^2], Sigma_SFR, error) for the FIGGS points in KS.txt."""
    ks = np.genfromtxt(config.data_path("ks_data"), unpack=True)
    return 10 ** ks[0, :6], ks[3, :6], ks[4, :6]


def ks_sigma_things():
    ks = np.genfromtxt(config.data_path("ks_data"), unpack=True)
    return 10 ** ks[0, 6:11], ks[3, 6:11], ks[4, 6:11]


class McConnachie2012:
    """
    Local Group dwarf catalogue (McConnachie 2012, Jan 2021 version). Values of 999 become NaN.
    Attributes: dist_kpc, rh [pc], AbsV, AbsB, FeH, Ell, Sigs, Vrots and their *err pairs; _names lists all columns.
    """

    def __init__(self, path=None):
        from astropy.io import fits
        with fits.open(path or config.data_path("mcconnachie")) as NG:
            d = NG[1].data
            self._names = d.names

            def col(name):
                if name not in d.names:          # e.g. the Jan 2021 file has no B magnitudes
                    return np.full(len(d), np.nan)
                v = np.array(d[name], dtype=float)
                return np.where(v == 999, np.nan, v)

            self.dmod = col("dmod")
            self.dist_kpc = 10 ** (self.dmod / 5 + 1) / 1e3
            self.dist_kpc_err = [10 ** (col("dmod-") / 5 + 1) / 1e3, 10 ** (col("dmod+") / 5 + 1) / 1e3]
            arcmin_to_pc = lambda x: np.tan(x * 2 * np.pi / 360 / 60 / 2) * 2 * self.dist_kpc * 1e3
            self.rh = arcmin_to_pc(col("rh"))
            self.rherr = [arcmin_to_pc(col("rh-")), arcmin_to_pc(col("rh+"))]
            self.AbsV = col("Vmag") - self.dmod
            self.MVerr = [col("Vmag-"), col("Vmag+")]
            self.AbsB = col("Bmag") - self.dmod
            self.MBerr = [col("Bmag-"), col("Bmag+")]
            self.FeH, self.FeHerr = col("[Fe/H]"), [col("feh-"), col("feh+")]
            self.Ell, self.Ellerr = col("e=1-b/a"), [col("e-"), col("e+")]
            self.Sigs, self.Sigserr = col("sigma_s"), [col("sigma_s-"), col("sigma_s+")]
            self.Vrots, self.Vrotserr = col("vrot_s"), [col("vrot_s-"), col("vrot_s+")]
