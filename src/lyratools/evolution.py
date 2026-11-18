"""
Main-halo properties across all group catalogs of a run (replaces SubfindEvolution),
read directly with h5py so that it is cheap to build for hundreds of outputs.
All quantities in physical units (Msun, kpc, km/s, Zsun) unless noted.
"""
import glob
import os

import h5py
import numpy as np

from .constants import ZSOLAR
from .cosmology import TimeConversion


class SubfindEvolution:
    """
    Properties of FoF group 0 and its central subhalo in every group catalog of `output_dir`.

    Arrays (one entry per catalog, NaN where missing):
      Mvir, Rvir, Rhalf, Rhalf_d, Rhalf_g, Rhalf_s, Rmax, Mgas, Mdm, Mstar, Mgas_rh, Mdm_rh, Mstar_rh,
      Zgas, Zstar, Zgas_rh, Zstar_rh, Zgas_sf, Sigma, Vmax, redshift, a, time [Myr],
      N_lum_subs_rvir, M_lum_subs_rvir, boundID (first 200 most-bound IDs)
    Comoving arrays (for halo tracks): sub_pos_ckpc, sub_vel, sub_rad_ckpc
    """

    def __init__(self, output_dir, dmo=False, min_rad_pc=4.0, min_mass=1e4, frac_rvir=1.0, n_bound=200):
        self.output_dir = output_dir
        cats = sorted(glob.glob(os.path.join(output_dir, "groups_*")))
        N = len(cats)
        names = ["Mvir", "Rvir", "Rhalf", "Rhalf_d", "Rhalf_g", "Rhalf_s", "Rmax", "Mgas", "Mdm", "Mstar",
                 "Mgas_rh", "Mdm_rh", "Mstar_rh", "Zgas", "Zstar", "Zgas_rh", "Zstar_rh", "Zgas_sf",
                 "Sigma", "Vmax", "redshift", "a", "N_lum_subs_rvir", "M_lum_subs_rvir", "sub_rad_ckpc"]
        for n in names:
            setattr(self, n, np.full(N, np.nan))
        self.sub_pos_ckpc = np.full((N, 3), np.nan)
        self.sub_vel = np.full((N, 3), np.nan)
        self.boundID = np.full((N, n_bound), np.nan)
        self.snap = np.full(N, -1)
        tconv = None

        for i, cat in enumerate(cats):
            self.snap[i] = int(os.path.basename(cat).split("_")[1])
            f0 = sorted(glob.glob(os.path.join(cat, "fof_subhalo_tab_*.0.hdf5")))
            if not f0:
                continue
            with h5py.File(f0[0], "r") as hc:
                if "Subhalo" not in hc or len(hc["Subhalo"].keys()) == 0:
                    continue
                hd = hc["Header"].attrs
                h, a = float(hd["HubbleParam"]), float(hd["Time"])
                if tconv is None and "Omega0" in hd:
                    tconv = TimeConversion.from_params(h=h, Om0=float(hd["Omega0"]))
                L = float(hd.get("UnitLength_in_cm", 3.085678e24)) / 3.085678e21     # kpc per code length
                M = float(hd.get("UnitMass_in_g", 1.989e43)) / 1.98841e33             # Msun per code mass
                pos_f, m_f = L * a / h, M / h
                G, S = hc["Group"], hc["Subhalo"]
                self.a[i], self.redshift[i] = a, float(hd["Redshift"])
                self.Mvir[i] = G["Group_M_Crit200"][0] * m_f
                self.Rvir[i] = G["Group_R_Crit200"][0] * pos_f
                self.Rhalf[i] = S["SubhaloHalfmassRad"][0] * pos_f
                self.Rhalf_d[i] = S["SubhaloHalfmassRadType"][0, 1] * pos_f
                self.Rmax[i] = S["SubhaloVmaxRad"][0] * pos_f
                self.Mdm[i] = S["SubhaloMassType"][0, 1] * m_f
                self.Mdm_rh[i] = S["SubhaloMassInHalfRadType"][0, 1] * m_f
                if not dmo:
                    self.Rhalf_g[i] = S["SubhaloHalfmassRadType"][0, 0] * pos_f
                    self.Rhalf_s[i] = S["SubhaloHalfmassRadType"][0, 4] * pos_f
                    self.Mgas_rh[i] = S["SubhaloMassInHalfRadType"][0, 0] * m_f
                    self.Mstar_rh[i] = S["SubhaloMassInHalfRadType"][0, 4] * m_f
                    self.Mgas[i] = S["SubhaloMassType"][0, 0] * m_f
                    self.Mstar[i] = S["SubhaloMassType"][0, 4] * m_f
                    for attr, key, grp in [("Zgas", "GroupGasMetalFractions", G), ("Zstar", "GroupStarMetalFractions", G),
                                           ("Zgas_rh", "SubhaloGasMetalFractionsHalfRad", S),
                                           ("Zstar_rh", "SubhaloStarMetalFractionsHalfRad", S),
                                           ("Zgas_sf", "SubhaloGasMetalFractionsSfr", S)]:
                        if key in grp:
                            getattr(self, attr)[i] = np.sum(grp[key][0, 2:]) / ZSOLAR
                self.Sigma[i] = S["SubhaloVelDisp"][0]
                self.Vmax[i] = S["SubhaloVmax"][0]
                nb = min(n_bound, len(S["SubhaloIDMostbound"]))
                self.boundID[i, :nb] = S["SubhaloIDMostbound"][:nb]
                self.sub_pos_ckpc[i] = G["GroupPos"][0] * L / h
                self.sub_vel[i] = G["GroupVel"][0]
                self.sub_rad_ckpc[i] = G["Group_R_Crit200"][0] * L / h

                # luminous satellites of group 0 within frac_rvir x R200 (assumes they are in file .0)
                d = np.linalg.norm(S["SubhaloPos"][1:] - G["GroupPos"][0], axis=1)
                sel = ((G["Group_R_Crit200"][0] * frac_rvir > d) & (S["SubhaloLenType"][1:, 4] > 1)
                       & (S["SubhaloHalfmassRad"][1:] * pos_f * 1e3 > min_rad_pc)
                       & (S["SubhaloMassType"][1:, 4] * m_f > min_mass))
                self.N_lum_subs_rvir[i] = np.count_nonzero(sel)
                self.M_lum_subs_rvir[i] = np.sum(S["SubhaloMassType"][1:, 4][sel] * m_f)
        self.time = tconv.t_at_a(self.a) if tconv is not None else np.full(N, np.nan)

    def keys(self):
        return [k for k in self.__dict__ if not k.startswith("_")]

    def halo_track(self):
        """(time [Myr], position [ckpc], R200 [ckpc]) of the main halo, for ex-situ classification."""
        ok = np.isfinite(self.time) & np.all(np.isfinite(self.sub_pos_ckpc), axis=1)
        return self.time[ok], self.sub_pos_ckpc[ok], self.sub_rad_ckpc[ok]
