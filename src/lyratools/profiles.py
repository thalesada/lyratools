"""Radial profiles, characteristic radii and shapes. Lengths in kpc unless stated otherwise."""
import numpy as np

from .constants import G_KPC_KMS2_PER_MSUN


def radii(pos, center=(0, 0, 0), mode="3d"):
    pos = np.asarray(pos, float)
    c = np.asarray(center, float)
    if mode == "3d":
        return np.linalg.norm(pos - c, axis=1)
    return np.hypot(pos[:, 0] - c[0], pos[:, 1] - c[1])


def density_profile(pos, mass, bins, mode="3d", center=(0, 0, 0), counts=False):
    """
    (bin centres [kpc], log10 density [Msun/kpc^3] in 3d or surface density [Msun/kpc^2] in 2d).
    With counts=True also returns the number of particles per bin.
    """
    r = radii(pos, center, mode)
    bins = np.asarray(bins, float)
    vol = 4.0 / 3.0 * np.pi * np.diff(bins ** 3) if mode == "3d" else np.pi * np.diff(bins ** 2)
    M = np.histogram(r, bins=bins, weights=mass)[0]
    rc = 0.5 * (bins[1:] + bins[:-1])
    with np.errstate(divide="ignore"):
        prof = np.log10(M / vol)
    if counts:
        return rc, prof, np.histogram(r, bins=bins)[0]
    return rc, prof


def rotation_curve(pos, mass, bins, center=(0, 0, 0), mass_factor=1.0):
    """
    Circular velocity sqrt(G M(<r) / r) [km/s] at the bin edges' midpoints [kpc].
    mass_factor rescales masses (e.g. 1 - f_baryon for dark-matter-only runs).
    """
    r = radii(pos, center)
    M = np.histogram(r, bins=bins, weights=np.asarray(mass) * mass_factor)[0]
    rc = 0.5 * (bins[1:] + bins[:-1])
    menc = np.cumsum(M) + np.sum(np.asarray(mass)[r < bins[0]]) * mass_factor
    return rc, np.sqrt(G_KPC_KMS2_PER_MSUN * menc / rc)


def surface_brightness_profile(pos_pc, luminosity, bins_pc, center=(0, 0), counts=False, M_sun=4.8):
    """
    Surface brightness [mag/arcsec^2] in projected annuli [pc] from luminosities in Lsun
    (projection along z): mu = M_sun + 21.572 - 2.5 log10(L / pc^2).
    """
    r = radii(pos_pc, center, mode="2d")
    good = np.isfinite(luminosity)
    area = np.pi * np.diff(np.asarray(bins_pc, float) ** 2)
    L = np.histogram(r[good], bins=bins_pc, weights=np.asarray(luminosity)[good])[0]
    rc = 0.5 * (bins_pc[1:] + bins_pc[:-1])
    with np.errstate(divide="ignore"):
        sb = M_sun + 21.572 - 2.5 * np.log10(L / area)
    if counts:
        return rc, sb, np.histogram(r[good], bins=bins_pc)[0]
    return rc, sb


def half_mass_radius(r, mass):
    """Radius enclosing half the total (positive) mass."""
    r, mass = np.asarray(r), np.asarray(mass)
    good = np.isfinite(mass) & (mass > 0)
    if not good.any():
        return np.nan
    o = np.argsort(r[good])
    cm = np.cumsum(mass[good][o], dtype=np.float64)     # float32 sums stall near 2^24 x the particle mass
    return r[good][o][np.searchsorted(cm, 0.5 * cm[-1])]


def half_light_radius(r, luminosity, rmax=None):
    """Radius enclosing half the luminosity within rmax (all if None)."""
    r, L = np.asarray(r), np.asarray(luminosity)
    sel = np.isfinite(L) & (L > 0) & ((r < rmax) if rmax is not None else True)
    return half_mass_radius(r[sel], L[sel])


def ellipticity(mass, pos):
    """Ellipticity sqrt(1 - (I_min/I_max)^2) from the eigenvalues of the inertia tensor."""
    mass, pos = np.asarray(mass, float), np.asarray(pos, float)
    if len(mass) == 0:
        return np.nan
    x, y, z = pos.T
    I = np.array([[np.sum(mass * (y ** 2 + z ** 2)), -np.sum(mass * x * y), -np.sum(mass * x * z)],
                  [-np.sum(mass * x * y), np.sum(mass * (x ** 2 + z ** 2)), -np.sum(mass * y * z)],
                  [-np.sum(mass * x * z), -np.sum(mass * y * z), np.sum(mass * (x ** 2 + y ** 2))]])
    I /= np.nansum(mass)
    ev = np.linalg.eigvalsh(I)
    return np.sqrt((ev.max() ** 2 - ev.min() ** 2) / ev.max() ** 2)
