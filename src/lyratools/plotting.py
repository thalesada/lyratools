"""Plot styles, redshift axes, scale bars and particle images."""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

STYLE_DIR = os.path.join(os.path.dirname(__file__), "styles")


def use_style(name="ttt", quiet_fonts=True):
    """
    Apply a bundled matplotlib style: 'ttt' (the original TTT settings) or 'lyra' (paper figures).

    The 'lyra' style embeds fonts as TrueType in PDFs (pdf.fonttype 42: editable text, as journals
    require), which matplotlib does through fontTools. With quiet_fonts (default), fontTools' per-font
    log messages are silenced, which otherwise appear for every saved PDF when the root logger is at INFO.
    """
    plt.style.use(os.path.join(STYLE_DIR, f"{name}.mplstyle"))
    if quiet_fonts:
        import logging
        logging.getLogger("fontTools").setLevel(logging.ERROR)


def add_zaxis(ax, tconv, zticks=(10, 6, 4, 3, 2, 1, 0.5, 0), lookback=False, time_unit="Gyr"):
    """
    Redshift axis on top of an axis whose x-axis is cosmic (or lookback) time.
    tconv: lyratools.cosmology.TimeConversion; time_unit of the x-axis: 'Gyr' or 'Myr'.
    """
    scale = 1e-3 if time_unit == "Gyr" else 1.0
    t = tconv.t_at_z(np.asarray(zticks, float)) * scale
    if lookback:
        t = tconv.t_at_z(0) * scale - t
    ax2 = ax.twiny()
    lo, hi = ax.get_xlim()
    ok = (t >= min(lo, hi)) & (t <= max(lo, hi))
    ax2.set_xlim(lo, hi)
    ax2.set_xticks(t[ok])
    ax2.set_xticklabels([f"{z:g}" for z in np.asarray(zticks)[ok]])
    ax2.set_xlabel("$z$")
    return ax2


def add_scale_bar(ax, length, label=None, color="w", loc=(0.05, 0.9), lw=3, fontsize=14, hide_ticks=True):
    """Scale bar of `length` in data units, starting at axes fraction `loc`."""
    x0, x1 = ax.get_xlim()
    frac = length / abs(x1 - x0)
    ax.plot([loc[0], loc[0] + frac], [loc[1], loc[1]], color=color, lw=lw, transform=ax.transAxes)
    ax.text(loc[0] + 0.01, loc[1] + 0.03, label if label is not None else f"{length:g}", color=color,
            fontsize=fontsize, transform=ax.transAxes)
    if hide_ticks:
        ax.set_xticks([]); ax.set_yticks([])


def particle_image(pos, weights, boxsize, res=500, axes=(0, 1, 2), depth=None, values=None, sigma=0.0, log=True):
    """
    Projected image of particles within a cube of side `boxsize` (and |z| < depth/2 if given).
    Without `values`: surface density of `weights` per unit area (log10 if log=True);
    with `values`: weighted mean of `values`. Returns (image for imshow, extent).
    """
    pos = np.asarray(pos, float)
    half = boxsize / 2.0
    depth = boxsize if depth is None else depth
    sel = np.abs(pos[:, axes[2]]) < depth / 2
    rng = [[-half, half]] * 2
    W = np.histogram2d(pos[sel, axes[0]], pos[sel, axes[1]], bins=res, range=rng, weights=np.asarray(weights)[sel])[0]
    if values is not None:
        V = np.histogram2d(pos[sel, axes[0]], pos[sel, axes[1]], bins=res, range=rng,
                           weights=(np.asarray(weights) * np.asarray(values))[sel])[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            img = V / W
    else:
        img = W / (boxsize / res) ** 2
        if log:
            with np.errstate(divide="ignore"):
                img = np.log10(img)
    if sigma > 0:
        img = gaussian_filter(np.nan_to_num(img, neginf=np.nanmin(img[np.isfinite(img)]) if np.isfinite(img).any() else 0), sigma)
    return img.T, [-half, half, -half, half]
