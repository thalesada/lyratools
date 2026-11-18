"""Star formation histories, burstiness, Kennicutt-Schmidt and phase diagrams. Times in Myr."""
import numpy as np

from .constants import G, MSUN, YR


def sfh(tform, m_init, dt=10.0, bins=None, t_range=None):
    """
    Star formation history: (bin centres [Myr], SFR [Msun/yr]) from formation times [Myr]
    and initial masses [Msun]. `bins` overrides dt and t_range.
    """
    tform = np.asarray(tform, float)
    if bins is None:
        lo, hi = t_range if t_range is not None else (0.0, np.nanmax(tform) + dt)
        bins = np.arange(lo, hi + dt, dt)
    H, e = np.histogram(tform, bins=bins, weights=m_init)
    widths = np.diff(e)
    return e[:-1] + widths / 2, H / (widths * 1e6)


def sfr_trailing(tform, m_init, windows=(5, 10, 50, 100), dt=1.0, t_start=None, t_end=None):
    """
    SFR averaged over trailing windows [Myr] ending at each time of a grid with spacing dt.
    Returns (t_grid [Myr], {window: SFR [Msun/yr]}).
    """
    tform = np.asarray(tform, float)
    t_start = np.floor(np.nanmin(tform)) if t_start is None else t_start
    t_end = np.nanmax(tform) if t_end is None else t_end
    edges = np.arange(t_start, t_end + dt, dt)
    cum = np.concatenate([[0.0], np.cumsum(np.histogram(tform, bins=edges, weights=m_init)[0])])
    i = np.arange(len(edges))
    out = {}
    for w in windows:
        n = int(round(w / dt))
        out[w] = (cum[i] - cum[np.clip(i - n, 0, None)]) / (w * 1e6)
    return edges, out


def find_bursts(t, sfr_short, sfr_long, factor=2.0, t_min=None):
    """
    Contiguous periods where sfr_short > factor * sfr_long. Returns a list of dicts with
    start, end, duration [Myr], peak short-window SFR and peak ratio.
    """
    t = np.asarray(t)
    mask = sfr_short > factor * sfr_long
    if t_min is not None:
        mask &= t >= t_min
    m = np.concatenate([[False], mask, [False]]).astype(int)
    starts, ends = np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0]
    dt = np.median(np.diff(t)) if len(t) > 1 else 0
    bursts = []
    for i0, i1 in zip(starts, ends):
        k = i0 + np.argmax(sfr_short[i0:i1])
        bursts.append(dict(start=t[i0], end=t[i1 - 1], duration=(i1 - i0) * dt, t_peak=t[k],
                           peak_sfr=sfr_short[k], peak_ratio=sfr_short[k] / sfr_long[k] if sfr_long[k] > 0 else np.inf))
    return bursts


def burstiness_stats(t, sfr10, sfr100, window=300.0, t_now=None):
    """sigma(log SFR10), duty cycle and max SFR10/SFR100 over the last `window` Myr."""
    t = np.asarray(t)
    t_now = t[-1] if t_now is None else t_now
    sel = (t > t_now - window) & (t > t[0] + 100)
    pos = sel & (sfr10 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sfr100 > 0, sfr10 / sfr100, np.nan)
    return dict(sigma_log_sfr10=np.std(np.log10(sfr10[pos])) if pos.sum() > 5 else np.nan,
                duty_cycle=pos.sum() / max(sel.sum(), 1),
                max_ratio=np.nanmax(ratio[sel]) if sel.any() else np.nan)


def sfr_from_density(m_gas, rho, efficiency=1.0):
    """SFR [Msun/yr] per cell from mass [Msun] and density [g/cm^3]: eps * m / t_ff."""
    t_ff = np.sqrt(3 * np.pi / (32 * G * np.asarray(rho, float)))      # s
    return efficiency * np.asarray(m_gas, float) / (t_ff / YR)


def compute_ks(gas_pos, gas_mass, star_pos, star_mass, star_age, radius, timebin=10.0):
    """
    Kennicutt-Schmidt point within a projected radius [kpc] (projection along z).
    Stars younger than timebin [Myr] count towards the SFR.
    Returns Sigma_gas [Msun/pc^2], Sigma_SFR [Msun/yr/kpc^2].
    """
    area = np.pi * radius ** 2
    rg = np.hypot(gas_pos[:, 0], gas_pos[:, 1])
    rs = np.hypot(star_pos[:, 0], star_pos[:, 1])
    young = (rs < radius) & (np.asarray(star_age) < timebin)
    sigma_sfr = np.sum(star_mass[young]) / area / (timebin * 1e6)
    sigma_gas = np.sum(gas_mass[rg < radius]) / (area * 1e6)
    return sigma_gas, sigma_sfr


def compute_ks_grid(gas_pos, gas_mass, star_pos, star_mass, star_age, box=1.0, ncell=16, timebin=10.0):
    """KS points on an ncell x ncell grid of a box [kpc] projected along z (flattened arrays)."""
    bins = np.linspace(-box / 2, box / 2, ncell + 1)
    area = (box / ncell) ** 2
    young = np.asarray(star_age) < timebin
    Hs = np.histogram2d(star_pos[young, 0], star_pos[young, 1], bins=(bins, bins), weights=star_mass[young])[0]
    Hg = np.histogram2d(gas_pos[:, 0], gas_pos[:, 1], bins=(bins, bins), weights=gas_mass)[0]
    return (Hg / (area * 1e6)).ravel(), (Hs / area / (timebin * 1e6)).ravel()


def phase_diagram(x, y, weights=None, values=None, extent=(-8, 6, 0, 6), pix=100, log_inputs=True):
    """
    2D histogram for phase diagrams (e.g. x = nH, y = T). With `weights` only: log10 of the
    summed weights per pixel; with `values`: log10 of the (weighted) mean of values per pixel.
    Returns (image transposed for imshow, extent).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if log_inputs:
        with np.errstate(divide="ignore"):
            x, y = np.log10(x), np.log10(y)
    bins = (np.linspace(extent[0], extent[1], pix), np.linspace(extent[2], extent[3], pix))
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    W = np.histogram2d(x, y, bins=bins, weights=w)[0]
    if values is None:
        img = W
    else:
        V = np.histogram2d(x, y, bins=bins, weights=w * np.asarray(values, float))[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            img = V / W
    with np.errstate(divide="ignore"):
        return np.log10(img.T), extent
