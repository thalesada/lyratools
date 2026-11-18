"""
Slices and projections that respect AREPO's Voronoi mesh (port of gadget_snap.get_Aslice /
calcGrid.calcASlice).

Every image pixel is sampled at `nz` points along the line of sight; each sample takes the
value of its nearest mesh-generating point, i.e. of the Voronoi cell that contains it. A
projection therefore integrates the actual cell shapes, and a slice shows the cells themselves
(with their boundaries available as `contours`). Optionally the values are linearly
reconstructed inside each cell from gradients, as in AREPO.

    from lyratools import voronoi
    halo = lt.load_halo(ds)
    img = voronoi.column_density(halo.gas, box=20, res=512)               # Msun / pc^2
    T   = voronoi.weighted_projection(halo.gas, halo.gas.get_as("Temperature", "K"), box=20)
    sl  = voronoi.cell_slice(halo.gas, halo.gas.get_as("Density", "Msun/kpc**3"), box=2)
    plt.imshow(np.log10(img["grid"]), extent=img["extent"], origin="lower")
    plt.imshow(np.ma.masked_equal(sl["contours"], 0), extent=sl["extent"], origin="lower", cmap="gray")

Cost: one nearest-neighbour query per sample, i.e. res x res x nz queries (about 15 s for
256^3 samples on one core; threads scale this down). Lower `nz` for quick looks.
Not ported from get_Aslice: polar and all-sky projections.

Images are returned ready for imshow(origin="lower"): grid[iy, ix], first axis along `axes[1]`.
"""
import numpy as np
from scipy.spatial import cKDTree

from .units import as_array


def _cell_radius(gas):
    """Radius of a sphere with each cell's volume [kpc], from mass and density."""
    vol = gas["mass"] / as_array(gas["Density"], "Msun/kpc**3")
    return (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)


def _select(pos, center, axes, half_x, half_y, half_z, pad):
    """Cells whose generating point lies within the image volume, enlarged by `pad` (per cell or scalar)."""
    d = pos - center
    ax2 = 3 - axes[0] - axes[1]
    return ((np.abs(d[:, axes[0]]) < half_x + pad) & (np.abs(d[:, axes[1]]) < half_y + pad)
            & (np.abs(d[:, ax2]) < half_z + pad))


def voronoi_grid(pos, values, box, res=512, center=(0.0, 0.0, 0.0), axes=(0, 1), proj=True, depth=None,
                 proj_fact=0.5, nx=None, ny=None, nz=None, weights=None, grad=None, mode="integral",
                 cell_radius=None, pad=None, workers=-1, return_neighbours=None, tree=None, verbose=True):
    """
    Sample a quantity defined on Voronoi cells onto a regular image grid.

    pos:     (N, 3) cell positions (any length unit; box, depth, center in the same unit)
    values:  (N,) cell values
    box:     image size, scalar or (bx, by)
    proj:    True: project along the third axis over `depth` (default 2 * proj_fact * max(box));
             False: slice through center
    nz:      samples along the line of sight (default 2 * proj_fact * res, as in get_Aslice)
    weights: (N,) optional weights: result is sum(w * v) / sum(w) along the line of sight
             (e.g. pass density to get a mass-weighted projection)
    grad:    (N, 3) optional gradients of `values` for linear reconstruction inside cells
    mode:    for unweighted projections: "integral" = sum(v * dz) (e.g. density -> column density),
             "mean" = average along the line of sight, "sum" = raw sum of samples (calcASlice output)
    cell_radius: (N,) approximate cell radii; cells are kept if within the image volume plus
             2 radii, so pixels near the edges find their true nearest cell (default pad: 5% of box)
    workers: threads for the KD-tree queries (-1 = all cores)
    tree:    a cKDTree built on exactly the selected positions, to reuse between images

    Returns dict with grid (ny, nx) for imshow, x and y pixel edges, extent, and for slices
    `neighbours` (index of the cell shown in each pixel, into the input arrays) and `contours`
    (1 on Voronoi cell boundaries).
    """
    pos = np.asarray(pos, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    bx, by = (float(box), float(box)) if np.ndim(box) == 0 else (float(box[0]), float(box[1]))
    nx = res if nx is None else nx
    ny = res if ny is None else ny
    a0, a1 = axes
    a2 = 3 - a0 - a1
    if proj:
        bz = 2.0 * proj_fact * max(bx, by) if depth is None else float(depth)
        nz = int(2 * proj_fact * res) if nz is None else int(nz)
    else:
        bz, nz = 0.0, 1
    return_neighbours = (not proj) if return_neighbours is None else return_neighbours

    if pad is None:
        pad = 2.0 * np.asarray(cell_radius) if cell_radius is not None else 0.05 * max(bx, by, bz)
    sel = np.flatnonzero(_select(pos, center, axes, bx / 2, by / 2, bz / 2, pad))
    if verbose:
        print(f"Selected {sel.size} of {len(pos)} cells; nx={nx}, ny={ny}, nz={nz}, depth={bz:g}")
    if sel.size == 0:
        raise ValueError("no cells inside the image volume")
    p, v = pos[sel], values[sel]
    w = None if weights is None else np.asarray(weights, dtype=np.float64)[sel]
    g = None if grad is None else np.asarray(grad, dtype=np.float64)[sel]
    tree = cKDTree(p) if tree is None else tree

    xc = center[a0] - bx / 2 + bx / nx * (np.arange(nx) + 0.5)
    yc = center[a1] - by / 2 + by / ny * (np.arange(ny) + 0.5)
    zc = center[a2] - bz / 2 + (bz / nz * (np.arange(nz) + 0.5) if proj else np.zeros(1))
    X, Y = np.meshgrid(xc, yc)                      # (ny, nx): image orientation
    coord = np.empty((ny * nx, 3))
    coord[:, a0] = X.ravel()
    coord[:, a1] = Y.ravel()

    acc = np.zeros(ny * nx)
    accw = np.zeros(ny * nx) if w is not None else None
    for z in zc:                                    # one plane of samples at a time: bounded memory
        coord[:, a2] = z
        _, nb = tree.query(coord, k=1, workers=workers)
        val = v[nb]
        if g is not None:
            val = val + np.einsum("ij,ij->i", coord - p[nb], g[nb])
        if w is None:
            acc += val
        else:
            acc += w[nb] * val
            accw += w[nb]

    if w is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = acc / accw
    elif proj and mode == "integral":
        grid = acc * (bz / nz)
    elif proj and mode == "mean":
        grid = acc / nz
    else:
        grid = acc

    out = dict(grid=grid.reshape(ny, nx),
               x=center[a0] - bx / 2 + bx * np.arange(nx + 1) / nx,
               y=center[a1] - by / 2 + by * np.arange(ny + 1) / ny,
               extent=[center[a0] - bx / 2, center[a0] + bx / 2, center[a1] - by / 2, center[a1] + by / 2],
               depth=bz, nz=nz, n_cells=sel.size)
    if return_neighbours:
        nbimg = sel[nb].reshape(ny, nx)             # last sampled plane (the slice plane for slices)
        out["neighbours"] = nbimg
        out["contours"] = cell_boundaries(nbimg)
    return out


def cell_boundaries(neighbours):
    """1 where any of the 8 surrounding pixels belongs to a different cell (as in calcASlice)."""
    n = neighbours
    c = np.zeros(n.shape, dtype=np.int8)
    core = n[1:-1, 1:-1]
    diff = np.zeros(core.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                diff |= n[1 + dy:n.shape[0] - 1 + dy, 1 + dx:n.shape[1] - 1 + dx] != core
    c[1:-1, 1:-1] = diff
    return c


# ---------------------------------------------------------------------------------------------
# convenience wrappers for lyratools particle sets (positions in kpc)
# ---------------------------------------------------------------------------------------------
def _common(gas, kwargs):
    kwargs.setdefault("cell_radius", _cell_radius(gas) if "Density" in gas else None)
    return np.asarray(gas["pos"], float), kwargs


def projection(gas, values, box, **kwargs):
    """Unweighted projection of `values` (see voronoi_grid for options, e.g. mode='mean')."""
    pos, kwargs = _common(gas, kwargs)
    return voronoi_grid(pos, values, box, proj=True, **kwargs)


def weighted_projection(gas, values, box, weights=None, **kwargs):
    """Mass-weighted (default: density-weighted along the line of sight) projection of `values`."""
    pos, kwargs = _common(gas, kwargs)
    if weights is None:
        weights = as_array(gas["Density"], "Msun/kpc**3")
    return voronoi_grid(pos, values, box, proj=True, weights=weights, **kwargs)


def column_density(gas, box, unit="Msun/pc**2", values=None, **kwargs):
    """
    Projected column density of the gas: integral of density along the line of sight.
    unit 'Msun/pc**2' or 'Msun/kpc**2'; with unit='cm**-2', pass values = nH [cm^-3] for N_H.
    """
    pos, kwargs = _common(gas, kwargs)
    if unit == "cm**-2":
        if values is None:
            raise ValueError("for a number column density pass values = number density [cm^-3]")
        img = voronoi_grid(pos, values, box, proj=True, mode="integral", **kwargs)
        img["grid"] = img["grid"] * 3.0856776e21          # cm^-3 * kpc -> cm^-2
        return img
    rho = as_array(gas["Density"], "Msun/kpc**3") if values is None else values
    img = voronoi_grid(pos, rho, box, proj=True, mode="integral", **kwargs)
    if unit == "Msun/pc**2":
        img["grid"] = img["grid"] / 1e6
    elif unit != "Msun/kpc**2":
        raise ValueError("unit must be 'Msun/pc**2', 'Msun/kpc**2' or 'cm**-2'")
    return img


def cell_slice(gas, values, box, **kwargs):
    """Slice through the centre showing the Voronoi cells; result includes `contours` of cell edges."""
    pos, kwargs = _common(gas, kwargs)
    return voronoi_grid(pos, values, box, proj=False, **kwargs)
