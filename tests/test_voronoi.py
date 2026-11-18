import time, numpy as np
from scipy.spatial import cKDTree
from lyratools import voronoi as vo
rng = np.random.default_rng(1)
res = {}
def check(name, cond, info=""):
    res[name] = bool(cond); print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")

N = 20000
pos = rng.uniform(-1, 1, (N, 3))

# 1. each pixel shows exactly the nearest cell (brute force)
vals = rng.normal(size=N)
sl = vo.voronoi_grid(pos, vals, box=1.0, res=40, proj=False, pad=0.5, verbose=False)
xc = np.linspace(-0.5, 0.5, 41)[:-1] + 0.0125
X, Y = np.meshgrid(xc, xc)
pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
nb = np.argmin(((pts[:, None, :] - pos[None, :, :]) ** 2).sum(-1), axis=1)
check("slice = brute-force nearest cell", np.array_equal(sl["grid"].ravel(), vals[nb]) and np.array_equal(sl["neighbours"].ravel(), nb))
check("orientation: grid[iy, ix]", np.allclose(sl["grid"][3, 7], vals[nb.reshape(40, 40)[3, 7]]))

# 2. uniform density -> column = rho * depth everywhere
img = vo.voronoi_grid(pos, np.full(N, 2.5), box=1.0, res=32, depth=0.8, nz=64, pad=0.5, verbose=False)
check("uniform column density", np.allclose(img["grid"], 2.5 * 0.8))

# 3. mass conservation with Voronoi-consistent densities (volumes from independent Monte Carlo)
mc = rng.uniform(-1, 1, (4_000_000, 3))
_, owner = cKDTree(pos).query(mc, workers=-1)
V = np.bincount(owner, minlength=N) * 8.0 / len(mc)
m = rng.uniform(1, 2, N)
rho = m / np.maximum(V, 1e-12)
img = vo.voronoi_grid(pos, rho, box=2.0, res=200, depth=2.0, nz=200, pad=0.0, verbose=False)
M_img = img["grid"].sum() * (2.0 / 200) ** 2
check("projected mass = total mass", abs(M_img / m.sum() - 1) < 0.01, f"ratio {M_img / m.sum():.4f}")

# 4. linear field with exact gradients -> exact reconstruction
a = np.array([0.3, -1.2, 0.7])
f = pos @ a
gradf = np.tile(a, (N, 1))
sl = vo.voronoi_grid(pos, f, box=1.0, res=50, proj=False, grad=gradf, pad=0.5, center=(0.1, 0.0, 0.2), verbose=False)
xs = 0.1 - 0.5 + (np.arange(50) + 0.5) / 50; ys = -0.5 + (np.arange(50) + 0.5) / 50
Xs, Ys = np.meshgrid(xs, ys)
check("gradient reconstruction exact", np.allclose(sl["grid"], a[0] * Xs + a[1] * Ys + a[2] * 0.2))
sl0 = vo.voronoi_grid(pos, f, box=1.0, res=50, proj=False, pad=0.5, center=(0.1, 0.0, 0.2), verbose=False)
check("without gradients: piecewise constant", not np.allclose(sl0["grid"], sl["grid"]))
pr = vo.voronoi_grid(pos, f, box=1.0, res=20, proj=True, depth=1.0, nz=40, grad=gradf, mode="mean", pad=0.5, verbose=False)
xs = -0.5 + (np.arange(20) + 0.5) / 20; Xp, Yp = np.meshgrid(xs, xs)
check("projected mean of linear field", np.allclose(pr["grid"], a[0] * Xp + a[1] * Yp))

# 5. weighted projection of a constant -> constant; weights matter otherwise
wimg = vo.voronoi_grid(pos, np.full(N, 7.0), box=1.0, res=20, weights=rng.uniform(0, 1, N), pad=0.5, verbose=False)
check("weighted projection of constant", np.allclose(wimg["grid"], 7.0))

# 6. other axes
img_xz = vo.voronoi_grid(pos, np.full(N, 1.0), box=(1.0, 0.5), res=20, axes=(0, 2), depth=0.4, nz=10, pad=0.5, verbose=False)
check("axes and rectangular box", img_xz["grid"].shape == (20, 20) and np.allclose(img_xz["grid"], 0.4) and img_xz["extent"] == [-0.5, 0.5, -0.25, 0.25])

# 7. contours mark cell boundaries
sl = vo.voronoi_grid(pos, vals, box=0.5, res=200, proj=False, pad=0.5, verbose=False)
frac = sl["contours"].mean()
check("contours", 0.02 < frac < 0.6, f"{frac:.2f} of pixels on cell edges")

# 8. speed: 80k cells, 256^2 pixels x 256 samples
pos8 = rng.normal(0, 1, (80000, 3))
t0 = time.time()
vo.voronoi_grid(pos8, np.ones(80000), box=4, res=256, verbose=False)
check("speed", True, f"256x256x256 samples, 80k cells: {time.time() - t0:.1f} s on {__import__('os').cpu_count()} cores")


# wrappers on a lyratools-style particle set
gas = {"pos": pos * 3, "mass": np.full(N, 4.0), "Density": np.full(N, 1e7)}   # plain arrays: Msun/kpc^3 assumed
col = vo.column_density(gas, box=2.0, res=32, depth=2.0, verbose=False)
check("column_density wrapper", np.allclose(col["grid"], 1e7 * 2.0 / 1e6))
sl = vo.cell_slice(gas, np.arange(N, dtype=float), box=1.0, res=32, verbose=False)
check("cell_slice wrapper", "contours" in sl and np.array_equal(sl["grid"], sl["neighbours"].astype(float)))
print(f"\n{sum(res.values())}/{len(res)} voronoi checks passed")
if not all(res.values()):
    raise SystemExit(1)
