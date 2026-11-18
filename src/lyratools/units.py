"""Helpers to accept both plain arrays and unit-carrying (pint) quantities, e.g. from scida (optional)."""
import numpy as np

# lyratools default units, tried in order against the dimensionality of a quantity
# (comoving and h factors, which scida carries as the dimensionless units `a` and `h`,
# disappear in the conversion, so results are physical)
DEFAULT_UNITS = ("kpc", "km/s", "Msun", "Myr", "cm**-3", "K", "erg/g", "g/cm**3", "Msun/yr", "erg/s", "erg")


def _is_dask(x):
    return type(x).__module__.startswith("dask")


def _compute(x):
    return x.compute() if _is_dask(x) else x


def as_array(q, unit=None):
    """
    Return a plain numpy array.

    If `q` carries units (e.g. a scida field), it is converted to `unit` first;
    plain arrays are assumed to already be in `unit`.
    """
    if _is_dask(getattr(q, "magnitude", q)):
        q = q.compute()
    if hasattr(q, "magnitude"):
        if unit is not None:
            q = q.to(unit)
        return np.asarray(q.magnitude)
    return np.asarray(q)


def magnitude(q):
    """Strip units from a numpy/dask array or pint quantity, keeping laziness."""
    return getattr(q, "magnitude", q)


def default_unit(units):
    """The lyratools default unit (string) with the dimensionality of pint `units`, or None."""
    reg = units._REGISTRY
    dim = units.dimensionality
    if not dim:
        return ""
    for u in DEFAULT_UNITS:
        try:
            if reg(u).dimensionality == dim:
                return u
        except Exception:        # unit not defined in this registry
            continue
    return None


def unit_factor(from_units, to_unit):
    """Numeric factor converting `from_units` (pint units) to `to_unit` (string or units)."""
    return float((1.0 * from_units).to(to_unit).magnitude)


def to_numpy(q, unit=None):
    """
    A field as a numpy array in `unit` (default: the lyratools unit of its dimensionality,
    else cgs base units). The magnitude is taken first and dask arrays are computed before
    the numeric conversion factor is applied, so pint's dask support is never used.
    Plain (unit-less) arrays are returned as they are.
    """
    if not hasattr(q, "units"):
        return np.asarray(_compute(q))
    mag = np.asarray(_compute(q.magnitude))
    if unit is None:
        unit = default_unit(q.units)
    if unit is None:
        f = float((1.0 * q.units).to_base_units().magnitude)
    else:
        f = unit_factor(q.units, unit)
    if f == 1.0:
        return mag
    if mag.dtype.kind == "f" and mag.dtype.itemsize < 8:
        mag = mag.astype(np.float64)      # code-unit factors (e.g. 1e43 g) overflow float32 on the way
    return mag * f


def require_scida(what):
    """Import check for the optional scida helpers, with an install hint."""
    try:
        import scida  # noqa: F401
        import dask  # noqa: F401
    except ImportError as e:
        raise ImportError(f"{what} needs the optional scida extra: pip install 'lyratools[scida]'") from e
