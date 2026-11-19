"""
Derived per-particle fields, computed with numpy from the fields a Region reads on demand.

    halo = lyratools.load_halo(snapdir, 0)
    halo.gas["nH"], halo.gas["Temperature"], halo.stars["Fe_H"], halo.gas.values("CoolingTime")

Gas (PartType0):  nH [cm^-3], Temperature [K], Pressure_k [K cm^-3], Z_Zsun,
                  CoolingTime [Myr], SelfShieldingFactor, element ratios X_Y,
                  net_true [erg cm^3 s^-1] (FUV runs)
Stars (PartType4): FormationTime [Myr], Age [Myr], SNState (0 none, 1 pending, 2 exploded),
                  DeathTime [Myr], ESN [erg], Z_Zsun, element ratios X_Y
Both: 12logOH (12 + log O/H by number), X_H and X_O ([X/H], [X/O] relative to
      Asplund et al. 2009; mass-fraction ratios over the solar ones, i.e. by number).

Fields stored in the snapshot always win over a derived field of the same name: LYRA
writes Temperature, and that is what halo.gas["Temperature"] returns.

add_fields(ds) registers the same fields as scida recipes on a scida dataset (optional; needs scida).
"""
import numpy as np

from . import chemistry, stellar_evolution
from .constants import BOLTZMANN, GAMMA_MINUS1, HYDROGEN_MASSFRAC, PROTONMASS, ZSOLAR

RATIOS = [("Fe", "H"), ("O", "Fe"), ("Mg", "Fe"), ("C", "Fe"), ("N", "Fe"), ("Si", "Fe"), ("Mn", "Fe"), ("O", "H")]
RATIOS += [(el, "H") for el in ("He", "C", "N", "Mg", "Si", "Mn")]
RATIOS += [(el, "O") for el in ("C", "N", "Mg", "Si", "Mn", "Fe")]


# ------------------------------------------------------------------ FUV runs (VIRGODWARF_PE_HEATING)
# Fields LYRA writes with VIRGODWARF_OUTPUT_PE_RATES, stored with all unit exponents 0: their units are listed
# here (Particles.values() uses them). Either bypass name is accepted.
FIELD_REGISTRY = {
    "FUV_G0": dict(units="", description=(
        "incident (unattenuated) stellar FUV field in Habing units (5.29e-14 erg cm^-3 in 6-13.6 eV), fuv.c")),
    "FUV_PEHeatingRate": dict(units="erg*cm**3/s", description=(
        "photoelectric heating Gamma_PE / n_H^2. Old scheme: fuv.c's analytic term, NOT included in GFM_CoolingRate. "
        "G0 table (VIRGODWARF_G0_COOLING_TABLE): the table's photoelectric channel, already part of GFM_CoolingRate "
        "(except in cells flagged by G0TableDebugBypass, where it is again the analytic term outside GFM_CoolingRate)")),
    "G0TableDebugBypass": dict(units=None, description=(
        "debug flag (VIRGODWARF_G0_TABLE_DEBUG_BYPASS): 1 = the cell used the G0 = 0 slice plus the analytic PE term; "
        "written as G0TableBypass by the first G0-table version")),
    "GFM_CoolingRate": dict(units="erg*cm**3/s", description="(heating - cooling) / n_H^2 of the cooling solver; negative = net cooling"),
    "net_true": dict(units="erg*cm**3/s", description=(
        "net rate including photoelectric heating: GFM_CoolingRate + FUV_PEHeatingRate in the old scheme (and for "
        "bypassed cells), GFM_CoolingRate with the G0 table")),
}
FIELD_REGISTRY["G0TableBypass"] = FIELD_REGISTRY["G0TableDebugBypass"]


def uses_g0_table(snap):
    """True if the run was compiled with VIRGODWARF_G0_COOLING_TABLE (from the snapshot's Config)."""
    cfg = getattr(snap, "config", None) or {}
    if callable(cfg):
        cfg = cfg()
    return "VIRGODWARF_G0_COOLING_TABLE" in cfg


def net_true_rate(cooling_rate, pe_rate, g0_table, bypass=None):
    """GFM_CoolingRate plus the photoelectric heating it does not contain (see FIELD_REGISTRY)."""
    if not g0_table:
        return cooling_rate + pe_rate
    if bypass is None:
        return cooling_rate
    return cooling_rate + pe_rate * (bypass != 0)


# ------------------------------------------------------------------ numpy recipes
# Each recipe takes (p, region): p is the Particles set (p.values(name) gives any field in lyratools units,
# reading it if needed), region gives z, t_now, g0_table. Results are in the unit listed.
def _X(p):
    return p.column("GFM_Metals", 0).astype(np.float64) if p.has("GFM_Metals") else HYDROGEN_MASSFRAC


def _xe(p, X):
    return np.asarray(p["ElectronAbundance"], np.float64) if p.has("ElectronAbundance") else 1.0 + (1 - X) / (4 * X)


def _nH(p, reg):
    return _X(p) * p.values("Density") / PROTONMASS


def _temperature(p, reg):
    """mu = 4 / (1 + 3X + 4X x_e) (atomic H and He); only used when the snapshot stores no Temperature."""
    X = _X(p)
    return chemistry.temperature(p.values("InternalEnergy"), _xe(p, X), X)


def _pressure_k(p, reg):
    return GAMMA_MINUS1 * p.values("Density") * p.values("InternalEnergy") / BOLTZMANN


def _cooling_time(p, reg):
    return chemistry.cooling_time(p.values("InternalEnergy"), p.values("Density"),
                                  np.asarray(p["GFM_CoolingRate"], np.float64), _X(p))


def _self_shielding(p, reg):
    return chemistry.self_shielding_factor(p.values("nH"), reg.z)


def _net_true(p, reg):
    bypass = None
    if reg.g0_table:
        for name in ("G0TableDebugBypass", "G0TableBypass"):
            if p.has(name):
                bypass = np.asarray(p[name])
                break
    return net_true_rate(np.asarray(p["GFM_CoolingRate"], np.float64), np.asarray(p["FUV_PEHeatingRate"], np.float64),
                         reg.g0_table, bypass)


def _sntime(p):
    return np.asarray(p["SNTime"], np.float64)


DERIVED = {
    "PartType0": {
        "nH": (_nH, "cm**-3"),
        "Temperature": (_temperature, "K"),
        "Pressure_k": (_pressure_k, "K/cm**3"),
        "CoolingTime": (_cooling_time, "Myr"),
        "SelfShieldingFactor": (_self_shielding, ""),
        "net_true": (_net_true, "erg*cm**3/s"),
    },
    "PartType4": {
        "FormationTime": (lambda p, reg: np.asarray(p["tform"], np.float64), "Myr"),
        "Age": (lambda p, reg: np.asarray(p["age"], np.float64), "Myr"),
        "SNState": (lambda p, reg: stellar_evolution.sn_state(_sntime(p)), ""),
        "DeathTime": (lambda p, reg: np.asarray(p["t_death"], np.float64), "Myr"),
        "ESN": (lambda p, reg: stellar_evolution.sn_energies(np.asarray(p["m_init"], np.float64), _sntime(p)), "erg"),
    },
}
for _pt in ("PartType0", "PartType4"):
    DERIVED[_pt]["Z_Zsun"] = (lambda p, reg: np.asarray(p["GFM_Metallicity"], np.float64) / ZSOLAR, "")
    for _e1, _e2 in RATIOS:
        DERIVED[_pt][f"{_e1}_{_e2}"] = (lambda p, reg, e1=_e1, e2=_e2: chemistry.abundance_ratio(np.asarray(p["GFM_Metals"]), e1, e2), "")
    DERIVED[_pt]["12logOH"] = (lambda p, reg: chemistry.twelve_log_oh_metals(np.asarray(p["GFM_Metals"])), "")


def derived(pt, name):
    """(function, unit) of a derived field of particle type pt ('PartType0'/'PartType4'), or None."""
    return DERIVED.get(pt, {}).get(name)


def add_fields(ds, overwrite=True):
    """Register the lyratools derived fields as scida recipes on a scida dataset (requires scida)."""
    from .units import require_scida
    require_scida("lyratools.add_fields")
    from .scida_fields import add_fields as _add
    return _add(ds, overwrite=overwrite)


def load_scida(path, catalog=None, units=True, **kwargs):
    """scida.load of a LYRA snapshot with the code units of its header (requires scida);
    see lyratools.scida_fields.load_scida."""
    from .units import require_scida
    require_scida("lyratools.load_scida")
    from .scida_fields import load_scida as _load
    return _load(path, catalog=catalog, units=units, **kwargs)
