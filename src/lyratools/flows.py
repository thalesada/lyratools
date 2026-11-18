"""Gas flow rates through spherical shells or slabs, and mass/metal/energy loading factors."""
import numpy as np

from .constants import KMS_PER_KPC_TO_PER_YR, MSUN, SN_MIN_MASS
from .stellar_evolution import died_within, sn_energies, sn_metal_return, agb_returns


def _rates(sel, speed, mass, thickness, Z=None, energy=None):
    out = dict(mass=np.sum(mass[sel] * speed[sel]) / thickness * KMS_PER_KPC_TO_PER_YR)
    if Z is not None:
        out["metals"] = np.sum(Z[sel] * mass[sel] * speed[sel]) / thickness * KMS_PER_KPC_TO_PER_YR
    if energy is not None:
        out["energy"] = np.sum(energy[sel] * speed[sel]) / thickness * KMS_PER_KPC_TO_PER_YR
    return out


def _specific_energy(vel, u):
    """Kinetic + thermal energy per unit mass [erg/g] (vel km/s, u cm^2/s^2)."""
    return 0.5 * np.sum((vel * 1e5) ** 2, axis=1) + (0 if u is None else np.asarray(u, float))


def spherical_flow_rate(pos, vel, mass, radius, thickness, vcut=0.0, Z=None, u=None, inflow=True):
    """
    Outflow (and inflow) rates through a spherical shell of radius and thickness [kpc] centred
    on the origin. pos [kpc], vel [km/s] (bulk motion removed), mass [Msun], Z metal mass
    fraction, u specific internal energy [cm^2/s^2]. Gas with |v_r| <= vcut is ignored.
    Returns {'out': {...}, 'in': {...}} with mass and metal rates [Msun/yr] and energy rate [erg/yr].
    """
    r = np.linalg.norm(pos, axis=1)
    vr = np.einsum("ij,ij->i", pos, vel) / np.where(r > 0, r, np.inf)
    shell = np.abs(r - radius) < thickness / 2
    energy = _specific_energy(vel, u) * mass * MSUN if (u is not None) else None
    out = {"out": _rates(shell & (vr > vcut), np.abs(vr), mass, thickness, Z, energy)}
    if inflow:
        out["in"] = _rates(shell & (vr < -vcut), np.abs(vr), mass, thickness, Z, energy)
    return out


def disk_flow_rate(pos, vel, mass, height, thickness, Z=None, u=None, inflow=True, radius=None):
    """
    Flow rates through two slabs at +-height [kpc] above and below a disc in the x-y plane.
    Optionally restricted to cylindrical radius < radius [kpc]. Same returns as spherical_flow_rate.
    """
    z, vz = pos[:, 2], vel[:, 2]
    upper = np.abs(z - height) < thickness / 2
    lower = np.abs(z + height) < thickness / 2
    if radius is not None:
        inr = np.hypot(pos[:, 0], pos[:, 1]) < radius
        upper, lower = upper & inr, lower & inr
    energy = _specific_energy(vel, u) * mass * MSUN if (u is not None) else None
    outflowing = (upper & (vz > 0)) | (lower & (vz < 0))
    res = {"out": _rates(outflowing, np.abs(vz), mass, thickness, Z, energy)}
    if inflow:
        inflowing = (upper & (vz < 0)) | (lower & (vz > 0))
        res["in"] = _rates(inflowing, np.abs(vz), mass, thickness, Z, energy)
    return res


def loading_factors(gas, stars, t_now, radius, thickness=0.1, dt=100.0, vcut=0.0, return_rates=False):
    """
    Mass, metal and energy loading factors through a spherical shell (cosmological runs).

    gas:   dict with pos, vel, mass, Z (metal mass fraction), u [cm^2/s^2]
    stars: dict with tform [Myr], m_init [Msun], mass [Msun], Z, sntime (raw LYRA SNTime)
    SFR, metal production and SN energy injection are averaged over the last dt [Myr].
    """
    flows = spherical_flow_rate(gas["pos"], gas["vel"], gas["mass"], radius, thickness, vcut,
                                Z=gas.get("Z"), u=gas.get("u"), inflow=False)["out"]
    young = stars["tform"] > t_now - dt
    sfr = np.sum(stars["m_init"][young]) / (dt * 1e6)

    sn = died_within(stars["tform"], stars["sntime"], stars["m_init"], t_now, dt, massive=True)
    agb = died_within(stars["tform"], stars["sntime"], stars["m_init"], t_now, dt, massive=False)
    mz_sn = np.sum(sn_metal_return(stars["m_init"][sn])) if sn.any() else 0.0
    mz_agb = np.sum(agb_returns(stars["m_init"][agb], stars["Z"][agb])[1]) if agb.any() else 0.0
    mz_rate = (mz_sn + mz_agb) / (dt * 1e6)
    e_rate = np.sum(sn_energies(stars["m_init"][sn], stars["sntime"][sn])) / (dt * 1e6)

    if return_rates:
        return dict(outflow_rate=flows["mass"], sfr=sfr, metal_outflow_rate=flows.get("metals"),
                    metal_production_rate=mz_rate, energy_outflow_rate=flows.get("energy"),
                    energy_injection_rate=e_rate)
    div = lambda a, b: a / b if (a is not None and b > 0) else 0.0
    return dict(mass=div(flows["mass"], sfr), metals=div(flows.get("metals"), mz_rate),
                energy=div(flows.get("energy"), e_rate))


def loading_factors_isolated(gas, stars, time, height=2.0, thickness=0.05, dt=10.0, initial_metal_mass=6.658e-07 * 1e10,
                             sn_energy_mode="table", return_rates=False):
    """
    Loading factors for idealised (non-cosmological) disc runs, measured in slabs above/below the disc.

    gas:   dict with pos, vel, mass, Z, u
    stars: dict with tform [Myr], m_init, mass, Z, sntime (raw SNTime)
    time:  current simulation time [Myr]
    initial_metal_mass: total metal mass [Msun] in the initial conditions
    sn_energy_mode: 'table' (Sukhbold energies, failed SNe 1e49 erg), 'fixed' (1e51 erg each),
                    or 'fixed_winds' (table, but 1e51 erg for every non-failed SN)
    As in the original analysis, the energy injection rate uses all exploded SNe divided by dt.
    """
    young = stars["tform"] > time - dt
    sfr = np.sum(stars["m_init"][young]) / (dt * 1e6)
    flows = disk_flow_rate(gas["pos"], gas["vel"], gas["mass"], height, thickness, Z=gas["Z"], u=gas.get("u"), inflow=False)["out"]

    mz_produced = np.sum(gas["mass"] * gas["Z"]) + np.sum(stars["mass"] * stars["Z"]) - initial_metal_mass
    mstar = np.sum(stars["m_init"])

    s = np.asarray(stars["sntime"], float)
    sn = (s != -1) & (s < 0) & (stars["m_init"] > SN_MIN_MASS)
    E = np.zeros(len(s))
    if sn_energy_mode == "fixed":
        E[sn] = 1e51
    else:
        e = sn_energies(stars["m_init"][sn], s[sn])
        if sn_energy_mode == "fixed_winds":
            e = np.where(e > 1e49, 1e51, e)
        E[sn] = e
    e_rate = E.sum() / (dt * 1e6)

    mlf = flows["mass"] / sfr if sfr > 0 else 0.0
    mlz = flows["metals"] / mz_produced / sfr * mstar if (sfr > 0 and mz_produced > 0) else 0.0
    mle = flows.get("energy", 0.0) / e_rate if e_rate > 0 else 0.0
    if return_rates:
        return dict(mass=mlf, metals=mlz, energy=mle, outflow_rate=flows["mass"],
                    metal_outflow_rate=flows["metals"], energy_outflow_rate=flows.get("energy"))
    return dict(mass=mlf, metals=mlz, energy=mle)
