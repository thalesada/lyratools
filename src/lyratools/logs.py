"""Parsers for AREPO/LYRA log files: SN injections, PopIII enrichment, CPU usage, particle counts."""
import glob
import os
import re

import numpy as np

from .constants import CODE_UNITS_COSMOLOGICAL, CODE_UNITS_ISOLATED, KPC, PROTONMASS, HYDROGEN_MASSFRAC

NUMBER_PATTERN = r"[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)|-?\d+(?:\.\d+)?"
SN_FILE = "supernovae.txt"
ZDEP_COLUMNS = re.compile(r",\s*x=([^,\s]+),\s*interp=(-?\d+),\s*vrot=(-?\d+),\s*fate=(-?\d+),\s*efate=(-?\d+)")


def floats_from_string(string):
    return np.array([float(v) for v in re.findall(NUMBER_PATTERN, string)], dtype=float)


def _logfiles(simpath):
    files = glob.glob(os.path.join(simpath, "logs", "log.out.*"))
    files.sort(key=os.path.getmtime)
    return files


def start_logfile(simpath):
    """Index of the last log file (by modification time) that contains a READIC (run start)."""
    start = 0
    for k, f in enumerate(_logfiles(simpath)):
        with open(f, "r", errors="ignore") as fh:
            if "READIC" in fh.read():
                start = k
    return start


class SNLogData:
    """
    SN and mass injection events. Read from output/supernovae.txt when it exists (source="auto" or "file"; runs
    from Nov 2026 on, which also have StarID as int, cellID, task, Minit, Mej [Msun], Zbirth, XObirth, nH_arepo, mcell),
    otherwise from the 'Injecting SN' / 'Injecting mass' log lines (source="logs").
    Attributes: rhoSN [cm^-3, hydrogen], posSN [kpc], velSN, ESN [1e51 erg], Mstar [Msun], StarID, timeSN
    (scale factor in cosmological runs, code time minus 0.2 in isolated runs).
    Runs with VIRGODWARF_SNII_ZDEP end each line with ', x=%g, interp=%d, vrot=%d, fate=%d, efate=%d':
    x = log(X_O/X_O,sun) of the lookup (clamped to the table), interp (1 interpolated between the two metallicity bins,
    0 nearer bin), vrot [km/s], fate of the yields and efate of the energy (1 explodes, 0 collapses; they differ only
    with VIRGODWARF_SNII_ZDEP_SUKHBOLD_ENERGY, flagged in fate_mismatch). x = NaN and -1 for stars without a lookup
    and for runs without the columns.
    """

    def __init__(self, simpath, start=0, cosmo=True, code_units=None, source="auto"):
        cu = code_units or (CODE_UNITS_COSMOLOGICAL if cosmo else CODE_UNITS_ISOLATED)
        unit_energy = cu["mass_g"] * cu["velocity_cms"] ** 2
        convert_rho = cu["mass_g"] / cu["length_cm"] ** 3 / PROTONMASS
        convert_pos = cu["length_cm"] / KPC
        self.simpath = simpath
        path = os.path.join(simpath, "output", SN_FILE)
        if source == "file" or (source == "auto" and os.path.exists(path)):
            self._read_file(path, cosmo, convert_rho, convert_pos)
            return
        rows, zdep = [], []
        for f in _logfiles(simpath)[start:]:
            with open(f, "r", errors="ignore") as fh:
                for line in fh:
                    extra = ZDEP_COLUMNS.search(line)
                    if extra:
                        line = line[:extra.start()]
                    if "Injecting SN " in line:
                        arr = floats_from_string(line)
                        if len(arr) == 12:
                            arr = np.insert(arr, 3, 0.0)
                    elif "Injecting mass " in line:
                        arr = np.insert(floats_from_string(line), 1, 0.0)
                        if len(arr) == 12:
                            arr = np.insert(arr, 3, 0.0)
                    else:
                        continue
                    if len(arr) == 13:
                        rows.append(arr)
                        zdep.append([float(v) for v in extra.groups()] if extra else [np.nan, -1, -1, -1, -1])
        arr = np.array(rows).reshape(-1, 13)
        z = np.array(zdep, dtype=float).reshape(-1, 5)
        self.x = z[:, 0]
        self.interp, self.vrot, self.fate, self.efate = z[:, 1:].astype(int).T
        self.fate_mismatch = (self.fate >= 0) & (self.efate >= 0) & (self.fate != self.efate)
        time = arr[:, 12]
        self.ESN = arr[:, 1] * unit_energy / 1e51
        self.StarID = arr[:, 2]
        self.Mstar = arr[:, 3] * cu["mass_g"] / 1.98841e33
        self.velSN = arr[:, 9:12]
        if cosmo:
            self.rhoSN = arr[:, 5] / time ** 3 * convert_rho * HYDROGEN_MASSFRAC
            self.posSN = arr[:, 6:9] * time[:, None] * convert_pos
            self.timeSN = time
        else:
            self.rhoSN = arr[:, 5] * convert_rho * HYDROGEN_MASSFRAC
            self.posSN = arr[:, 6:9] * convert_pos - 5000
            self.timeSN = time - 0.2

    def _read_file(self, path, cosmo, convert_rho, convert_pos):
        """output/supernovae.txt, one line per injection written by task 0. After a restart from a restart file the
        repeated steps append their injections again; each star dies once, so only its last record is kept."""
        d = np.loadtxt(path, comments="#", ndmin=2)
        ids = np.loadtxt(path, comments="#", usecols=(1, 2), dtype=np.int64, ndmin=2)
        last = np.sort(len(ids) - 1 - np.unique(ids[::-1, 0], return_index=True)[1])
        d, ids = d[last], ids[last]
        time = d[:, 0]
        self.StarID, self.cellID, self.task = ids[:, 0], ids[:, 1], d[:, 3].astype(int)
        self.ESN = d[:, 4] / 1e51
        self.Minit, self.Mej, self.Zbirth, self.XObirth = d[:, 5], d[:, 6], d[:, 7], d[:, 8]
        self.Mstar = np.full(len(d), np.nan)          # star particle mass, only in the stdout lines
        self.nH_arepo = d[:, 10]                      # n_H as Arepo computes it (UnitDensity_in_cgs h^2 / a^3)
        self.mcell = d[:, 11]
        self.velSN = d[:, 15:18]
        self.x = d[:, 18]
        self.interp, self.vrot, self.fate, self.efate = d[:, 19:23].astype(int).T
        self.fate_mismatch = (self.fate >= 0) & (self.efate >= 0) & (self.fate != self.efate)
        if cosmo:                                     # same conventions as the stdout parser
            self.rhoSN = d[:, 9] / time ** 3 * convert_rho * HYDROGEN_MASSFRAC
            self.posSN = d[:, 12:15] * time[:, None] * convert_pos
            self.timeSN = time
        else:
            self.rhoSN = d[:, 9] * convert_rho * HYDROGEN_MASSFRAC
            self.posSN = d[:, 12:15] * convert_pos - 5000
            self.timeSN = time - 0.2


class PopIIILogData:
    """PopIII enrichment events ('Mcrit200' lines) since the last READIC: GrNr, Mhalo [Msun], Zhalo, z."""

    def __init__(self, simpath, hubbleparam=0.6777, mass_unit_msun=1e10, verbose=False):
        rows = []
        for f in _logfiles(simpath)[start_logfile(simpath):]:
            with open(f, "r", errors="ignore") as fh:
                for line in fh:
                    if "Mcrit200" in line:
                        arr = floats_from_string(line)
                        if verbose:
                            print(arr)
                        if len(arr) == 6:
                            rows.append(arr)
        p = np.array(rows).reshape(-1, 6).T
        self.task, self.GrNr = p[0], p[1]
        self.Mhalo = p[3] * mass_unit_msun / hubbleparam
        self.Zhalo, self.z = p[4], p[5]
        self.nhalos = len(self.Mhalo)


def nparticles_ic(simpath, start=None):
    """Particle numbers per type from the READIC lines of the starting log file."""
    start = start_logfile(simpath) if start is None else start
    npart = np.zeros(6)
    with open(_logfiles(simpath)[start], "r", errors="ignore") as fh:
        for line in fh:
            for j in range(6):
                if f"READIC: Type {j}" in line:
                    npart[j] = floats_from_string(line)[2]
    return npart


def nparticles_timebins(simpath):
    """Total active particle number at each 'Occupied timebins' block of timebins.txt."""
    npart, flag = [], False
    with open(os.path.join(simpath, "output", "timebins.txt"), "r") as fh:
        for line in fh:
            if flag:
                npart.append(floats_from_string(line)[4])
                flag = False
            elif "Occupied timebins" in line:
                flag = True
    return np.array(npart)


class CPUData:
    """Wall-clock accounting from cpu.txt: simulation_time, seconds_running, CPUhours, treegrav, voronoi, hydro, domain."""

    def __init__(self, filepath):
        self.filepath = filepath
        time, timebin, total, tree, vor, hyd, dom = [], [], [], [], [], [], []
        ncpu = 0
        with open(filepath, "r") as fh:
            for line in fh:
                if "Time" in line:
                    parts = re.split(r"\s|,", line)
                    time.append(float(parts[4]))
                    if ncpu == 0:
                        ncpu = int(parts[7])
                    timebin.append(int(re.split(r", |: |\n| ", line)[9]))
                elif "total" in line:
                    total.append(float(line.split()[3]))
                elif "treegrav" in line:
                    tree.append(float(line.split()[3]))
                elif "voronoi" in line:
                    vor.append(float(line.split()[3]))
                elif "hydro" in line:
                    hyd.append(float(line.split()[3]))
                elif "domain" in line:
                    dom.append(float(line.split()[3]))
        self.highest_timebin = np.array(timebin)
        self.seconds_running = np.array(total)
        self.treegrav, self.voronoi = np.array(tree), np.array(vor)
        self.hydro, self.domain = np.array(hyd), np.array(dom)
        self.simulation_time = np.array(time)
        self.n_cores = ncpu
        self.CPUhours = self.seconds_running * ncpu / 3600
        self.dt = np.diff(np.concatenate([[0], self.seconds_running]))
        top = self.seconds_running[self.highest_timebin == self.highest_timebin.max()] if len(timebin) else np.array([])
        self.dt_global = np.diff(top)
