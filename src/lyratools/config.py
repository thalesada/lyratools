"""
Location of the external data tables lyratools uses (stellar yields, isochrones,
self-shielding fits, observational catalogs).

Point lyratools at one directory holding all tables:

    import lyratools
    lyratools.set_data_dir("/path/to/lyra_data")

or set the environment variable LYRATOOLS_DATA. Individual files that live
elsewhere can be set with `set_data_file(key, path)`.

DATA_FILES lists, for every key, the file name inside the data directory, the reference to cite,
where the file comes from and, for public tables that may be redistributed, a download URL
(`python scripts/fetch_data.py DIR` fetches those). `lyratools.config.describe()` prints the list.
"""
import glob
import os
from typing import NamedTuple, Optional


class DataFile(NamedTuple):
    path: str                        # file name / glob pattern relative to the data directory
    reference: str                   # what to cite
    source: str                      # where to get it
    url: Optional[str] = None        # direct download (public, redistributable) or None
    archive_member: Optional[str] = None   # url is an archive: `path` is extracted from it
    md5: Optional[str] = None
    size: Optional[int] = None       # bytes, for the download prompt
    release: str = "first"           # "later": only used by later-release modules (lyratools._release)

    def __str__(self):               # DATA_FILES[key] used to be the plain file name
        return self.path

    def __fspath__(self):
        return self.path


_LYRA = ("LYRA / AREPO input table (Gutcke et al. 2021, MNRAS 501, 5597); distributed with the LYRA "
         "simulation code, not public: ask the LYRA team")
_PARSEC = ("PARSEC isochrones (Bressan et al. 2012) from the CMD web interface "
           "http://stev.oapd.inaf.it/cgi-bin/cmd, one file per [M/H] and age bin; no direct download")

DATA_FILES = {
    "sukhbold": DataFile("StellarEvolution_Sukhbold2015.txt",
                         "Sukhbold et al. 2016, ApJ 821, 38 (LYRA format)", _LYRA),
    "karakas": DataFile("StellarEvolution_Karakas.txt", "Karakas 2010, MNRAS 403, 1413 (LYRA format)", _LYRA),
    "lc18_br26_v000": DataFile("StellarEvolution_LC18_BR26_v000.txt",
                               "Limongi & Chieffi 2018, ApJS 237, 13; Boccioli & Roberti (2026)",
                               "STEM tables of the LYRA team (not public yet)", release="later"),
    "lc18_br26_v150": DataFile("StellarEvolution_LC18_BR26_v150.txt",
                               "Limongi & Chieffi 2018, ApJS 237, 13; Boccioli & Roberti (2026)",
                               "STEM tables of the LYRA team (not public yet)", release="later"),
    "lc18_br26_v300": DataFile("StellarEvolution_LC18_BR26_v300.txt",
                               "Limongi & Chieffi 2018, ApJS 237, 13; Boccioli & Roberti (2026)",
                               "STEM tables of the LYRA team (not public yet)", release="later"),
    "solar": DataFile("solar_composition_asplund_ARAA09.txt", "Asplund et al. 2009, ARA&A 47, 481",
                      "LYRA / AREPO utilities (table of Asplund et al. 2009 solar abundances)"),
    "rahmati": DataFile("SelfShielding_Rahmati12", "Rahmati et al. 2013, MNRAS 430, 2427",
                        "table of the Rahmati et al. (2013) self-shielding fit parameters (LYRA team)"),
    "ion_table": DataFile("UVB_dust0_CR0_G0_shield0.hdf5", "Ploeckinger & Schaye 2020, MNRAS 497, 4857",
                          "Harvard Dataverse doi:10.7910/DVN/GR3L5N (CC0)",
                          url="https://dataverse.harvard.edu/api/access/datafile/3985212",
                          md5="369c8cbd6a608b9f008c5fa4ffcf9e10", size=2158901614),
    "parsec_optical": DataFile("MH_?_*/agebin0.03*", "Bressan et al. 2012, MNRAS 427, 127", _PARSEC),
    "parsec_jwst": DataFile("JWST_NirCam_filters/MH_?_*/*.dat", "Bressan et al. 2012, MNRAS 427, 127",
                            _PARSEC + " (JWST NIRCam photometric system)"),
    "jwst_zeropoints": DataFile("JWST_NirCam_filters/NRC_ZPs_1126pmap.txt",
                                "JWST NIRCam absolute flux calibration, pmap 1126 (STScI)",
                                "JDox 'NIRCam Absolute Flux Calibration and Zeropoints', previous deliveries: "
                                "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/"
                                "nircam-absolute-flux-calibration-and-zeropoints"),
    "ks_data": DataFile("KS.txt", "Roychowdhury et al. 2015, MNRAS 449, 3700 (their FIGGS, THINGS and Bluedisk points)",
                        "table transcribed by the LYRA team from Roychowdhury et al. (2015)"),
    "mcconnachie": DataFile("NearbyGalaxies_Jan2021_PUBLIC.fits", "McConnachie 2012, AJ 144, 4 (Jan 2021 update)",
                            "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/community/nearby/",
                            url="https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/PANDAS/NearbyGals/"
                                "NearbyGalaxies_Jan2021_PUBLIC.fits"),
    "behroozi": DataFile("release-sfh_z0_z8_052913/smmr/c_smmr_then_z0.10_red_all_smf_m1p1s1_bolshoi_fullcosmos_ms.dat",
                         "Behroozi, Wechsler & Conroy 2013, ApJ 770, 57 and ApJ 762, L31",
                         "https://www.peterbehroozi.com/data.html (release of 5/29/2013)",
                         url="https://www.peterbehroozi.com/uploads/6/5/4/8/6548418/sfh_z0_z8.tar.gz",
                         archive_member="release-sfh_z0_z8_052913"),
}

_data_dir = None
_overrides = {}


def set_data_dir(path):
    """Directory containing the files in DATA_FILES."""
    global _data_dir
    _data_dir = os.path.expanduser(path)


def set_data_file(key, path):
    """Use a specific path for one table (glob patterns allowed for the PARSEC entries)."""
    if key not in DATA_FILES:
        raise KeyError(f"unknown data key '{key}'; known: {sorted(DATA_FILES)}")
    _overrides[key] = os.path.expanduser(path)


def where_to_get(key):
    """One-paragraph description of a data table: file name, reference, source and download."""
    d = DATA_FILES[key]
    s = f"  file:      {d.path}\n  reference: {d.reference}\n  source:    {d.source}"
    if d.url:
        s += f"\n  download:  python scripts/fetch_data.py <data dir> --only {key}   ({d.url})"
    return s


def describe():
    """Print every data table with its reference and source."""
    for key in DATA_FILES:
        print(f"{key}:\n{where_to_get(key)}")


def data_path(key):
    """Path (or glob pattern) of a data table, with a helpful error if it is missing."""
    if key in _overrides:
        return _overrides[key]
    if key not in DATA_FILES:
        raise KeyError(f"unknown data key '{key}'; known: {sorted(DATA_FILES)}")
    base = _data_dir or os.environ.get("LYRATOOLS_DATA")
    if base is None:
        raise FileNotFoundError(
            f"No data directory set for '{key}'. Call lyratools.set_data_dir(path), "
            f"set LYRATOOLS_DATA, or lyratools.set_data_file('{key}', path).\n{where_to_get(key)}")
    path = os.path.join(base, DATA_FILES[key].path)
    if any(c in path for c in "*?["):
        if not glob.glob(path):
            raise FileNotFoundError(f"'{key}': no files match {path}\n{where_to_get(key)}")
    elif not os.path.exists(path):
        raise FileNotFoundError(f"'{key}' table not found at {path}\n{where_to_get(key)}")
    return path
