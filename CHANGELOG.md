# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Packaging for a public release: `pyproject.toml` metadata (Python >= 3.10, MIT), `LICENSE`, `CITATION.cff`,
  this changelog, `lyra-timeseries` console script.
- `lyratools.config.DATA_FILES` entries are `DataFile` records with reference, source, download URL (public,
  redistributable tables) and MD5; missing-file errors print where to get the file; `lt.config.describe()`.
- `scripts/fetch_data.py`: downloads the public tables (McConnachie 2012 catalogue, Behroozi et al. 2013
  release, Ploeckinger & Schaye 2020 table with `--large`) and lists the others with their sources.
- Later-release marking (`lyratools/_release.py`, `RELEASE.md`, `scripts/make_release.py`): `fuv`, `cooling`
  and the LC18 yields option (now `lyratools/yields_lc18.py`) can be left out of a public tag.
- GitHub Actions: `tests/run_tests.py` on Python 3.10-3.12, with and without scida.
- `lt.load_scida(path, catalog=None)`: `scida.load` of a LYRA snapshot and its catalogue with the code units
  of the header (`lyratools.scida_fields.code_unit_hints`).

### Fixed
- scida fields of LYRA snapshots: scida's default unit file assumes `code_length = ckpc/h`, LYRA uses Mpc, so
  `add_fields` on a plain `scida.load` gave `nH` 1e9 too large on a fresh install. `add_fields` now checks
  `code_length` against the header and points to `lt.load_scida`.
- `tests/run_tests.py` uses the fork start method on POSIX, so scida's process pool works on macOS.

### Changed
- scida is an optional extra (`pip install "lyratools[scida]"`); scida/dask are imported only by
  `lt.add_fields` and `photometry.add_magnitude_fields`, which now say how to install them.
- `tests/run_tests.py` exits non-zero when any test file fails, skips test files of later-release modules
  that are not present, and reports the real-snapshot check as skipped when its environment variables are unset.
- Removed the unused `tables` (pandas) extra.

## [0.1.0]
- Port of `TTT.py` (loadmodules) to an installable package; h5py snapshot reader; Region field access;
  time-series pipeline; isolated-run support; yields correction; fuv and cooling reproductions.
