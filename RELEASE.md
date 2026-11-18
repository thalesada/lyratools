# Release notes for maintainers

## Later-release parts

These modules work and are tested but are not meant for the first public tag. They are listed in
`src/lyratools/_release.py` (`LATER_RELEASE`), and each sets `__release__ = "later"`:

| Path | What |
|---|---|
| `src/lyratools/fuv.py` | FUV field (G0) as AREPO's `fuv.c` computes it |
| `src/lyratools/cooling.py` | LYRA metal / G0 cooling-table interpolation |
| `src/lyratools/yields_lc18.py` | LC18 + BR26 massive-star yields (`YieldCorrection(high_mass_table="LC18")`) |
| `tests/test_fuv.py`, `tests/test_cooling.py`, `tests/check_fuv_snapshot.py` | their tests |

Without them, `import lyratools` works (`lt.fuv` / `lt.cooling` are simply absent),
`high_mass_table="LC18"` raises a `ValueError`, and `tests/run_tests.py` skips their tests. The data keys
`lc18_br26_v*` are marked `release="later"` (`fetch_data.py --first-release` skips them).

Not removed automatically (edit by hand for the public tag if wanted): the README sections on `fuv`,
`cooling`, the LC18 option, and the FUV snapshot field descriptions in `fields.FIELD_REGISTRY` (those
describe AREPO output and need no lyratools module).

## Making the first public tag

    python scripts/make_release.py ../lyratools-public     # copy without the later-release files
    cd ../lyratools-public && pip install . && python tests/run_tests.py

## Before tagging

- Fill in authors in `CITATION.cff` and `pyproject.toml` (TODOs), the repository URL, and the version
  (`src/lyratools/_version.py`, `CITATION.cff`, `CHANGELOG.md`).
- Check the data-table sources in `lyratools/config.py` (`lt.config.describe()`): the LYRA tables
  (Sukhbold, Karakas, solar, Rahmati, KS compilation) are marked "not public: ask the LYRA team".
