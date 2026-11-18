"""
Parts of lyratools intended for a later public release.

They work and are tested, but are not meant for the first public tag (validation or companion
publications pending). scripts/make_release.py copies the repository without them; lyratools and its
tests run either way (missing later-release modules are skipped, `high_mass_table="LC18"` then raises).
Each listed module also sets `__release__ = "later"`.
"""

LATER_RELEASE = {
    # path relative to the repository root: what it is
    "src/lyratools/fuv.py": "FUV field (G0) as AREPO's fuv.c computes it",
    "src/lyratools/cooling.py": "LYRA metal / G0 cooling-table interpolation",
    "src/lyratools/yields_lc18.py": "LC18 + BR26 massive-star yields (YieldCorrection(high_mass_table='LC18'))",
    "tests/test_fuv.py": "tests of lyratools.fuv",
    "tests/test_cooling.py": "tests of lyratools.cooling",
    "tests/check_fuv_snapshot.py": "FUV_G0 check on a real snapshot",
}
