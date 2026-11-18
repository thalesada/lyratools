#!/usr/bin/env python
"""
Time series of the main progenitor and its substructure for a LYRA zoom (lyratools.timeseries).

    python scripts/lyra_timeseries.py --snapdir D --groupdir D --out D [--halo 0] [--snaps a:b]
        [--nproc auto] [--config cfg.json] [--redo] [--dry-run] [--combine-only] [--cleanup]
"""
import sys

from lyratools.timeseries.run import main

if __name__ == "__main__":      # required: worker processes are spawned and re-import this file
    sys.exit(main())
