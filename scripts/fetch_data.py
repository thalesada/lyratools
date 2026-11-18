"""
Download the public, redistributable data tables lyratools uses into a data directory, and list the
others with their references and sources (lyratools.config.DATA_FILES).

    python scripts/fetch_data.py ~/lyra_data                 # small public tables; lists the rest
    python scripts/fetch_data.py ~/lyra_data --large         # also the 2.2 GB Ploeckinger & Schaye table
    python scripts/fetch_data.py ~/lyra_data --only mcconnachie
    python scripts/fetch_data.py ~/lyra_data --dry-run       # show what would be downloaded

Then: lyratools.set_data_dir("~/lyra_data") or export LYRATOOLS_DATA=~/lyra_data.
Files already present are kept (checked against the MD5 when one is known).
Uses only the standard library and lyratools.config (no network access at import).
"""
import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

try:
    from lyratools.config import DATA_FILES
except ImportError:                                  # running from a source tree without installing
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from lyratools.config import DATA_FILES

LARGE = 200 * 1024 ** 2                              # bytes: needs --large


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dest):
    """Stream url to dest (via a temporary file in the same directory), printing progress."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "lyratools-fetch-data"})
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest) or ".", suffix=".part")
    try:
        with urllib.request.urlopen(req) as r, os.fdopen(fd, "wb") as out:
            total, done = int(r.headers.get("Content-Length") or 0), 0
            for b in iter(lambda: r.read(1 << 20), b""):
                out.write(b)
                done += len(b)
                if total:
                    print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def extract_member(archive, member, data_dir):
    """Extract the directory/file `member` of a tar archive into data_dir (safe paths only)."""
    with tarfile.open(archive) as t:
        names = [m for m in t.getmembers() if m.name == member or m.name.startswith(member.rstrip("/") + "/")]
        if not names:
            raise RuntimeError(f"{archive} has no member '{member}'; it contains e.g. {t.getnames()[:5]}")
        root = os.path.realpath(data_dir)
        for m in names:
            target = os.path.realpath(os.path.join(data_dir, m.name))
            if not target.startswith(root + os.sep) or m.issym() or m.islnk():
                raise RuntimeError(f"refusing unsafe archive member {m.name}")
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        t.extractall(data_dir, members=names, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="target directory (later: lyratools.set_data_dir / LYRATOOLS_DATA)")
    ap.add_argument("--only", nargs="+", metavar="KEY", help=f"subset of: {', '.join(DATA_FILES)}")
    ap.add_argument("--large", action="store_true", help="also download tables larger than 200 MB")
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded, download nothing")
    ap.add_argument("--first-release", action="store_true", help="skip tables only used by later-release modules")
    a = ap.parse_args(argv)

    keys = a.only or list(DATA_FILES)
    unknown = sorted(set(keys) - set(DATA_FILES))
    if unknown:
        ap.error(f"unknown keys {unknown}; known: {list(DATA_FILES)}")
    data_dir = os.path.expanduser(a.data_dir)
    manual, failed = [], []
    for key in keys:
        d = DATA_FILES[key]
        if a.first_release and d.release != "first":
            continue
        dest = os.path.join(data_dir, d.archive_member or d.path)
        if d.url is None:
            manual.append(key)
            continue
        if os.path.exists(dest) and (d.md5 is None or os.path.isdir(dest) or md5sum(dest) == d.md5):
            print(f"[have] {key}: {dest}")
            continue
        if d.size and d.size > LARGE and not a.large:
            print(f"[skip] {key}: {d.size / 1e9:.1f} GB, rerun with --large ({d.url})")
            continue
        size = f" ({d.size / 1e9:.2f} GB)" if d.size else ""
        print(f"[get]  {key}{size}: {d.url}\n       -> {dest}")
        if a.dry_run:
            continue
        try:
            if d.archive_member:
                with tempfile.TemporaryDirectory(dir=data_dir if os.path.isdir(data_dir) else None) as tmp:
                    arc = os.path.join(tmp, os.path.basename(d.url))
                    download(d.url, arc)
                    extract_member(arc, d.archive_member, data_dir)
                if not os.path.exists(os.path.join(data_dir, d.path)):
                    raise RuntimeError(f"{d.path} not found after extracting {d.archive_member}")
            else:
                download(d.url, dest)
                if d.md5 and md5sum(dest) != d.md5:
                    shutil.move(dest, dest + ".bad_md5")
                    raise RuntimeError(f"MD5 mismatch, kept as {dest}.bad_md5")
        except Exception as e:                           # report and continue with the other tables
            print(f"[fail] {key}: {e}")
            failed.append(key)

    if manual:
        print("\nNot downloadable automatically (place them in the data directory under the given name):")
        for key in manual:
            d = DATA_FILES[key]
            later = "   [later release]" if d.release != "first" else ""
            print(f"  {key}{later}\n    file:      {d.path}\n    reference: {d.reference}\n    source:    {d.source}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
