"""
Copy the repository to DEST without the later-release files (lyratools/_release.py: LATER_RELEASE), for the
first public tag. Build artefacts, caches, test output and editor backups are not copied.

    python scripts/make_release.py ../lyratools-public [--force]
"""
import argparse
import ast
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "_data", "*~", ".DS_Store", "build", "dist",
                                ".git", ".pytest_cache", "*.part")


def later_release_paths():
    """LATER_RELEASE keys, read without importing lyratools."""
    tree = ast.parse(open(os.path.join(ROOT, "src", "lyratools", "_release.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "LATER_RELEASE":
            return list(ast.literal_eval(node.value))
    raise RuntimeError("LATER_RELEASE not found in _release.py")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest")
    ap.add_argument("--force", action="store_true", help="replace DEST if it exists")
    a = ap.parse_args(argv)
    dest = os.path.abspath(a.dest)
    if dest == ROOT or dest.startswith(ROOT + os.sep):
        ap.error("DEST must be outside the repository")
    if os.path.exists(dest):
        if not a.force:
            ap.error(f"{dest} exists (use --force to replace it)")
        shutil.rmtree(dest)
    shutil.copytree(ROOT, dest, ignore=IGNORE)
    for rel in later_release_paths():
        p = os.path.join(dest, rel)
        if os.path.exists(p):
            os.remove(p)
            print(f"[drop] {rel}")
        else:
            print(f"[warn] {rel} listed in LATER_RELEASE but not found")
    print(f"release tree written to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
