"""
simple Snapshot current experiment results before running new improvements.

Creates a timestamped copy of all grid_summary.json files so i can always
compare to my latest results against what i had before.

Usage:
  python scripts/snapshot_results.py
  python scripts/snapshot_results.py --label v1_post_hoc
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

DATASETS = {
    "ml-1m":  Path("outputs/movielens/ml-1m"),
    "ml-10m": Path("outputs/movielens/ml-10m"),
    "ml-20m": Path("outputs/movielens/ml-20m"),
    "spotify": Path("outputs/spotify"),
}

SNAPSHOT_ROOT = Path("outputs/snapshots")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default=None,
                        help="Short label for the snapshot folder, e.g. v1_post_hoc. "
                             "Defaults to the current timestamp.")
    args = parser.parse_args()

    label = args.label or datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = SNAPSHOT_ROOT / label
    snap_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"label": label, "created_at": datetime.now().isoformat(), "files": []}

    for ds, folder in DATASETS.items():
        src = folder / "grid_summary.json"
        if not src.exists():
            print(f"  SKIP {ds}: {src} not found")
            continue

        dest = snap_dir / f"{ds}_grid_summary.json"
        shutil.copy(src, dest)
        print(f"  Copied {src} -> {dest}")
        manifest["files"].append({"dataset": ds, "source": str(src), "snapshot": str(dest)})

    manifest_path = snap_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSnapshot saved to: {snap_dir}")
    print(f"Datasets captured: {[e['dataset'] for e in manifest['files']]}")
    print(f"To compare: load outputs/snapshots/{label}/ alongside the current outputs/")


if __name__ == "__main__":
    main()