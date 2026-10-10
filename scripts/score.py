"""Score a predictions file against labels with the official challenge points.

Both files use the official format: {camera_id: {image_file: {area_id: {planogram_compliant,
stock_level, stock_value}}}}. --labels accepts files and/or folders of *.json (merged).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.scoring import ID_POINTS, STOCK_POINTS, load_annotations, score  # noqa: E402

CASE_NAMES = {
    "id_correct": "Identification correct",
    "id_wrong": "Identification wrong",
    "stock_correct": "Stock level correct",
    "stock_below": "Stock level below the real stock",
    "stock_above": "Stock level above the real stock",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True, help="predictions JSON")
    ap.add_argument("--labels", nargs="+", required=True, help="label JSON files or folders")
    ap.add_argument("--json", action="store_true", help="print machine-readable output")
    args = ap.parse_args()

    predictions = load_annotations([args.pred])
    labels = load_annotations(args.labels)
    result = score(predictions, labels)

    if args.json:
        print(json.dumps({"total": result.total, "max": result.max, "counts": dict(result.counts),
                          "per_camera": {cam: {"points": p, "max": m} for cam, (p, m) in result.per_camera.items()}},
                         indent=2))
        return

    points_each = ID_POINTS | STOCK_POINTS
    print(f"{'Case':<34}{'Areas':>7}{'Each':>7}{'Points':>8}")
    for case, name in CASE_NAMES.items():
        n = result.counts[case]
        print(f"{name:<34}{n:>7}{points_each[case]:>+7}{n * points_each[case]:>+8}")
    print(f"{'Total':<48}{result.total:>+8}  (max {result.max}, {result.total / result.max:.1%})"
          if result.max else "No labelled areas.")
    if result.counts["missing"] or result.counts["extra"]:
        print(f"\nmissing areas (scored -5 each): {result.counts['missing']}   "
              f"extra areas (ignored): {result.counts['extra']}")
    print(f"\n{'Camera':<10}{'Points':>8}{'Max':>6}")
    for cam, (points, max_points) in result.per_camera.items():
        print(f"{cam:<10}{points:>+8}{max_points:>6}")


if __name__ == "__main__":
    main()
