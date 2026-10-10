"""Constant-answer baseline: the same answer for every area of every image.

Writes an official-format predictions file covering every planogram area of every image in --images.
It sets the floor every real model must beat.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.planogram import load_planogram  # noqa: E402
from shelfmon.verify import LEVEL_VALUES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", choices=LEVEL_VALUES, default="full")
    ap.add_argument("--compliant", choices=["true", "false"], default="true")
    ap.add_argument("--images", type=Path, default=Path("Frutas e Legumes (train & test)"))
    ap.add_argument("--planogram", type=Path, default=Path("planogram.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    answer = {"planogram_compliant": args.compliant == "true", "stock_level": args.stock,
              "stock_value": LEVEL_VALUES[args.stock]}
    predictions = {
        cam: {img.name: {a.area_id: dict(answer) for a in areas} for img in sorted((args.images / cam).glob("*.jpg"))}
        for cam, areas in load_planogram(args.planogram).items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(predictions, indent=1), encoding="utf-8")
    n = sum(len(a) for imgs in predictions.values() for a in imgs.values())
    print(f"wrote {args.out}: {n} areas, all {answer}")


if __name__ == "__main__":
    main()
