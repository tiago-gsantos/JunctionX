"""Verify labels or predictions in the official format before training or submitting.

Checks that every planogram area of every image in --images is present (a missing area scores -5),
that values are valid (planogram_compliant true/false, stock_level empty/half/full, stock_value
0/0.5/1 matching the level), and that nothing unknown is included. Exits with status 1 on errors.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.planogram import load_planogram  # noqa: E402
from shelfmon.scoring import load_annotations  # noqa: E402
from shelfmon.verify import verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="official-format JSON files or folders (merged)")
    ap.add_argument("--images", type=Path, default=Path("Frutas e Legumes (train & test)"),
                    help="image folder (<camera_id>/<timestamp>.jpg) whose images must all be covered")
    ap.add_argument("--planogram", type=Path, default=Path("planogram.json"))
    ap.add_argument("--max-lines", type=int, default=30, help="errors/warnings to print (0 = all)")
    args = ap.parse_args()

    planogram = load_planogram(args.planogram)
    images = {cam: sorted(p.name for p in (args.images / cam).glob("*.jpg")) for cam in planogram}
    try:
        annotations = load_annotations(args.files)
    except ValueError as e:
        sys.exit(f"error: {e}")

    report = verify(annotations, planogram, images)

    print(f"{'Camera':<8}{'Images complete':>17}{'Areas present':>16}")
    for cam, (done_imgs, n_imgs, done_areas, n_areas) in report.coverage.items():
        flag = "" if done_areas == n_areas else "  <-- incomplete"
        print(f"{cam:<8}{f'{done_imgs}/{n_imgs}':>17}{f'{done_areas}/{n_areas}':>16}{flag}")
    total_done = sum(c[2] for c in report.coverage.values())
    total = sum(c[3] for c in report.coverage.values())
    print(f"{'Total':<8}{'':>17}{f'{total_done}/{total}':>16}")

    for title, lines in (("ERRORS", report.errors), ("WARNINGS", report.warnings)):
        if lines:
            shown = lines if args.max_lines == 0 else lines[:args.max_lines]
            print(f"\n{title} ({len(lines)}):")
            print("\n".join(f"  {line}" for line in shown))
            if len(shown) < len(lines):
                print(f"  ... {len(lines) - len(shown)} more (use --max-lines 0 to see all)")

    print("\nOK: every area present and valid." if not report.errors else "\nFAILED: fix the errors above.")
    sys.exit(1 if report.errors else 0)


if __name__ == "__main__":
    main()
