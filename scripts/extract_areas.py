"""Cut every camera image into one rectified, letterboxed crop per planogram area.

Output layout:
    <out>/<camera_id>/<timestamp>/<area_id>_<PRODUCT_SLUG>.jpg
    <out>/<camera_id>/<timestamp>/_overview.jpg        area outlines + labels
    <out>/<camera_id>/<timestamp>/_overview_boxes.jpg  area outlines only (coloured by product)
    <out>/manifest.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.draw import draw_annotated, draw_boxes  # noqa: E402
from shelfmon.planogram import load_planogram, slugify  # noqa: E402
from shelfmon.rectify import rectify  # noqa: E402

MANIFEST_FIELDS = ["camera_id", "image_file", "area_id", "product_id", "product_name",
                   "crop_path", "visible_frac", "native_w", "native_h"]


def extract_areas(images_dir: Path, planogram: dict, out: Path, size: int = 256, cameras: list[str] | None = None,
                  overviews: bool = True, log=print) -> list[dict]:
    """Write the crops (and overviews) of every <camera_id>/<timestamp>.jpg under images_dir, plus
    out/manifest.csv; returns the manifest rows."""
    for cam_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        if cam_dir.name not in planogram:
            print(f"warning: camera {cam_dir.name} not in planogram, skipping", file=sys.stderr)

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cam, areas in planogram.items():
        if cameras and cam not in cameras:
            continue
        images = sorted((images_dir / cam).glob("*.jpg"))
        if not images:
            print(f"warning: no images for camera {cam}", file=sys.stderr)
            continue
        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"warning: cannot read {img_path}, skipping", file=sys.stderr)
                continue
            ts_out = out / cam / img_path.stem
            ts_out.mkdir(parents=True, exist_ok=True)
            if overviews:
                cv2.imwrite(str(ts_out / "_overview.jpg"), draw_annotated(image, areas))
                cv2.imwrite(str(ts_out / "_overview_boxes.jpg"), draw_boxes(image, areas))
            for area in areas:
                crop, _, visible_frac, (nw, nh) = rectify(image, area.quad, size)
                crop_path = ts_out / f"{area.area_id}_{slugify(area.product_name)}.jpg"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({
                    "camera_id": cam, "image_file": img_path.name, "area_id": area.area_id,
                    "product_id": area.product_id, "product_name": area.product_name,
                    "crop_path": crop_path.relative_to(out).as_posix(),
                    "visible_frac": f"{visible_frac:.3f}", "native_w": nw, "native_h": nh,
                })
        log(f"{cam}: {len(images)} images x {len(areas)} areas")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote {len(rows)} crops and {out / 'manifest.csv'}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("Frutas e Legumes (train & test)"),
                    help="directory laid out as <camera_id>/<timestamp>.jpg")
    ap.add_argument("--planogram", type=Path, default=Path("planogram.json"))
    ap.add_argument("--out", type=Path, default=Path("data/areas"))
    ap.add_argument("--size", type=int, default=256, help="side of the square output crops")
    ap.add_argument("--cameras", nargs="*", help="only process these camera ids")
    args = ap.parse_args()
    extract_areas(args.images, load_planogram(args.planogram), args.out, args.size, args.cameras)

if __name__ == "__main__":
    main()
