"""Cut every camera image into one rectified, letterboxed crop per planogram area.

Output layout:
    <out>/<camera_id>/<area_id>_<PRODUCT_SLUG>_<timestamp>.jpg
    <out>/<camera_id>/_overview.jpg        area outlines + labels on a mid-day frame
    <out>/<camera_id>/_overview_boxes.jpg  area outlines only (coloured by product)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("Frutas e Legumes (train & test)"),
                    help="directory laid out as <camera_id>/<timestamp>.jpg")
    ap.add_argument("--planogram", type=Path, default=Path("planogram.json"))
    ap.add_argument("--out", type=Path, default=Path("data/areas"))
    ap.add_argument("--size", type=int, default=256, help="side of the square output crops")
    ap.add_argument("--cameras", nargs="*", help="only process these camera ids")
    args = ap.parse_args()

    planogram = load_planogram(args.planogram)
    for cam_dir in sorted(p for p in args.images.iterdir() if p.is_dir()):
        if cam_dir.name not in planogram:
            print(f"warning: camera {cam_dir.name} not in planogram, skipping", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cam, areas in planogram.items():
        if args.cameras and cam not in args.cameras:
            continue
        images = sorted((args.images / cam).glob("*.jpg"))
        if not images:
            print(f"warning: no images for camera {cam}", file=sys.stderr)
            continue
        cam_out = args.out / cam
        cam_out.mkdir(exist_ok=True)

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"warning: cannot read {img_path}, skipping", file=sys.stderr)
                continue
            for area in areas:
                crop, _, visible_frac, (nw, nh) = rectify(image, area.quad, args.size)
                crop_path = cam_out / f"{area.area_id}_{slugify(area.product_name)}_{img_path.stem}.jpg"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({
                    "camera_id": cam, "image_file": img_path.name, "area_id": area.area_id,
                    "product_id": area.product_id, "product_name": area.product_name,
                    "crop_path": crop_path.relative_to(args.out).as_posix(),
                    "visible_frac": f"{visible_frac:.3f}", "native_w": nw, "native_h": nh,
                })

        mid = cv2.imread(str(images[len(images) // 2]))
        cv2.imwrite(str(cam_out / "_overview.jpg"), draw_annotated(mid, areas))
        cv2.imwrite(str(cam_out / "_overview_boxes.jpg"), draw_boxes(mid, areas))
        print(f"{cam}: {len(images)} images x {len(areas)} areas")

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} crops and {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
