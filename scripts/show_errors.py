"""Contact sheet of the areas where a predictions file disagrees with the labels.

--task compliance: missed non-compliant areas and false "wrong product" alarms, with SigLIP's best
guess of the product actually shown. --task stock: wrong stock levels.
Writes data/analysis/errors/<task>_<run>.jpg and prints one line per error.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.draw import FONT_LEGEND, FONT_NAME  # noqa: E402
from shelfmon.scoring import load_annotations  # noqa: E402

TZ = ZoneInfo("Europe/Lisbon")
TILE, CAPTION, COLUMNS = 256, 74, 6


def siglip_guesses(rows: list[dict], indices: list[int], features_dir: Path, prompts: Path) -> dict[int, str]:
    """Description of the product whose text best matches each crop (SigLIP)."""
    from shelfmon.embeddings import Encoder
    from shelfmon.zeroshot import build_text_bank, load_descriptions, load_kinds
    descriptions = load_descriptions(prompts)
    bank = build_text_bank(Encoder("siglip2"), descriptions, load_kinds(Path("analysis/products.csv")))
    emb = np.load(features_dir / "siglip2.npy")[indices]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    best = (emb @ bank.products.T).argmax(axis=1)
    return {i: descriptions[bank.product_ids[b]] for i, b in zip(indices, best)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--task", choices=["compliance", "stock"], default="compliance")
    ap.add_argument("--labels", type=Path, default=Path("annotations/annotations.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/areas/manifest.csv"))
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--prompts", type=Path, default=Path("analysis/product_prompts.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/analysis/errors"))
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(args.prompts, encoding="utf-8") as f:
        expected_desc = {r["product_id"]: r["description_en"] for r in csv.DictReader(f)}
    labels, preds = load_annotations([args.labels]), load_annotations([args.pred])
    key = "planogram_compliant" if args.task == "compliance" else "stock_level"

    errors = []
    for i, r in enumerate(rows):
        truth = labels[r["camera_id"]][r["image_file"]][r["area_id"]]
        pred = preds[r["camera_id"]][r["image_file"]][r["area_id"]]
        if truth[key] != pred[key]:
            errors.append((i, truth, pred))
    # Missed problems first (the costly direction), then false alarms; grouped by area.
    errors.sort(key=lambda e: (e[1][key] is not False, rows[e[0]]["area_id"], rows[e[0]]["image_file"]))
    guesses = siglip_guesses(rows, [e[0] for e in errors], args.features_dir, args.prompts) \
        if args.task == "compliance" and errors else {}

    n_rows = max(1, -(-len(errors) // COLUMNS))
    sheet = Image.new("RGB", (COLUMNS * TILE, n_rows * (TILE + CAPTION)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    print(f"{len(errors)} {args.task} errors in {args.pred.name}")
    for n, (i, truth, pred) in enumerate(errors):
        r = rows[i]
        when = datetime.fromtimestamp(int(Path(r["image_file"]).stem), TZ).strftime("%H:%M")
        if args.task == "compliance":
            kind = "MISSED wrong product" if truth[key] is False else "FALSE ALARM"
            lines = [f"{r['area_id']}  {when}  {kind}", f"expected: {expected_desc[r['product_id']]}",
                     f"SigLIP sees: {guesses[i]}", f"stock label: {truth['stock_level']}"]
        else:
            lines = [f"{r['area_id']}  {when}", f"true {truth[key]} -> predicted {pred[key]}",
                     f"expected: {expected_desc[r['product_id']]}"]
        x, y = (n % COLUMNS) * TILE, (n // COLUMNS) * (TILE + CAPTION)
        crop = cv2.cvtColor(cv2.imread(str(args.manifest.parent / r["crop_path"])), cv2.COLOR_BGR2RGB)
        sheet.paste(Image.fromarray(crop), (x, y))
        colour = (200, 0, 0) if truth[key] is False or args.task == "stock" else (200, 120, 0)
        for k, line in enumerate(lines):
            draw.text((x + 4, y + TILE + 2 + 17 * k), line[:44], font=FONT_LEGEND if k == 0 else FONT_NAME,
                      fill=colour if k == 0 else (30, 30, 30))
        print(f"  {r['area_id']:<8} {when}  " + " | ".join(lines[1:]) + f"  [{lines[0].split('  ')[-1]}]")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.task}_{args.pred.stem}.jpg"
    sheet.save(out, quality=90)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
