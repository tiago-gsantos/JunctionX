"""Compute handcrafted features for every area crop in the manifest.

Groups A (container colour) and B (texture) describe each crop on its own. Group C compares each
crop with its area's references (typical and fullest photo of the day). Group D compares the crop's
SigLIP embedding with text descriptions of its expected product, the other products and empty
containers (needs <features-dir>/siglip2.npy and analysis/product_prompts.csv). By default the references
are built from this manifest and saved; pass --references to compare new photos (e.g. the
validation set) against references saved earlier instead.

Writes <features-dir>/handcrafted.csv: one row per manifest row (same order), with the key columns
(camera_id, image_file, area_id) followed by the feature columns. Embedding distances use the
<features-dir>/<backbone>.npy files that exist (see scripts/embed_crops.py).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.features import AB_FEATURES, area_mask, crop_features  # noqa: E402
from shelfmon.planogram import load_planogram  # noqa: E402
from shelfmon.references import (REFERENCE_BACKBONES, build_reference, load_references,  # noqa: E402
                                 relative_feature_names, relative_features, save_references)
from shelfmon.zeroshot import (ZEROSHOT_FEATURES, build_text_bank, load_descriptions, load_kinds,  # noqa: E402
                               zeroshot_features)

KEYS = ["camera_id", "image_file", "area_id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/areas/manifest.csv"))
    ap.add_argument("--planogram", type=Path, default=Path("planogram.json"))
    ap.add_argument("--frame-size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--references", type=Path, help="load these saved references instead of building new ones")
    ap.add_argument("--save-references", type=Path, default=Path("data/references/references.npz"))
    ap.add_argument("--product-prompts", type=Path, default=Path("analysis/product_prompts.csv"))
    ap.add_argument("--products", type=Path, default=Path("analysis/products.csv"))
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    planogram = load_planogram(args.planogram)
    areas = {a.area_id: a for cam_areas in planogram.values() for a in cam_areas}
    camera_products = {cam: {a.product_id for a in cam_areas} for cam, cam_areas in planogram.items()}
    embeddings = {bb: np.load(args.features_dir / f"{bb}.npy") for bb in REFERENCE_BACKBONES
                  if (args.features_dir / f"{bb}.npy").exists()}
    for bb, e in embeddings.items():
        assert len(e) == len(rows), f"{bb}.npy has {len(e)} rows, manifest has {len(rows)}; re-run embed_crops.py"
    refs = load_references(args.references) if args.references else {}
    if args.references and set(next(iter(refs.values())).typical_emb) != set(embeddings):
        sys.exit(f"error: references use backbones {sorted(next(iter(refs.values())).typical_emb)}, "
                 f"features dir has {sorted(embeddings)}")

    bank = None
    if "siglip2" in embeddings:
        from shelfmon.embeddings import Encoder
        bank = build_text_bank(Encoder("siglip2"), load_descriptions(args.product_prompts), load_kinds(args.products))

    by_area = defaultdict(list)
    for i, r in enumerate(rows):
        by_area[r["area_id"]].append(i)

    results = [None] * len(rows)
    for n, (area_id, idxs) in enumerate(by_area.items(), 1):
        crops = [cv2.imread(str(args.manifest.parent / rows[i]["crop_path"])) for i in idxs]
        mask = area_mask(areas[area_id], tuple(args.frame_size), crops[0].shape[0])
        ab = np.array([[crop_features(c, mask)[f] for f in AB_FEATURES] for c in crops])
        if not args.references:
            refs[area_id] = build_reference(crops, ab, {bb: e[idxs] for bb, e in embeddings.items()},
                                            [rows[i]["image_file"] for i in idxs])
        elif area_id not in refs:
            sys.exit(f"error: no saved reference for area {area_id}")
        for k, i in enumerate(idxs):
            rel = relative_features(ab[k], crops[k], {bb: e[i] for bb, e in embeddings.items()}, refs[area_id], mask)
            results[i] = dict(zip(AB_FEATURES, ab[k])) | rel
            if bank:
                results[i] |= zeroshot_features(embeddings["siglip2"][i], rows[i]["product_id"],
                                                camera_products[rows[i]["camera_id"]], bank)
        print(f"\r{n}/{len(by_area)} areas", end="", flush=True)
    print()

    if not args.references:
        save_references(refs, args.save_references)
        print(f"wrote {args.save_references} (+ .json)")
    out = args.features_dir / "handcrafted.csv"
    columns = AB_FEATURES + relative_feature_names(list(embeddings)) + (ZEROSHOT_FEATURES if bank else [])
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEYS + columns)
        writer.writeheader()
        for r, feats in zip(rows, results):
            writer.writerow({k: r[k] for k in KEYS} | {c: f"{feats[c]:.4f}" for c in columns})
    print(f"wrote {out} ({len(rows)} rows, {len(columns)} features)")


if __name__ == "__main__":
    main()
