"""Group C features per cross-validation round, with references built without the held-out photos.

For round k, each area's typical/fullest references come only from the photos *not* held out in
round k, so held-out photos are compared against references they're not part of, exactly like
the hidden validation photos. Run once after compute_features.py; train_eval.py then uses
data/features/group_c_cv.npz.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.dataset import KEYS, time_folds  # noqa: E402
from shelfmon.features import AB_FEATURES, area_mask  # noqa: E402
from shelfmon.planogram import load_planogram  # noqa: E402
from shelfmon.references import (REFERENCE_BACKBONES, build_reference, relative_feature_names,  # noqa: E402
                                 relative_features)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/areas/manifest.csv"))
    ap.add_argument("--planogram", type=Path, default=Path("planogram.json"))
    ap.add_argument("--frame-size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(args.features_dir / "handcrafted.csv", encoding="utf-8") as f:
        feats = list(csv.DictReader(f))
    if [tuple(r[k] for k in KEYS) for r in feats] != [tuple(r[k] for k in KEYS) for r in rows]:
        sys.exit("error: handcrafted.csv rows don't match the manifest; re-run compute_features.py")
    ab_all = np.array([[float(r[c]) for c in AB_FEATURES] for r in feats])
    embeddings = {bb: np.load(args.features_dir / f"{bb}.npy") for bb in REFERENCE_BACKBONES
                  if (args.features_dir / f"{bb}.npy").exists()}
    areas = {a.area_id: a for cam_areas in load_planogram(args.planogram).values() for a in cam_areas}
    folds = time_folds(rows)
    rounds = np.unique(folds)
    names = relative_feature_names(list(embeddings))
    out = np.zeros((len(rounds), len(rows), len(names)), np.float32)

    by_area = defaultdict(list)
    for i, r in enumerate(rows):
        by_area[r["area_id"]].append(i)
    for n, (area_id, idxs) in enumerate(by_area.items(), 1):
        idxs = np.array(idxs)
        crops = [cv2.imread(str(args.manifest.parent / rows[i]["crop_path"])) for i in idxs]
        mask = area_mask(areas[area_id], tuple(args.frame_size), crops[0].shape[0])
        files = [rows[i]["image_file"] for i in idxs]
        for k in rounds:
            keep = np.where(folds[idxs] != k)[0]
            ref = build_reference([crops[j] for j in keep], ab_all[idxs[keep]],
                                  {bb: e[idxs[keep]] for bb, e in embeddings.items()}, [files[j] for j in keep])
            for j, i in enumerate(idxs):
                rel = relative_features(ab_all[i], crops[j], {bb: e[i] for bb, e in embeddings.items()}, ref, mask)
                out[k, i] = [rel[c] for c in names]
        print(f"\r{n}/{len(by_area)} areas", end="", flush=True)
    print()

    path = args.features_dir / "group_c_cv.npz"
    np.savez_compressed(path, features=out, names=np.array(names), rounds=rounds)
    print(f"wrote {path}: {len(rounds)} rounds x {len(rows)} rows x {len(names)} features")


if __name__ == "__main__":
    main()
