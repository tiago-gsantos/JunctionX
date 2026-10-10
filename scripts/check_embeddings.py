"""Label-free sanity checks of the cached crop embeddings (data/features/*.npy).

1. Nearest neighbour: for each crop, is the most similar other crop the same area (at another time)?
   Fixed cameras mean a good embedding should mostly say yes.
2. SigLIP zero-shot: does each crop match the text of its planogram product type ("a photo of banana")
   better than the other types? Compared against the planogram, not labels, so swapped or empty areas
   count as misses; treat it as a rough lower bound.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.embeddings import Encoder, cosine  # noqa: E402

PROMPTS = {"en": "a photo of {}", "pt": "uma foto de {}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--manifest", type=Path, default=Path("data/areas/manifest.csv"))
    ap.add_argument("--products", type=Path, default=Path("analysis/products.csv"))
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(args.products, encoding="utf-8") as f:
        products = {p["product_id"]: p for p in csv.DictReader(f)}
    area_ids = np.array([r["area_id"] for r in rows])
    gtypes = np.array([products[r["product_id"]]["generic_type"] for r in rows])

    print("1. Nearest neighbour (most similar other crop)")
    print(f"   {'model':<10}{'dim':>6}{'same area':>12}{'same type':>12}")
    for path in sorted(args.features.glob("*.npy")):
        feats = np.load(path)
        assert len(feats) == len(rows), f"{path} has {len(feats)} rows, manifest has {len(rows)}"
        sim = cosine(feats, feats)
        np.fill_diagonal(sim, -np.inf)
        nn = sim.argmax(axis=1)
        print(f"   {path.stem:<10}{feats.shape[1]:>6}{(area_ids[nn] == area_ids).mean():>12.1%}"
              f"{(gtypes[nn] == gtypes).mean():>12.1%}")

    siglip_path = args.features / "siglip2.npy"
    if not siglip_path.exists():
        return
    print("\n2. SigLIP zero-shot product type vs planogram")
    image_feats = np.load(siglip_path)
    encoder = Encoder("siglip2")
    cams = np.array([r["camera_id"] for r in rows])
    camera_types = {cam: set(gtypes[cams == cam]) for cam in set(cams)}
    print(f"   {'prompt':<8}{'all 44 types':>14}{'same camera':>14}")
    for lang, template in PROMPTS.items():
        column = f"generic_{lang}"
        names = {p["generic_type"]: p[column] for p in products.values()}
        texts = sorted(set(names.values()))  # papaya/melon map two types to one English name
        sim = cosine(image_feats, encoder.embed_texts([template.format(t) for t in texts]))
        expected = np.array([names[g] for g in gtypes])
        pred_all = np.array(texts)[sim.argmax(axis=1)]
        pred_cam = []
        for i, cam in enumerate(cams):
            allowed = [j for j, t in enumerate(texts) if t in {names[g] for g in camera_types[cam]}]
            pred_cam.append(texts[allowed[int(sim[i, allowed].argmax())]])
        pred_cam = np.array(pred_cam)
        print(f"   {lang:<8}{(pred_all == expected).mean():>14.1%}{(pred_cam == expected).mean():>14.1%}")
        if lang == "en":
            confusions = Counter(zip(expected[pred_cam != expected], pred_cam[pred_cam != expected]))
            print("   most common same-camera confusions (expected -> predicted, crops):")
            for (exp, pred), n in confusions.most_common(8):
                print(f"     {exp} -> {pred}: {n}")


if __name__ == "__main__":
    main()
