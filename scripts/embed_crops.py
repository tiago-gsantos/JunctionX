"""Embed every area crop in the manifest with a frozen pretrained model.

Writes data/features/<model>.npy (one row per manifest row, same order) and <model>.json (metadata).
Models: see shelfmon.embeddings.BACKBONES (siglip2, dinov2, dinov3, radio).
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.embeddings import BACKBONES, Encoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=BACKBONES)
    ap.add_argument("--manifest", type=Path, default=Path("data/areas/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/features"))
    ap.add_argument("--batch-size", type=int, help="override the model's default batch size")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    crops_dir = args.manifest.parent
    encoder = Encoder(args.model)
    batch_size = args.batch_size or encoder.spec.batch_size

    start = time.time()
    chunks = []
    for i in range(0, len(rows), batch_size):
        batch = [Image.open(crops_dir / r["crop_path"]).convert("RGB") for r in rows[i:i + batch_size]]
        chunks.append(encoder.embed_images(batch))
        print(f"\r{args.model}: {min(i + batch_size, len(rows))}/{len(rows)} crops", end="", flush=True)
    features = np.concatenate(chunks)
    elapsed = time.time() - start

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / f"{args.model}.npy", features)
    meta = {
        "model": args.model, "hf_id": encoder.spec.hf_id, "rows": len(rows), "dim": features.shape[1],
        "manifest": str(args.manifest), "device": encoder.device, "seconds": round(elapsed, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (args.out / f"{args.model}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out / (args.model + '.npy')} {features.shape} in {elapsed:.1f}s on {encoder.device}")


if __name__ == "__main__":
    main()
