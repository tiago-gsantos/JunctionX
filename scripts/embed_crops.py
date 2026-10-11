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


def embed_manifest(rows: list[dict], crops_dir: Path, encoder: Encoder, batch_size: int | None = None,
                   progress=None) -> np.ndarray:
    """(n_rows, dim) embeddings of the manifest's crops, in row order. progress(done, total) is optional."""
    batch_size = batch_size or encoder.spec.batch_size
    chunks = []
    for i in range(0, len(rows), batch_size):
        batch = [Image.open(crops_dir / r["crop_path"]).convert("RGB") for r in rows[i:i + batch_size]]
        chunks.append(encoder.embed_images(batch))
        if progress:
            progress(min(i + batch_size, len(rows)), len(rows))
    return np.concatenate(chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=BACKBONES)
    ap.add_argument("--manifest", type=Path, default=Path("data/areas/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/features"))
    ap.add_argument("--batch-size", type=int, help="override the model's default batch size")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    encoder = Encoder(args.model)

    start = time.time()
    features = embed_manifest(rows, args.manifest.parent, encoder, args.batch_size, lambda done, total: print(
        f"\r{args.model}: {done}/{total} crops", end="", flush=True))
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
