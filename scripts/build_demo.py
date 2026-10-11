"""Build the live demo: photos the demo model has never seen, and that model.

Holds out the newest --photos photos of every camera (default 2: 28 photos, like the hidden validation
set) and builds everything the demo model learns from with the other photos only, so uploading the
held-out photos in the demo gives honest predictions and confidences:
    data/demo/photos/<camera_id>/<ts>.jpg     the held-out photos: upload this folder in the demo
    data/demo/features/                       embeddings, handcrafted.csv, group_c_cv.npz of the training photos
    data/demo/references/references.npz       "usual shelf" references built from the training photos
    data/models/demo.joblib                   the demo model (same recipe as data/models/final.joblib)
It also clears data/demo/runs, so the demo app starts with no data. Needs the full pipeline outputs
(data/areas/manifest.csv and data/features/*.npy). The real model (final.joblib) is not touched.

    .venv/bin/python scripts/build_demo.py      # ~4 min
    .venv/bin/python app/server.py --demo
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compute_features import compute_features  # noqa: E402
from extract_areas import MANIFEST_FIELDS  # noqa: E402

from shelfmon.planogram import load_planogram  # noqa: E402
from shelfmon.references import REFERENCE_BACKBONES  # noqa: E402

MODEL_ARGS = ["--model", "hgb", "--features", "handcrafted", "radio", "--override", "proto"]  # as final.joblib
BACKBONES = sorted(set(REFERENCE_BACKBONES) | {"radio", "dinov3", "siglip2"})  # features + album votes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", type=int, default=2, help="newest photos per camera held out for the demo")
    ap.add_argument("--images", type=Path, default=ROOT / "Frutas e Legumes (train & test)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/demo")
    args = ap.parse_args()

    manifest = ROOT / "data/areas/manifest.csv"
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    images = {}
    for r in rows:
        images.setdefault(r["camera_id"], set()).add(r["image_file"])
    held_out = {(cam, img) for cam, imgs in images.items()
                for img in sorted(imgs, key=lambda f: int(Path(f).stem))[-args.photos:]}
    train = [i for i, r in enumerate(rows) if (r["camera_id"], r["image_file"]) not in held_out]
    print(f"demo photos: {len(held_out)} ({args.photos} per camera); training photos: "
          f"{sum(len(v) for v in images.values()) - len(held_out)}")

    shutil.rmtree(args.out, ignore_errors=True)
    for cam, img in sorted(held_out):
        (args.out / "photos" / cam).mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.images / cam / img, args.out / "photos" / cam / img)

    # The training photos' crops already exist; this manifest sits next to them so crop paths resolve.
    train_manifest = manifest.with_name("manifest_demo_train.csv")
    with open(train_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows[i] for i in train)
    features = args.out / "features"
    features.mkdir(parents=True)
    for bb in BACKBONES:
        np.save(features / f"{bb}.npy", np.load(ROOT / "data/features" / f"{bb}.npy")[train])

    print("references + features from the training photos ...")
    compute_features(train_manifest, load_planogram(ROOT / "planogram.json"), features,
                     save_references_to=args.out / "references/references.npz")
    print("per-round references for cross-validation ...")
    subprocess.run([sys.executable, str(ROOT / "scripts/compute_cv_references.py"), "--manifest", str(train_manifest),
                    "--features-dir", str(features)], check=True, stdout=subprocess.DEVNULL)
    print("training the demo model ...")
    subprocess.run([sys.executable, str(ROOT / "scripts/train_eval.py"), *MODEL_ARGS, "--manifest", str(train_manifest),
                    "--features-dir", str(features), "--name", "demo", "--no-log",
                    "--save-model", str(ROOT / "data/models/demo.joblib")], check=True)
    print(f"\ndemo ready: upload {args.out / 'photos'} in  .venv/bin/python app/server.py --demo")


if __name__ == "__main__":
    main()
