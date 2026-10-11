"""Run the saved model on a folder of new camera photos (e.g. the hidden validation set).

    .venv/bin/python scripts/predict.py --images <dir> --out data/runs/<name>

<dir> is laid out as <camera_id>/<unix_timestamp>.jpg. Writes to <out>:
    areas/                crops + manifest.csv (as scripts/extract_areas.py)
    features/             embeddings + handcrafted.csv, compared with the saved references
    predictions.json      official submission format, checked with shelfmon.verify
    details.json          {camera: {image: {area_id: {p_stock, p_compliant, p_wrong, override}}}} for the app
                          (p_wrong: P(non-compliant) after the product-album override, if the model has one)
Needs models/final.joblib and models/references.npz (committed; rebuilt by train_eval.py and compute_features.py).
--dinov2 uses models/dinov2/ instead: the same model built with the open DINOv2, for users without access
to the gated DINOv3.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compute_features import compute_features  # noqa: E402
from embed_crops import embed_manifest  # noqa: E402
from extract_areas import extract_areas  # noqa: E402

from shelfmon.dataset import KEYS  # noqa: E402
from shelfmon.io import write_json  # noqa: E402
from shelfmon.model import feature_matrix, load_bundle, predict_proba, wrong_probability  # noqa: E402
from shelfmon.planogram import load_planogram  # noqa: E402
from shelfmon.prototypes import prototype_margins  # noqa: E402
from shelfmon.references import DINO_CHOICES  # noqa: E402
from shelfmon.scoring import STOCK_LEVELS, STOCK_VALUES, decide_compliance, decide_stock  # noqa: E402
from shelfmon.verify import verify  # noqa: E402
from shelfmon.zeroshot import build_text_bank, load_descriptions, load_kinds  # noqa: E402

_encoders = {}  # name -> Encoder, kept between runs when imported by the app


DINOV2 = {"model": ROOT / "models/dinov2/final.joblib", "references": ROOT / "models/dinov2/references.npz"}


def encoder(name: str):
    from shelfmon.embeddings import Encoder
    if name not in _encoders:
        try:
            _encoders[name] = Encoder(name)
        except OSError as e:  # gated or missing model on Hugging Face
            hint = (" DINOv3 is gated: request access and run `hf auth login`, or use the DINOv2 model "
                    "(predict.py --dinov2, app/server.py --dinov2).") if name == "dinov3" else ""
            raise ValueError(f"cannot load the {name} image model.{hint}") from e
    return _encoders[name]


def run_pipeline(images: Path, out: Path, model: Path = ROOT / "models/final.joblib",
                 references: Path = ROOT / "models/references.npz",
                 planogram_path: Path = ROOT / "planogram.json", progress=None) -> dict:
    """Write predictions.json + details.json for every photo under images; returns the predictions.

    progress(step, fraction) is called with a short human-readable step name and 0..1 overall progress.
    """
    def report(step: str, fraction: float):
        if progress:
            progress(step, fraction)

    bundle = load_bundle(model)
    planogram = load_planogram(planogram_path)
    ref_backbones = sorted({k.split("__")[-1] for k in np.load(references).files if "_emb__" in k})
    dino = next((bb for bb in DINO_CHOICES if bb in ref_backbones), "dinov3")  # the references decide
    override = bundle.get("override")
    needed = [f for f in bundle["features"] if f != "handcrafted"] + (override["backbones"] if override else [])
    backbones = ref_backbones + [bb for bb in dict.fromkeys(needed) if bb not in ref_backbones]

    report("Finding the shelf areas", 0.0)
    rows = extract_areas(images, planogram, out / "areas", overviews=False, log=lambda *_: None)
    if not rows:
        raise ValueError(f"no photos of known cameras under {images}")

    features_dir = out / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    embeddings = {}
    for k, bb in enumerate(backbones):
        def step(done, total, k=k):
            report("Looking at the products", 0.1 + 0.6 * (k + done / total) / len(backbones))
        embeddings[bb] = embed_manifest(rows, out / "areas", encoder(bb), progress=step)
        np.save(features_dir / f"{bb}.npy", embeddings[bb])

    report("Comparing with the usual shelf", 0.7)
    bank = build_text_bank(encoder("siglip2"), load_descriptions(ROOT / "analysis/product_prompts.csv"),
                           load_kinds(ROOT / "analysis/products.csv"))
    handcrafted_csv = compute_features(out / "areas/manifest.csv", planogram, features_dir, references=references,
                                       bank=bank, dino=dino, progress=lambda d, t: report("Comparing with the usual shelf",
                                                                              0.7 + 0.25 * d / t))
    with open(handcrafted_csv, encoding="utf-8") as f:
        feats = list(csv.DictReader(f))
    names = [c for c in feats[0] if c not in KEYS]
    handcrafted = np.array([[float(r[c]) for c in names] for r in feats])

    report("Checking stock and products", 0.95)
    p_stock, p_comp = predict_proba(bundle, feature_matrix(bundle, handcrafted, names, embeddings))
    stock_idx, compliant = decide_stock(p_stock), decide_compliance(p_comp)
    flagged = np.zeros(len(rows), bool)
    if override:  # product album: another product's photos match better, for enough backbones
        kinds = load_kinds(ROOT / "analysis/products.csv")
        cam_products = {cam: {a.product_id for a in areas} for cam, areas in planogram.items()}
        votes = sum(prototype_margins(embeddings[bb], rows, override["albums"][bb], kinds, cam_products)[:, 0]
                    < override["thresholds"][bb] for bb in override["backbones"])
        flagged = (votes >= override["votes"]) & (stock_idx != STOCK_LEVELS.index("empty"))  # empty => compliant
        compliant &= ~flagged
    p_wrong = wrong_probability(p_comp, flagged, override["precision"] if override else 0.0)

    predictions, details = {}, {}
    for i, r in enumerate(rows):
        cam, img, area = r["camera_id"], r["image_file"], r["area_id"]
        predictions.setdefault(cam, {}).setdefault(img, {})[area] = {
            "planogram_compliant": bool(compliant[i]), "stock_level": STOCK_LEVELS[stock_idx[i]],
            "stock_value": STOCK_VALUES[stock_idx[i]]}
        details.setdefault(cam, {}).setdefault(img, {})[area] = {
            "p_stock": [round(float(p), 4) for p in p_stock[i]], "p_compliant": round(float(p_comp[i]), 4),
            "p_wrong": round(float(p_wrong[i]), 4), "override": bool(flagged[i])}

    image_files = {}
    for r in rows:
        image_files.setdefault(r["camera_id"], set()).add(r["image_file"])
    report_ = verify(predictions, planogram, {cam: sorted(v) for cam, v in image_files.items()})
    if report_.errors:
        raise ValueError("predictions failed verification:\n" + "\n".join(report_.errors[:20]))

    write_json(out / "predictions.json", predictions, indent=1)
    write_json(out / "details.json", details, indent=None)
    report("Done", 1.0)
    return predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="directory laid out as <camera_id>/<timestamp>.jpg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "models/final.joblib")
    ap.add_argument("--references", type=Path, default=ROOT / "models/references.npz")
    ap.add_argument("--dinov2", action="store_true", help="use the model built with DINOv2 (no DINOv3 access needed)")
    args = ap.parse_args()
    if args.dinov2:
        args.model, args.references = DINOV2["model"], DINOV2["references"]

    start = time.time()
    try:
        predictions = run_pipeline(args.images, args.out, args.model, args.references,
                                   progress=lambda step, f: print(f"\r{f:4.0%}  {step:<40}", end="", flush=True))
    except ValueError as e:
        sys.exit(f"\nerror: {e}")
    n = sum(len(areas) for imgs in predictions.values() for areas in imgs.values())
    print(f"\nwrote {args.out / 'predictions.json'} ({n} areas) and details.json in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
