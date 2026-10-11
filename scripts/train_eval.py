"""Cross-validate the model, then train it on every labelled photo and save it for scripts/predict.py.

The model: gradient-boosted trees (stock and compliance) on the handcrafted features plus the RADIO
embedding (PCA to 64 dimensions), and a product-album check that marks an area as the wrong product when
all three album backbones say another product's photos match it better (never for areas predicted empty).

Cross-validation: each of the 7 rounds holds out 2 photos per camera (by time of day), trains on the
rest and predicts the held-out photos, so every crop gets a prediction from a model that never saw it.
They are scored with the official points and written to data/predictions/<name>.json (+ .proba.npz).
The cross-validated probabilities also set the confidence thresholds the app shows.

    .venv/bin/python scripts/train_eval.py      # ~20 s -> models/final.joblib
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.dataset import load_table  # noqa: E402
from shelfmon.model import (confidence_level, confidence_thresholds, save_bundle, task_confidence,  # noqa: E402
                            unit, wrong_probability)
from shelfmon.prototypes import album_mask, build_prototypes, camera_products, prototype_margins  # noqa: E402
from shelfmon.references import DINO_CHOICES  # noqa: E402
from shelfmon.scoring import (STOCK_LEVELS, STOCK_VALUES, decide_compliance, decide_stock,  # noqa: E402
                              score, task_points)
from shelfmon.zeroshot import load_kinds  # noqa: E402

BASELINE = 1598  # always compliant + always full, on all 1962 labelled areas
FEATURES = ["handcrafted", "radio"]  # handcrafted columns + embeddings (PCA) fed to the trees
PCA_DIMS = 64
ALBUM_BACKBONES = ["dinov3", "radio", "siglip2"]  # all three must agree to flag a wrong product (--dino swaps dinov3)
ALBUM_QUANTILE = 0.005  # threshold = this quantile of the margins of correct training photos


def album_margins(table, backbone: str, train: np.ndarray, kinds: dict[str, str]) -> np.ndarray:
    """(n, 2) product-album margins of every row, with the album built from the training rows only."""
    product_ids = np.array([r["product_id"] for r in table.rows])
    emb = table.embeddings[backbone]
    protos = build_prototypes(emb, product_ids, train & album_mask(table.compliant, table.stock))
    return prototype_margins(emb, table.rows, protos, kinds, camera_products(table.rows))


def build_matrices(table, train: np.ndarray, fold: int):
    """Feature matrices for the training and test rows; PCA is fitted on training rows only."""
    parts_train, parts_test = [], []
    for name in FEATURES:
        if name == "handcrafted":
            x = table.handcrafted_for_round(fold)
        else:
            x = unit(table.embeddings[name])
            x = PCA(n_components=min(PCA_DIMS, train.sum(), x.shape[1]), random_state=0).fit(x[train]).transform(x)
        parts_train.append(x[train])
        parts_test.append(x[~train])
    return np.hstack(parts_train), np.hstack(parts_test)


def make_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0)


def fit_predict_proba(x_train, y_train, x_test, n_classes: int) -> np.ndarray:
    """Class probabilities (n_test, n_classes), with zeros for classes absent from this training round."""
    model = make_model().fit(x_train, y_train)
    probs = np.zeros((len(x_test), n_classes))
    probs[:, model.classes_] = model.predict_proba(x_test)
    return probs


def album_flags(margins: list[np.ndarray], train: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rows where every margin falls below its threshold ("another product is here").

    Each threshold is ALBUM_QUANTILE of that margin over the training rows in `reference` (labelled
    compliant and not empty), so about that share of correct photos would be flagged. NaN never votes.
    """
    votes = sum(m < np.nanquantile(m[train & reference], ALBUM_QUANTILE) for m in margins)
    return votes >= len(margins)


def area_counts(rows, mask) -> str:
    """'6215_5×2 6383_7' for the rows in mask."""
    counts = {}
    for r, m in zip(rows, mask):
        if m:
            counts[r["area_id"]] = counts.get(r["area_id"], 0) + 1
    return " ".join(a + (f"×{c}" if c > 1 else "") for a, c in sorted(counts.items())) or "-"


def to_predictions(rows, stock_idx, compliant) -> dict:
    out = {}
    for r, s, c in zip(rows, stock_idx, compliant):
        out.setdefault(r["camera_id"], {}).setdefault(r["image_file"], {})[r["area_id"]] = {
            "planogram_compliant": bool(c), "stock_level": STOCK_LEVELS[s], "stock_value": STOCK_VALUES[s]}
    return out


def fit_final(table) -> dict:
    """Stock + compliance models fitted on every labelled crop (PCA too), as in one CV round."""
    parts, pcas = [], {}
    for name in FEATURES:
        if name == "handcrafted":
            parts.append(table.handcrafted)
            continue
        x = unit(table.embeddings[name])
        pcas[name] = PCA(n_components=min(PCA_DIMS, *x.shape), random_state=0).fit(x)
        parts.append(pcas[name].transform(x))
    x = np.hstack(parts)
    return {"model": "hgb", "features": FEATURES, "pca": pcas, "handcrafted_names": table.handcrafted_names,
            "stock_model": make_model().fit(x, table.stock),
            "compliance_model": make_model().fit(x, table.compliant.astype(int))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/areas/manifest.csv"))
    ap.add_argument("--labels", type=Path, default=Path("annotations/annotations.json"))
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--products", type=Path, default=Path("analysis/products.csv"), help="generic type per product")
    ap.add_argument("--out", type=Path, default=Path("models/final.joblib"), help="where to save the trained model")
    ap.add_argument("--name", default="final", help="name of the cross-validated predictions in data/predictions/")
    ap.add_argument("--dino", choices=DINO_CHOICES, default="dinov3",
                    help="dinov2: open fallback (build its features with compute_features.py --dino dinov2)")
    args = ap.parse_args()
    album_backbones = [args.dino if bb == "dinov3" else bb for bb in ALBUM_BACKBONES]
    warnings.filterwarnings("ignore", category=UserWarning)

    backbones = sorted({f for f in FEATURES if f != "handcrafted"} | set(album_backbones))
    table = load_table(args.manifest, args.labels, args.features_dir, backbones)
    kinds = load_kinds(args.products)
    album_rows = album_mask(table.compliant, table.stock)
    n = len(table.rows)
    p_stock, p_comp = np.zeros((n, 3)), np.zeros(n)
    flagged = np.zeros(n, bool)  # the album says another product is in the area
    for fold in np.unique(table.fold):
        train = table.fold != fold
        test_idx = np.where(~train)[0]
        x_tr, x_te = build_matrices(table, train, fold)
        p_stock[test_idx] = fit_predict_proba(x_tr, table.stock[train], x_te, 3)
        p_comp[test_idx] = fit_predict_proba(x_tr, table.compliant[train].astype(int), x_te, 2)[:, 1]
        margins = [album_margins(table, bb, train, kinds)[:, 0] for bb in album_backbones]  # margin_camera
        flagged[test_idx] = album_flags(margins, train, album_rows)[test_idx]

    stock_idx = decide_stock(p_stock)
    flagged &= stock_idx != STOCK_LEVELS.index("empty")  # empty => compliant (team convention)
    model_compliant = decide_compliance(p_comp)
    compliant = model_compliant & ~flagged
    result = score(to_predictions(table.rows, stock_idx, compliant), to_predictions(table.rows, table.stock, table.compliant))
    comp_pts, stock_pts = task_points(result)
    print(f"cross-validation ({len(np.unique(table.fold))} rounds, {n} areas): {result.total:+} of {result.max} "
          f"({result.total / result.max:.1%}), compliance {comp_pts:+}, stock {stock_pts:+}, "
          f"baseline {BASELINE:+}")
    print("\nstock confusion (rows = true, columns = predicted):")
    print(f"{'':<8}" + "".join(f"{lvl:>8}" for lvl in STOCK_LEVELS))
    for t, lvl in enumerate(STOCK_LEVELS):
        print(f"{lvl:<8}" + "".join(f"{np.sum((table.stock == t) & (stock_idx == p)):>8}" for p in range(3)))
    swap = ~table.compliant
    new = flagged & model_compliant
    print(f"\ncompliance: caught {np.sum(swap & ~compliant)}/{swap.sum()} wrong products, "
          f"{np.sum(~swap & ~compliant)} false alarms")
    print(f"  album flagged {flagged.sum()} areas ({np.sum(flagged & swap)} real swaps); "
          f"it overruled the model on {new.sum()}: {area_counts(table.rows, new)}")
    print(f"  still missed: {area_counts(table.rows, swap & compliant)}")

    pred_path = Path("data/predictions") / f"{args.name}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(to_predictions(table.rows, stock_idx, compliant), indent=1), encoding="utf-8")
    # Out-of-round probabilities in manifest order, for scripts/error_report.py.
    np.savez(pred_path.with_suffix(".proba.npz"), p_stock=p_stock, p_compliant=p_comp, override_flagged=flagged,
             keys=np.array([[r[k] for k in ("camera_id", "image_file", "area_id")] for r in table.rows]))
    print(f"\nwrote {pred_path} (+ .proba.npz)")

    # Final model: album from every labelled photo, thresholds as in each CV round.
    product_ids = np.array([r["product_id"] for r in table.rows])
    album = {"backbones": album_backbones, "votes": len(album_backbones), "albums": {}, "thresholds": {},
             "precision": float(swap[new].mean()) if new.any() else 0.5}
    for bb in album_backbones:
        protos = build_prototypes(table.embeddings[bb], product_ids, album_rows)
        margin = prototype_margins(table.embeddings[bb], table.rows, protos, kinds, camera_products(table.rows))[:, 0]
        album["albums"][bb] = protos
        album["thresholds"][bb] = float(np.nanquantile(margin[album_rows], ALBUM_QUANTILE))
    p_wrong = wrong_probability(p_comp, flagged, album["precision"])
    task_kinds, conf = task_confidence(p_stock, p_wrong, stock_idx, compliant)
    needed = {"restock": table.stock < 2, "wrong_product": swap}
    thresholds = {k: confidence_thresholds(conf[task_kinds == k], needed[k][task_kinds == k]) for k in needed}
    print(f"\nalbum precision when it overrules the model: {album['precision']:.0%}")
    for k, t in thresholds.items():
        sel = task_kinds == k
        levels = np.array([confidence_level(c, t) for c in conf[sel]])
        stats = ", ".join(f"{lvl} {np.sum(levels == lvl)} ({needed[k][sel][levels == lvl].mean():.0%} needed)"
                          for lvl in ["high", "medium", "low"] if np.any(levels == lvl))
        print(f"confidence, {k:<14} high >= {t['high']:.2f}, low <= {t['low']:.2f}: {stats}")
    bundle = fit_final(table) | {"name": args.name, "confidence": thresholds, "override": album,
                                 "cv_total": result.total, "cv_max": result.max,
                                 "created": datetime.now().isoformat(timespec="seconds")}
    save_bundle(bundle, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
