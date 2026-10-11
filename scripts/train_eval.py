"""Train and evaluate stock + compliance models with time-of-day cross-validation.

Each of the 7 rounds holds out 2 photos per camera, trains on the rest and predicts the held-out
photos, so every area crop gets a prediction from a model that never saw it. The predictions are
scored with the official points (vs the always-compliant/always-full baseline: +1598) and logged to
results/LEADERBOARD.md.

Models: logreg (logistic regression), hgb (gradient-boosted trees), knn (nearest labelled photos of the
same area). Features: "handcrafted" (groups A-D) and/or embedding names (siglip2, dinov2, dinov3,
radio), each compressed with PCA fitted on the training rounds only, and/or proto_<backbone>: the
product-album margins of shelfmon/prototypes.py, with the album built from the training rounds only.

--override proto marks an area non-compliant when the album says another product is there (margin
below a threshold set on the training rounds, for at least --override-votes backbones), unless the
area is predicted empty. It catches swaps the compliance model never saw in that area; with
--recogniser-only it is the whole compliance answer (no memory of the day's swaps). --cv camera holds
out one whole camera per round instead (runs tagged @camera).
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.dataset import camera_folds, load_table  # noqa: E402
from shelfmon.prototypes import album_mask, build_prototypes, camera_products, prototype_margins  # noqa: E402
from shelfmon.scoring import (STOCK_LEVELS, STOCK_VALUES, decide_compliance, decide_stock,  # noqa: E402
                              score, task_points)
from shelfmon.zeroshot import load_kinds  # noqa: E402

BASELINE = 1598  # always compliant + always full, on all 1962 labelled areas
EMBEDDINGS = ["siglip2", "dinov2", "dinov3", "radio"]
PROTOS = [f"proto_{bb}" for bb in EMBEDDINGS]
OVERRIDE_BACKBONES = ["dinov2", "radio", "siglip2"]  # album margins that vote in --override proto


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def album_margins(table, backbone: str, train: np.ndarray, kinds: dict[str, str]) -> np.ndarray:
    """(n, 2) product-album margins of every row, with the album built from the training rows only."""
    product_ids = np.array([r["product_id"] for r in table.rows])
    emb = table.embeddings[backbone]
    protos = build_prototypes(emb, product_ids, train & album_mask(table.compliant, table.stock))
    return prototype_margins(emb, table.rows, protos, kinds, camera_products(table.rows))


def build_matrices(table, features: list[str], train: np.ndarray, pca_dims: int, fold: int | None,
                   margins: dict[str, np.ndarray]):
    """Feature matrices for the training and test rows; PCA is fitted on training rows only.

    fold: the time round whose honest group C columns to use (None: the shared references).
    margins: backbone -> this round's album_margins().
    """
    parts_train, parts_test = [], []
    for name in features:
        if name == "handcrafted":
            x = table.handcrafted if fold is None else table.handcrafted_for_round(fold)
        elif name.startswith("proto_"):
            x = margins[name.removeprefix("proto_")]
        else:
            x = unit(table.embeddings[name])
            if pca_dims:
                pca = PCA(n_components=min(pca_dims, train.sum(), x.shape[1]), random_state=0).fit(x[train])
                x = pca.transform(x)
        parts_train.append(x[train])
        parts_test.append(x[~train])
    return np.hstack(parts_train), np.hstack(parts_test)


def make_model(kind: str):
    if kind == "logreg":  # the imputer fills album margins of products without a prototype (NaN)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              l2_regularization=1.0, random_state=0)
    raise ValueError(kind)


def fit_predict_proba(kind: str, x_train, y_train, x_test, n_classes: int) -> np.ndarray:
    """Class probabilities (n_test, n_classes), with zeros for classes absent from this training round."""
    model = make_model(kind).fit(x_train, y_train)
    probs = np.zeros((len(x_test), n_classes))
    probs[:, model.classes_] = model.predict_proba(x_test)
    return probs


def knn_proba(table, features, train, test_idx, labels, n_classes, k: int, alpha: float = 0.5) -> np.ndarray:
    """Vote of the k most similar labelled photos of the same area (cosine similarity), lightly smoothed."""
    embs = [unit(table.embeddings[f]) for f in features if f in EMBEDDINGS]
    if not embs:
        raise ValueError("knn needs at least one embedding feature")
    x = unit(np.hstack(embs))
    areas = np.array([r["area_id"] for r in table.rows])
    probs = np.zeros((len(test_idx), n_classes))
    for j, i in enumerate(test_idx):
        candidates = np.where(train & (areas == areas[i]))[0]
        sims = x[candidates] @ x[i]
        top = candidates[np.argsort(-sims)[:k]]
        weights = np.clip(np.sort(-sims)[:k] * -1, 0, None) + 1e-3
        counts = np.bincount(labels[top], weights=weights, minlength=n_classes) + alpha
        probs[j] = counts / counts.sum()
    return probs


def recogniser_flags(margins: list[np.ndarray], train: np.ndarray, reference: np.ndarray, quantile: float,
                     min_votes: int) -> np.ndarray:
    """Rows where at least min_votes margins fall below their threshold ("another product is here").

    Each threshold is the given quantile of that margin over the training rows in `reference` (labelled
    compliant and not empty), so about that share of correct photos would be flagged. NaN never votes.
    """
    votes = np.zeros(len(train), int)
    for m in margins:
        votes += m < np.nanquantile(m[train & reference], quantile)
    return votes >= min_votes


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["logreg", "hgb", "knn"], required=True)
    ap.add_argument("--features", nargs="+", required=True, choices=["handcrafted"] + EMBEDDINGS + PROTOS)
    ap.add_argument("--cv", choices=["time", "camera"], default="time",
                    help="time: 2 photos per camera held out per round; camera: one whole camera per round")
    ap.add_argument("--override", choices=["proto", "text"],
                    help="mark non-compliant when the product album (proto) or SigLIP text margin (text) says "
                         "another product is in the area")
    ap.add_argument("--override-quantile", type=float, default=0.005,
                    help="threshold = this quantile of the margins of correct training photos")
    ap.add_argument("--override-votes", type=int, default=3,
                    help=f"album backbones ({', '.join(OVERRIDE_BACKBONES)}) that must agree (proto only)")
    ap.add_argument("--recogniser-only", action="store_true",
                    help="compliance from the override alone, ignoring the compliance model")
    ap.add_argument("--pca", type=int, default=64, help="PCA dimensions per embedding (0 = no PCA)")
    ap.add_argument("--k", type=int, default=3, help="neighbours for knn")
    ap.add_argument("--name", help="run name (default: model + features)")
    ap.add_argument("--manifest", type=Path, default=Path("data/areas/manifest.csv"))
    ap.add_argument("--labels", type=Path, default=Path("annotations/annotations.json"))
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--products", type=Path, default=Path("analysis/products.csv"), help="generic type per product")
    ap.add_argument("--leaderboard", type=Path, default=Path("results/LEADERBOARD.md"))
    ap.add_argument("--no-log", action="store_true", help="don't append to the leaderboard")
    ap.add_argument("--shared-references", action="store_true",
                    help="use group C features built from all photos (slightly optimistic; the old protocol)")
    args = ap.parse_args()
    if args.recogniser_only and not args.override:
        ap.error("--recogniser-only needs --override")
    name = args.name or (f"{args.model}_{'+'.join(args.features)}"
                         + (f"+override-{args.override}" + ("-only" if args.recogniser_only else "") if args.override else "")
                         + ("@shared" if args.shared_references else "") + ("@camera" if args.cv == "camera" else ""))
    warnings.filterwarnings("ignore", category=UserWarning)

    album_backbones = {f.removeprefix("proto_") for f in args.features if f in PROTOS}
    if args.override == "proto":
        album_backbones |= set(OVERRIDE_BACKBONES)
    backbones = sorted({f for f in args.features if f in EMBEDDINGS} | album_backbones)
    table = load_table(args.manifest, args.labels, args.features_dir, backbones,
                       honest_references=not args.shared_references)
    kinds = load_kinds(args.products)
    folds = camera_folds(table.rows) if args.cv == "camera" else table.fold
    album_rows = album_mask(table.compliant, table.stock)
    n = len(table.rows)
    p_stock, p_comp = np.zeros((n, 3)), np.zeros(n)
    flagged = np.zeros(n, bool)  # --override: the recogniser says another product is in the area
    for fold in np.unique(folds):
        train = folds != fold
        test_idx = np.where(~train)[0]
        margins = {bb: album_margins(table, bb, train, kinds) for bb in album_backbones}
        if args.model == "knn":
            p_stock[test_idx] = knn_proba(table, args.features, train, test_idx, table.stock, 3, args.k)
            p_comp[test_idx] = knn_proba(table, args.features, train, test_idx, table.compliant.astype(int), 2, args.k)[:, 1]
        else:
            x_tr, x_te = build_matrices(table, args.features, train, args.pca,
                                        None if args.cv == "camera" else fold, margins)
            p_stock[test_idx] = fit_predict_proba(args.model, x_tr, table.stock[train], x_te, 3)
            p_comp[test_idx] = fit_predict_proba(args.model, x_tr, table.compliant[train].astype(int), x_te, 2)[:, 1]
        if args.override == "proto":
            votes = [margins[bb][:, 0] for bb in OVERRIDE_BACKBONES]  # margin_camera
            flagged[test_idx] = recogniser_flags(votes, train, album_rows, args.override_quantile,
                                                 args.override_votes)[test_idx]
        elif args.override == "text":
            text_margin = table.handcrafted[:, table.handcrafted_names.index("margin_camera_kind")]
            flagged[test_idx] = recogniser_flags([text_margin], train, album_rows, args.override_quantile, 1)[test_idx]

    labels = to_predictions(table.rows, table.stock, table.compliant)
    flagged &= decide_stock(p_stock) != STOCK_LEVELS.index("empty")  # empty => compliant (team convention)
    model_compliant = decide_compliance(p_comp)
    compliant = ~flagged if args.recogniser_only else model_compliant & ~flagged
    results = {}
    for rule, stock_idx in {"expected points": decide_stock(p_stock), "most likely": p_stock.argmax(axis=1)}.items():
        results[rule] = (stock_idx, score(to_predictions(table.rows, stock_idx, compliant), labels))

    print(f"{name}  ({len(np.unique(folds))} rounds, {args.cv} CV, {n} areas)")
    print(f"{'stock decision rule':<20}{'total':>8}{'vs baseline':>13}{'compliance':>12}{'stock':>8}")
    for rule, (_, res) in results.items():
        comp_pts, stock_pts = task_points(res)
        print(f"{rule:<20}{res.total:>+8}{res.total - BASELINE:>+13}{comp_pts:>+12}{stock_pts:>+8}   ({res.total / res.max:.1%} of max)")

    stock_idx, best = results["expected points"]
    print("\nstock confusion (rows = true, columns = predicted), expected-points rule:")
    print(f"{'':<8}" + "".join(f"{lvl:>8}" for lvl in STOCK_LEVELS))
    for t, lvl in enumerate(STOCK_LEVELS):
        print(f"{lvl:<8}" + "".join(f"{np.sum((table.stock == t) & (stock_idx == p)):>8}" for p in range(3)))
    tp = np.sum(~table.compliant & ~compliant)
    print(f"\ncompliance: caught {tp}/{np.sum(~table.compliant)} non-compliant areas, "
          f"{np.sum(table.compliant & ~compliant)} false alarms")
    if args.override:
        swap = ~table.compliant
        print(f"override ({args.override}): flagged {flagged.sum()} areas: {np.sum(flagged & swap)} swaps, "
              f"{np.sum(flagged & ~swap)} false alarms")
        if not args.recogniser_only:
            new = flagged & model_compliant
            print(f"  swaps the compliance model missed: {area_counts(table.rows, new & swap)}")
            print(f"  new false alarms: {area_counts(table.rows, new & ~swap)}")
    print(f"  still missed: {area_counts(table.rows, ~table.compliant & compliant)}")

    pred_path = Path("data/predictions") / f"{name}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(to_predictions(table.rows, stock_idx, compliant), indent=1), encoding="utf-8")
    # Out-of-round probabilities in manifest order, for error analysis (scripts/error_gallery.py).
    np.savez(pred_path.with_suffix(".proba.npz"), p_stock=p_stock, p_compliant=p_comp, override_flagged=flagged,
             keys=np.array([[r[k] for k in ("camera_id", "image_file", "area_id")] for r in table.rows]))
    print(f"\nwrote {pred_path} (+ .proba.npz)")

    if not args.no_log:
        comp_pts, stock_pts = task_points(best)
        if not args.leaderboard.exists():
            args.leaderboard.parent.mkdir(parents=True, exist_ok=True)
            args.leaderboard.write_text(
                "# Leaderboard\n\nTime-of-day cross-validation (7 rounds, 2 photos per camera held out each), "
                "official points on all 1962 labelled areas. Baseline (always compliant + always full): +1598 "
                "of 3924. Group C references are rebuilt per round without the held-out photos; runs marked "
                "@shared used references built from all photos (slightly optimistic); runs marked @camera hold out "
                "one whole camera per round instead (14 rounds, shared references).\n\n| date | run | total | vs baseline | compliance | stock | most-likely rule |\n"
                "|---|---|---|---|---|---|---|\n", encoding="utf-8")
        with open(args.leaderboard, "a", encoding="utf-8") as f:
            f.write(f"| {datetime.now():%Y-%m-%d %H:%M} | `{name}` | {best.total:+} | {best.total - BASELINE:+} | "
                    f"{comp_pts:+} | {stock_pts:+} | {results['most likely'][1].total:+} |\n")


if __name__ == "__main__":
    main()
