"""Shared harness for accuracy experiments: the 7-round time CV of scripts/train_eval.py, in memory.

It reproduces the current model (hgb on handcrafted + radio, with the product-album override) and can
add extra per-crop feature columns or blend in another model's out-of-fold probabilities. It writes
nothing (no predictions file, no leaderboard). Besides the CV total it reports the score on each
camera's two newest photos, which have no later neighbours in time (CV is optimistic for those).
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval as te  # noqa: E402

from shelfmon.dataset import load_table  # noqa: E402
from shelfmon.prototypes import album_mask  # noqa: E402
from shelfmon.scoring import STOCK_LEVELS, decide_compliance, decide_stock, score, task_points  # noqa: E402
from shelfmon.zeroshot import load_kinds  # noqa: E402

FEATURES = ["handcrafted", "radio"]
PCA_DIMS = 64
BASELINE = 3585  # current model's CV total (of 3924)

warnings.filterwarnings("ignore")


def load(features_dir: Path = ROOT / "data/features"):
    """The training table with the embeddings the current model and its album need."""
    backbones = sorted(set(te.OVERRIDE_BACKBONES) | {f for f in FEATURES if f != "handcrafted"})
    return load_table(ROOT / "data/areas/manifest.csv", ROOT / "annotations/annotations.json", features_dir, backbones)


def newest_mask(rows: list[dict], n: int = 2) -> np.ndarray:
    """Rows from each camera's n newest photos."""
    images = {}
    for r in rows:
        images.setdefault(r["camera_id"], set()).add(r["image_file"])
    newest = {(cam, f) for cam, fs in images.items() for f in sorted(fs, key=lambda f: int(Path(f).stem))[-n:]}
    return np.array([(r["camera_id"], r["image_file"]) in newest for r in rows])


def cross_validate(table, extra: np.ndarray | None = None, blend: dict | None = None) -> dict:
    """Out-of-fold decisions of the current model.

    extra: (n, k) columns appended to the model input (NaN allowed). blend: {"p_stock": (n, 3),
    "p_compliant": (n,)} out-of-fold probabilities of another model, averaged 50/50 with ours.
    """
    kinds = load_kinds(ROOT / "analysis/products.csv")
    album_rows = album_mask(table.compliant, table.stock)
    n = len(table.rows)
    p_stock, p_comp, flagged = np.zeros((n, 3)), np.zeros(n), np.zeros(n, bool)
    for fold in np.unique(table.fold):
        train = table.fold != fold
        test = np.where(~train)[0]
        margins = {bb: te.album_margins(table, bb, train, kinds) for bb in te.OVERRIDE_BACKBONES}
        x_tr, x_te = te.build_matrices(table, FEATURES, train, PCA_DIMS, fold, margins)
        if extra is not None:
            x_tr, x_te = np.hstack([x_tr, extra[train]]), np.hstack([x_te, extra[~train]])
        p_stock[test] = te.fit_predict_proba("hgb", x_tr, table.stock[train], x_te, 3)
        p_comp[test] = te.fit_predict_proba("hgb", x_tr, table.compliant[train].astype(int), x_te, 2)[:, 1]
        votes = [margins[bb][:, 0] for bb in te.OVERRIDE_BACKBONES]
        flagged[test] = te.recogniser_flags(votes, train, album_rows, 0.005, len(votes))[test]
    if blend is not None:
        p_stock = (p_stock + blend["p_stock"]) / 2
        p_comp = (p_comp + blend["p_compliant"]) / 2
    return {"p_stock": p_stock, "p_compliant": p_comp, "flagged": flagged}


def report(name: str, table, out: dict) -> int:
    """Print CV total / compliance / stock and the newest-photos score; returns the CV total."""
    stock = decide_stock(out["p_stock"])
    flagged = out["flagged"] & (stock != STOCK_LEVELS.index("empty"))  # empty => compliant
    compliant = decide_compliance(out["p_compliant"]) & ~flagged
    labels = te.to_predictions(table.rows, table.stock, table.compliant)
    result = score(te.to_predictions(table.rows, stock, compliant), labels)
    comp_pts, stock_pts = task_points(result)
    newest = newest_mask(table.rows)
    rows = [r for r, m in zip(table.rows, newest) if m]
    res_new = score(te.to_predictions(rows, stock[newest], compliant[newest]),
                    te.to_predictions(rows, table.stock[newest], table.compliant[newest]))
    print(f"{name:<44} CV {result.total:+5} ({result.total - BASELINE:+4} vs current)  compliance {comp_pts:+}  "
          f"stock {stock_pts:+}  | newest photos {res_new.total:+} of {res_new.max}")
    return result.total
