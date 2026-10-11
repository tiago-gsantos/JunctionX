"""The final, saved model: what scripts/train_eval.py --save-model writes and scripts/predict.py loads.

A bundle holds the fitted stock and compliance classifiers, the PCA of each embedding, the feature
order, and the confidence thresholds learned from the cross-validation probabilities.
"""
from pathlib import Path

import numpy as np

def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def save_bundle(bundle: dict, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict:
    import joblib
    return joblib.load(path)


def feature_matrix(bundle: dict, handcrafted: np.ndarray, handcrafted_names: list[str],
                   embeddings: dict[str, np.ndarray]) -> np.ndarray:
    """Model input for new crops, in the bundle's feature order (handcrafted columns matched by name)."""
    parts = []
    for name in bundle["features"]:
        if name == "handcrafted":
            missing = set(bundle["handcrafted_names"]) - set(handcrafted_names)
            if missing:
                raise ValueError(f"handcrafted features missing: {sorted(missing)}")
            parts.append(handcrafted[:, [handcrafted_names.index(c) for c in bundle["handcrafted_names"]]])
        else:
            x = unit(embeddings[name])
            parts.append(bundle["pca"][name].transform(x) if bundle["pca"].get(name) is not None else x)
    return np.hstack(parts)


def predict_proba(bundle: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(P(empty, half, full) of shape (n, 3), P(compliant) of shape (n,))."""
    p_stock = np.zeros((len(x), 3))
    p_stock[:, bundle["stock_model"].classes_] = bundle["stock_model"].predict_proba(x)
    comp = bundle["compliance_model"]
    p_comp = comp.predict_proba(x)[:, list(comp.classes_).index(1)] if 1 in comp.classes_ else np.zeros(len(x))
    return p_stock, p_comp


def wrong_probability(p_comp: np.ndarray, flagged: np.ndarray | None = None, override_precision: float = 0.0) -> np.ndarray:
    """P(non-compliant): the compliance model's, raised to the override's CV precision where the product
    album override fired (it decides on its own, so the model's probability alone would understate it)."""
    p = 1 - np.asarray(p_comp, dtype=float)
    return p if flagged is None else np.where(flagged, np.maximum(p, override_precision), p)


def task_confidence(p_stock: np.ndarray, p_wrong: np.ndarray, stock_idx: np.ndarray, compliant: np.ndarray):
    """For each area that becomes a staff task: P(the task is needed).

    Returns (kind, conf) arrays: kind is "wrong_product" (decided non-compliant), "restock" (decided empty
    or half) or "" (no task); conf is P(non-compliant) or P(not full) respectively.
    """
    kind = np.where(~compliant, "wrong_product", np.where(stock_idx < 2, "restock", ""))
    conf = np.where(~compliant, p_wrong, p_stock[:, 0] + p_stock[:, 1])
    return kind, conf


def confidence_thresholds(conf: np.ndarray, needed: np.ndarray, high_precision: float = 0.95,
                          low_precision: float = 0.6) -> dict[str, float]:
    """Thresholds on P(task needed), from cross-validated tasks whose truth is known.

    high: the lowest confidence at which the tasks at or above it are still >= high_precision needed.
    low: the highest confidence at which the tasks at or below it are needed < low_precision of the time.
    """
    order = np.argsort(-conf)
    c, v = conf[order], needed[order].astype(float)
    top_precision = np.cumsum(v) / np.arange(1, len(v) + 1)
    ok = np.where(top_precision >= high_precision)[0]
    high = float(c[ok.max()]) if ok.size else float("inf")
    c, v = c[::-1], v[::-1]
    bottom_precision = np.cumsum(v) / np.arange(1, len(v) + 1)
    bad = np.where(bottom_precision < low_precision)[0]
    low = float(c[bad.max()]) if bad.size else 0.0
    return {"high": high, "low": min(low, high - 1e-6)}


def confidence_level(conf: float, thresholds: dict[str, float]) -> str:
    if conf >= thresholds["high"]:
        return "high"
    return "low" if conf <= thresholds["low"] else "medium"
