"""One table for training: labels + handcrafted features + embeddings, one row per area crop (manifest order)."""
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shelfmon.scoring import STOCK_LEVELS, load_annotations

KEYS = ["camera_id", "image_file", "area_id"]


@dataclass
class Table:
    rows: list[dict]  # manifest rows
    stock: np.ndarray  # (n,) index into STOCK_LEVELS
    compliant: np.ndarray  # (n,) bool
    handcrafted: np.ndarray  # (n, k)
    handcrafted_names: list[str]
    embeddings: dict[str, np.ndarray]  # backbone -> (n, dim)
    fold: np.ndarray  # (n,) time-of-day round, see time_folds()
    # Group C recomputed per round with references built without that round's photos (or None):
    cv_relative: np.ndarray | None = None  # (rounds, n, len(cv_names))
    cv_names: list[str] | None = None

    def handcrafted_for_round(self, k: int) -> np.ndarray:
        """Handcrafted matrix whose group C columns come from round k's honest references."""
        if self.cv_relative is None:
            return self.handcrafted
        cols = [self.handcrafted_names.index(c) for c in self.cv_names]
        x = self.handcrafted.copy()
        x[:, cols] = self.cv_relative[k]
        return x


def time_folds(rows: list[dict], photos_per_fold: int = 2) -> np.ndarray:
    """Round k holds out photos 2k and 2k+1 (by time of day) of every camera, like the hidden set's 1-2 per camera."""
    images = defaultdict(set)
    for r in rows:
        images[r["camera_id"]].add(r["image_file"])
    rank = {(cam, img): i for cam, imgs in images.items() for i, img in enumerate(sorted(imgs))}
    return np.array([rank[(r["camera_id"], r["image_file"])] // photos_per_fold for r in rows])


def load_table(manifest: Path, labels: Path, features_dir: Path, backbones: list[str],
               honest_references: bool = True) -> Table:
    """honest_references: use data/features/group_c_cv.npz (see scripts/compute_cv_references.py)."""
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    annotations = load_annotations([labels])
    entries = [annotations[r["camera_id"]][r["image_file"]][r["area_id"]] for r in rows]

    with open(features_dir / "handcrafted.csv", encoding="utf-8") as f:
        feats = list(csv.DictReader(f))
    if [tuple(r[k] for k in KEYS) for r in feats] != [tuple(r[k] for k in KEYS) for r in rows]:
        raise ValueError("handcrafted.csv rows don't match the manifest; re-run compute_features.py")
    names = [c for c in feats[0] if c not in KEYS]

    embeddings = {bb: np.load(features_dir / f"{bb}.npy") for bb in backbones}
    for bb, e in embeddings.items():
        if len(e) != len(rows):
            raise ValueError(f"{bb}.npy has {len(e)} rows, manifest has {len(rows)}")

    cv_relative = cv_names = None
    if honest_references:
        path = features_dir / "group_c_cv.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/compute_cv_references.py")
        cv = np.load(path)
        cv_relative, cv_names = cv["features"], [str(n) for n in cv["names"]]
        if missing := set(cv_names) - set(names):
            raise ValueError(f"group_c_cv.npz has columns not in handcrafted.csv: {sorted(missing)}")

    return Table(
        rows=rows,
        stock=np.array([STOCK_LEVELS.index(e["stock_level"]) for e in entries]),
        compliant=np.array([bool(e["planogram_compliant"]) for e in entries]),
        handcrafted=np.array([[float(r[c]) for c in names] for r in feats]),
        handcrafted_names=names,
        embeddings=embeddings,
        fold=time_folds(rows),
        cv_relative=cv_relative,
        cv_names=cv_names,
    )
