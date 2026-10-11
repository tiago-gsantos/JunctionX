"""Product "photo album" for compliance: what each product looks like in photos labelled correct.

A product's prototype is the mean embedding of its crops labelled compliant and not empty, so swaps
never enter the album. Each crop is compared with its expected product's prototype and with rivals:
- margin_camera: expected minus the best other product planned on the same camera (swaps happen
  between neighbours, e.g. oranges moved into a banana area);
- margin_kind: expected minus the best product of another generic type anywhere in the store (ignores
  look-alike varieties).
Both are NaN when the expected product has no prototype. Unlike the SigLIP text margins (group D),
this compares images with images, so packaging and colour of the real products count.
"""
from pathlib import Path

import numpy as np

PROTO_FEATURES = ["margin_camera", "margin_kind"]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def album_mask(compliant: np.ndarray, stock: np.ndarray) -> np.ndarray:
    """Rows allowed in the album: labelled compliant and not empty (stock index 0 = empty)."""
    return compliant & (stock > 0)


def camera_products(rows: list[dict]) -> dict[str, set[str]]:
    """Camera id -> product ids planned on it (every area of a camera appears in each of its photos)."""
    out = {}
    for r in rows:
        out.setdefault(r["camera_id"], set()).add(r["product_id"])
    return out


def build_prototypes(emb: np.ndarray, product_ids: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Product id -> unit-length mean embedding of the rows in `mask`."""
    return {pid: _unit(_unit(emb[mask & (product_ids == pid)]).mean(axis=0)) for pid in np.unique(product_ids[mask])}


def prototype_margins(emb: np.ndarray, rows: list[dict], protos: dict[str, np.ndarray], kinds: dict[str, str],
                      cam_products: dict[str, set[str]]) -> np.ndarray:
    """(n, len(PROTO_FEATURES)) margins of each row's crop; kinds: product id -> generic type."""
    ids = sorted(protos)
    col = {pid: j for j, pid in enumerate(ids)}
    sims = _unit(emb) @ np.stack([protos[p] for p in ids]).T
    out = np.full((len(rows), len(PROTO_FEATURES)), np.nan)
    rivals = {}  # (expected product, camera) -> (camera rival columns, other-kind columns)
    for i, r in enumerate(rows):
        pid, cam = r["product_id"], r["camera_id"]
        if pid not in col:
            continue
        if (pid, cam) not in rivals:
            rivals[(pid, cam)] = ([col[p] for p in cam_products[cam] if p != pid and p in col],
                                  [j for p, j in col.items() if kinds[p] != kinds[pid]])
        same_camera, other_kind = rivals[(pid, cam)]
        expected = sims[i, col[pid]]
        if same_camera:
            out[i, 0] = expected - sims[i, same_camera].max()
        if other_kind:
            out[i, 1] = expected - sims[i, other_kind].max()
    return out


def save_prototypes(albums: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    """albums: backbone -> product id -> prototype. One .npz with keys '<backbone>__<product id>'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{bb}__{pid}": v for bb, protos in albums.items() for pid, v in protos.items()})


def load_prototypes(path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(path)
    albums = {}
    for key in data.files:
        bb, pid = key.split("__", 1)
        albums.setdefault(bb, {})[pid] = data[key]
    return albums
