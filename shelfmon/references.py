"""Group C: compare each crop with its own area's references ("how different from usual?").

Every area gets two label-free references, built from that area's photos of the day:
- typical: the median of each A/B feature, the pixel-wise median image, the mean embedding;
- fullest: the area's most textured photo (highest edge_frac_coarse), i.e. its most stocked moment.
New photos of the same area (e.g. the validation set) are compared against the saved references.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from shelfmon.features import AB_FEATURES

FULLEST_BY = "edge_frac_coarse"
PIXEL_BLUR = (5, 5)  # softens JPEG noise and sub-pixel jitter before comparing pixels


@dataclass
class Reference:
    typical_feats: np.ndarray  # median of each AB feature, in AB_FEATURES order
    fullest_feats: np.ndarray
    typical_img: np.ndarray  # uint8 crop, pixel-wise median
    fullest_img: np.ndarray
    typical_emb: dict[str, np.ndarray]  # backbone -> unit-length mean embedding
    fullest_emb: dict[str, np.ndarray]
    fullest_image_file: str


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def build_reference(crops: list[np.ndarray], ab: np.ndarray, embeddings: dict[str, np.ndarray],
                    image_files: list[str]) -> Reference:
    """ab: (n, len(AB_FEATURES)) features of the area's crops; embeddings: backbone -> (n, dim)."""
    best = int(np.argmax(ab[:, AB_FEATURES.index(FULLEST_BY)]))
    return Reference(
        typical_feats=np.median(ab, axis=0),
        fullest_feats=ab[best],
        typical_img=np.median(np.stack(crops), axis=0).astype(np.uint8),
        fullest_img=crops[best],
        typical_emb={bb: _unit(_unit(e).mean(axis=0)) for bb, e in embeddings.items()},
        fullest_emb={bb: _unit(e[best]) for bb, e in embeddings.items()},
        fullest_image_file=image_files[best],
    )


def pixel_difference(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute colour difference over real shelf pixels, 0 (identical) to 1."""
    diff = np.abs(cv2.GaussianBlur(a, PIXEL_BLUR, 0).astype(np.float32)
                  - cv2.GaussianBlur(b, PIXEL_BLUR, 0).astype(np.float32))
    return float(diff[mask].mean() / 255)


def relative_feature_names(backbones: list[str]) -> list[str]:
    names = [f"d_typ_{f}" for f in AB_FEATURES] + [f"d_full_{f}" for f in AB_FEATURES]
    names += ["pix_diff_typ", "pix_diff_full"]
    for bb in backbones:
        names += [f"emb_dist_typ_{bb}", f"emb_dist_full_{bb}"]
    return names


def relative_features(ab: np.ndarray, crop: np.ndarray, embeddings: dict[str, np.ndarray],
                      ref: Reference, mask: np.ndarray) -> dict[str, float]:
    """Group C for one crop. embeddings: backbone -> (dim,) vector of this crop."""
    out = {f"d_typ_{f}": float(v) for f, v in zip(AB_FEATURES, ab - ref.typical_feats)}
    out |= {f"d_full_{f}": float(v) for f, v in zip(AB_FEATURES, ab - ref.fullest_feats)}
    out["pix_diff_typ"] = pixel_difference(crop, ref.typical_img, mask)
    out["pix_diff_full"] = pixel_difference(crop, ref.fullest_img, mask)
    for bb, e in embeddings.items():
        e = _unit(e)
        out[f"emb_dist_typ_{bb}"] = float(1 - e @ ref.typical_emb[bb])
        out[f"emb_dist_full_{bb}"] = float(1 - e @ ref.fullest_emb[bb])
    return out


def save_references(refs: dict[str, Reference], path: Path) -> None:
    """One compressed .npz with every area's arrays, plus a .json with the fullest frame per area."""
    arrays = {}
    for area_id, r in refs.items():
        arrays[f"{area_id}__typical_feats"] = r.typical_feats
        arrays[f"{area_id}__fullest_feats"] = r.fullest_feats
        arrays[f"{area_id}__typical_img"] = r.typical_img
        arrays[f"{area_id}__fullest_img"] = r.fullest_img
        for bb in r.typical_emb:
            arrays[f"{area_id}__typical_emb__{bb}"] = r.typical_emb[bb]
            arrays[f"{area_id}__fullest_emb__{bb}"] = r.fullest_emb[bb]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    meta = {"ab_features": AB_FEATURES, "fullest_by": FULLEST_BY,
            "fullest_image_file": {a: r.fullest_image_file for a, r in refs.items()}}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_references(path: Path) -> dict[str, Reference]:
    data = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["ab_features"] != AB_FEATURES:
        raise ValueError(f"{path} was built with different A/B features; rebuild the references")
    refs = {}
    for area_id, image_file in meta["fullest_image_file"].items():
        emb_keys = [k for k in data.files if k.startswith(f"{area_id}__typical_emb__")]
        backbones = [k.split("__")[-1] for k in emb_keys]
        refs[area_id] = Reference(
            typical_feats=data[f"{area_id}__typical_feats"], fullest_feats=data[f"{area_id}__fullest_feats"],
            typical_img=data[f"{area_id}__typical_img"], fullest_img=data[f"{area_id}__fullest_img"],
            typical_emb={bb: data[f"{area_id}__typical_emb__{bb}"] for bb in backbones},
            fullest_emb={bb: data[f"{area_id}__fullest_emb__{bb}"] for bb in backbones},
            fullest_image_file=image_file,
        )
    return refs
