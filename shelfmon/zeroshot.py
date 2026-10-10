"""Group D: does the crop look like the planogram's product? (SigLIP image-text similarity)

Each of the 94 products has an English description (analysis/product_prompts.csv, editable), placed
in a few supermarket-scene sentences whose text embeddings are averaged. A crop is compared with its
expected product, the other products on the same camera, all other products, and empty containers.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEMPLATES = [
    "a photo of {} in a supermarket crate",
    "{} on a supermarket shelf",
    "a pile of {} for sale in a grocery store",
    "{} in a fruit and vegetable display, seen from above",
]
# Empty states seen in the store; a crop is scored against the best-matching one.
EMPTY_PROMPTS = [
    "an empty plastic crate in a supermarket",
    "an empty metal shelf tray in a supermarket",
    "an empty black display tray in a supermarket",
    "an empty cardboard box on a supermarket shelf",
    "a supermarket shelf covered with a grey cloth",
]
ZEROSHOT_FEATURES = ["sim_expected", "margin_camera", "margin_camera_kind", "margin_all", "margin_all_kind",
                     "empty_vs_expected"]


@dataclass
class TextBank:
    product_ids: list[str]
    products: np.ndarray  # (n_products, dim) unit vectors, one per product (templates averaged)
    empty: np.ndarray  # (n_empty, dim) unit vectors
    kinds: dict[str, str]  # product id -> generic type (BANANA, MACA, ...)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def load_descriptions(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {r["product_id"]: r["description_en"] for r in csv.DictReader(f)}


def load_kinds(products_csv: Path) -> dict[str, str]:
    """Product id -> generic type, from analysis/products.csv."""
    with open(products_csv, encoding="utf-8") as f:
        return {r["product_id"]: r["generic_type"] for r in csv.DictReader(f)}


def build_text_bank(encoder, descriptions: dict[str, str], kinds: dict[str, str]) -> TextBank:
    """encoder: shelfmon.embeddings.Encoder for siglip2 (the model whose image embeddings we compare)."""
    ids = sorted(descriptions)
    sentences = [t.format(descriptions[pid]) for pid in ids for t in TEMPLATES]
    per_sentence = _unit(encoder.embed_texts(sentences)).reshape(len(ids), len(TEMPLATES), -1)
    return TextBank(ids, _unit(per_sentence.mean(axis=1)), _unit(encoder.embed_texts(EMPTY_PROMPTS)), kinds)


def zeroshot_features(image_emb: np.ndarray, expected_id: str, camera_ids: set[str], bank: TextBank) -> dict[str, float]:
    """image_emb: the crop's SigLIP embedding. camera_ids: product ids planned on the crop's camera.

    Margins are expected minus the best rival: another product (catches variety swaps) or another kind
    of product (ignores look-alike varieties). When a camera only has one kind, the kind margin falls
    back to the product margin.
    """
    sims = dict(zip(bank.product_ids, bank.products @ _unit(image_emb)))
    expected = sims[expected_id]
    kind = bank.kinds[expected_id]

    def margin(candidates: list[str]) -> float:
        return float(expected - max(sims[p] for p in candidates)) if candidates else 0.0

    camera = [p for p in camera_ids if p != expected_id]
    store = [p for p in bank.product_ids if p != expected_id]
    camera_kind = [p for p in camera if bank.kinds[p] != kind]
    return {
        "sim_expected": float(expected),
        "margin_camera": margin(camera),
        "margin_camera_kind": margin(camera_kind or camera),
        "margin_all": margin(store),
        "margin_all_kind": margin([p for p in store if bank.kinds[p] != kind]),
        "empty_vs_expected": float((bank.empty @ _unit(image_emb)).max() - expected),
    }
