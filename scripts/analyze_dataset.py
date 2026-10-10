"""General dataset summary: cameras, frames, areas, products and partly visible areas.

Writes analysis/DATASET.md and analysis/products.csv. Needs data/areas/manifest.csv
from scripts/extract_areas.py.
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.planogram import CORNERS, load_planogram, slugify  # noqa: E402

TZ = ZoneInfo("Europe/Lisbon")

# The generic type is the first word of the accent-stripped name (MAÇA, MAÇÃ, MACA -> MACA),
# skipping leading modifiers and expanding catalogue abbreviations. Products that look different
# from the rest of their first-word group keep two words.
MODIFIERS = {"MINI"}
ABBREVIATIONS = {"CITR": "CITRINOS"}
TWO_WORD_TYPES = {"BATATA DOCE", "COUVE BROCOLO", "COUVE FLOR"}
# A pack quantity such as "500 G", "1.5 KG" or "EMB. 3KG", but not a caliber range ("73/85 KG") or grade ("R0 KG").
PACK_QTY = re.compile(r"(?<![\w/.])\d+(?:[.,]\d+)?\s*(?:KG|GR|G)\b")
# Generic type -> (English, Portuguese with accents), used for zero-shot text prompts.
GENERIC_NAMES = {
    "ABACATE": ("avocado", "abacate"), "ABACAXI": ("pineapple", "abacaxi"), "ABOBORA": ("pumpkin", "abóbora"),
    "ALFACE": ("lettuce", "alface"), "ALHO": ("garlic", "alho"), "ALPERCE": ("apricot", "alperce"),
    "BANANA": ("banana", "banana"), "BATATA": ("potato", "batata"), "BATATA DOCE": ("sweet potato", "batata-doce"),
    "BERINGELA": ("eggplant", "beringela"), "CEBOLA": ("onion", "cebola"), "CEBOLINHA": ("shallot", "cebolinha"),
    "CEREJA": ("cherry", "cereja"), "CHUCHU": ("chayote", "chuchu"), "CITRINOS": ("orange", "laranja"),
    "CLEMENTINA": ("clementine", "clementina"), "COUVE": ("cabbage", "couve"),
    "COUVE BROCOLO": ("broccoli", "brócolo"), "COUVE FLOR": ("cauliflower", "couve-flor"),
    "CURGETE": ("zucchini", "curgete"), "FEIJAO": ("green beans", "feijão verde"), "GENGIBRE": ("ginger", "gengibre"),
    "GOIABA": ("guava", "goiaba"), "GRANADILHA": ("granadilla", "granadilha"), "KIWI": ("kiwi", "kiwi"),
    "LIMA": ("lime", "lima"), "LIMAO": ("lemon", "limão"), "MACA": ("apple", "maçã"), "MAMAO": ("papaya", "mamão"),
    "MANGA": ("mango", "manga"), "MARACUJA": ("passion fruit", "maracujá"), "MELANCIA": ("watermelon", "melancia"),
    "MELAO": ("melon", "melão"), "MELOA": ("melon", "meloa"), "MORANGO": ("strawberry", "morango"),
    "NECTARINA": ("nectarine", "nectarina"), "PAPAIA": ("papaya", "papaia"), "PEPINO": ("cucumber", "pepino"),
    "PERA": ("pear", "pera"), "PESSEGO": ("peach", "pêssego"), "PIMENTO": ("bell pepper", "pimento"),
    "PITAYA": ("dragon fruit", "pitaya"), "TOMATE": ("tomato", "tomate"), "UVA": ("grape", "uva"),
}


def generic_type(name: str) -> str:
    words = slugify(name).split("_")
    i = next((i for i, w in enumerate(words) if w not in MODIFIERS), 0)
    if " ".join(words[i:i + 2]) in TWO_WORD_TYPES:
        return " ".join(words[i:i + 2])
    return ABBREVIATIONS.get(words[i], words[i])


def sale_type(name: str) -> str:
    """'packaged' when the name carries a pack weight, 'loose' when sold by the kilo."""
    return "packaged" if PACK_QTY.search(name) else "loose"


def local_time(image_file: str) -> datetime:
    return datetime.fromtimestamp(int(Path(image_file).stem), TZ)


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("Frutas e Legumes (train & test)"))
    ap.add_argument("--planogram", type=Path, default=Path("planogram.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/areas/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    args = ap.parse_args()

    planogram = load_planogram(args.planogram)
    areas = [a for cam_areas in planogram.values() for a in cam_areas]
    images = {cam: sorted(p.name for p in (args.images / cam).glob("*.jpg")) for cam in planogram}
    sizes = Counter(cv2.imread(str(args.images / cam / f)).shape[1::-1] for cam, fs in images.items() for f in fs)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    all_times = [local_time(f) for fs in images.values() for f in fs]

    # Products
    products = {}
    for a in areas:
        p = products.setdefault(a.product_id, {"product_id": a.product_id, "product_name": a.product_name,
                                               "generic_type": generic_type(a.product_name),
                                               "sale": sale_type(a.product_name), "area_ids": [], "cameras": []})
        p["area_ids"].append(a.area_id)
        if a.camera_id not in p["cameras"]:
            p["cameras"].append(a.camera_id)
    product_rows = sorted(products.values(), key=lambda p: (p["generic_type"], p["product_name"]))
    by_generic = defaultdict(lambda: [0, 0])
    for p in product_rows:
        by_generic[p["generic_type"]][0] += 1
        by_generic[p["generic_type"]][1] += len(p["area_ids"])
    sale_counts = Counter(p["sale"] for p in product_rows)

    # Partly visible areas (corners outside the frame)
    width, height = sizes.most_common(1)[0][0]
    visible = defaultdict(list)
    for r in manifest:
        visible[r["area_id"]].append(float(r["visible_frac"]))
    partial = []
    for a in areas:
        off = [c for c, (x, y) in zip(CORNERS, a.quad.tolist()) if not (0 <= x <= width and 0 <= y <= height)]
        if off:
            partial.append([a.area_id, a.product_name, ", ".join(off), f"{min(visible[a.area_id]):.0%}",
                            len(visible[a.area_id])])
    partial_crops = sum(r[4] for r in partial)

    general = [
        ["Cameras", len(planogram)],
        ["Images", sum(len(fs) for fs in images.values())],
        ["Image size", ", ".join(f"{w}×{h} ({n})" for (w, h), n in sizes.items())],
        ["Capture date", f"{min(all_times):%Y-%m-%d} ({TZ.key}, UTC{min(all_times):%z})"],
        ["Time range", f"{min(all_times):%H:%M} – {max(all_times):%H:%M}"],
        ["Frames per camera", f"{min(map(len, images.values()))}–{max(map(len, images.values()))}"],
        ["Planogram areas", len(areas)],
        ["Areas per camera", f"{min(map(len, planogram.values()))}–{max(map(len, planogram.values()))}"],
        ["Distinct products", len(products)],
        ["Loose / packaged products", f"{sale_counts['loose']} / {sale_counts['packaged']}"],
        ["Generic product types", len(by_generic)],
        ["Area crops (areas × frames)", len(manifest)],
        ["Partly visible areas", f"{len(partial)} ({partial_crops} crops)"],
    ]
    camera_rows = [
        [cam, len(cam_areas), len({a.product_id for a in cam_areas}), len(images[cam]),
         " ".join(f"{local_time(f):%H:%M}" for f in images[cam])]
        for cam, cam_areas in planogram.items()
    ]

    md = [
        "# Dataset summary",
        "",
        "Generated by `scripts/analyze_dataset.py`; do not edit by hand.",
        "",
        "## General",
        "",
        md_table(["", ""], general),
        "",
        "## Per camera",
        "",
        md_table(["Camera", "Areas", "Products", "Frames", "Times of day (local)"], camera_rows),
        "",
        "## Products per generic type",
        "",
        md_table(["Generic type", "Products", "Areas"],
                 [[g, n, na] for g, (n, na) in sorted(by_generic.items(), key=lambda kv: (-kv[1][0], kv[0]))]),
        "",
        "## Products",
        "",
        "`sale`: *loose* is sold by the kilo (a unit is one fruit), *packaged* carries a pack weight "
        "(a unit is one package).",
        "",
        md_table(["Product id", "Name", "Generic type", "Sale", "Areas", "Area ids", "Cameras"],
                 [[p["product_id"], p["product_name"], p["generic_type"], p["sale"], len(p["area_ids"]),
                   " ".join(p["area_ids"]), " ".join(p["cameras"])] for p in product_rows]),
        "",
        "## Partly visible areas (negative or out-of-frame coordinates)",
        "",
        md_table(["Area", "Product", "Corners off-frame", "Visible", "Crops"], partial),
        "",
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "DATASET.md").write_text("\n".join(md), encoding="utf-8")

    with open(args.out / "products.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["product_id", "product_name", "generic_type", "generic_en", "generic_pt", "sale",
                         "n_areas", "area_ids", "cameras"])
        for p in product_rows:
            en, pt = GENERIC_NAMES[p["generic_type"]]
            writer.writerow([p["product_id"], p["product_name"], p["generic_type"], en, pt, p["sale"],
                             len(p["area_ids"]), " ".join(p["area_ids"]), " ".join(p["cameras"])])

    print("\n".join(f"{k}: {v}" for k, v in general))
    print(f"\nwrote {args.out / 'DATASET.md'} and {args.out / 'products.csv'}")


if __name__ == "__main__":
    main()
