"""Write an HTML page with the model's stock and compliance errors, crops included (one self-contained file).

Two runs side by side:
- CV: out-of-fold predictions of every labelled photo (scripts/train_eval.py writes
  data/predictions/<run>.json + .proba.npz);
- Forward: the demo model on each camera's newest photos, which it never saw (the app's demo run,
  data/demo/runs/current), the closest stand-in for the hidden set.

    .venv/bin/python scripts/error_report.py          # -> data/analysis/errors.html, open it in a browser
"""
import argparse
import base64
import csv
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shelfmon.planogram import load_planogram  # noqa: E402
from shelfmon.scoring import STOCK_LEVELS, load_annotations, score, task_points  # noqa: E402
from shelfmon.tasks import load_display_names  # noqa: E402

TZ = ZoneInfo("Europe/Lisbon")
CASES = [("id_correct", "Product correct", 1), ("id_wrong", "Product wrong", -2), ("stock_correct", "Stock correct", 1),
         ("stock_below", "Stock below the real level", -2), ("stock_above", "Stock above the real level", -3)]


def thumbnail(path: Path, size: int = 176) -> str:
    img = cv2.resize(cv2.imread(str(path)), (size, size), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1]).decode()


def load_run(name, predictions: Path, areas_dir: Path, probs: dict) -> dict:
    with open(areas_dir / "manifest.csv", encoding="utf-8") as f:
        crops = {(r["camera_id"], r["image_file"], r["area_id"]): areas_dir / r["crop_path"] for r in csv.DictReader(f)}
    return {"name": name, "pred": json.loads(predictions.read_text()), "crops": crops, "probs": probs}


def cv_probs(npz: Path) -> dict:
    d = np.load(npz)
    return {tuple(k): (d["p_stock"][i], d["p_compliant"][i], bool(d["override_flagged"][i])) for i, k in enumerate(d["keys"])}


def forward_probs(details: Path) -> dict:
    d = json.loads(details.read_text())
    return {(c, i, a): (np.array(v["p_stock"]), v["p_compliant"], v.get("override", False))
            for c, imgs in d.items() for i, areas in imgs.items() for a, v in areas.items()}


def section(run, labels, names, products) -> str:
    pred = run["pred"]
    lab = {c: {i: labels[c][i] for i in imgs} for c, imgs in pred.items()}
    result = score(pred, lab)
    comp, stock = task_points(result)
    n = result.counts["id_correct"] + result.counts["id_wrong"]
    rows = "".join(f"<tr><td>{label}</td><td>{result.counts[k]}</td><td>{result.counts[k] / n:.1%}</td>"
                   f"<td>{result.counts[k] * pts:+}</td></tr>" for k, label, pts in CASES)
    summary = (f"<table class='sum'><tr><th>Case</th><th>Areas</th><th>Share</th><th>Points</th></tr>{rows}"
               f"<tr class='tot'><td>Total ({n} areas)</td><td colspan=2>{result.total / result.max:.1%} of max</td>"
               f"<td>{result.total:+} / {result.max}</td></tr></table>"
               f"<p class='muted'>Compliance {comp:+} · stock {stock:+}</p>")

    stock_err, comp_err = [], []
    for cam, imgs in pred.items():
        for img, areas in imgs.items():
            for area, p in areas.items():
                t = lab[cam][img][area]
                key = (cam, img, area)
                ps, pc, flagged = run["probs"].get(key, (np.full(3, np.nan), np.nan, False))
                name, pack = names.get(products[area], (area, ""))
                info = {"area": area, "product": name + (f" · {pack}" if pack and pack != "loose" else ""),
                        "time": datetime.fromtimestamp(int(img[:-4]), TZ).strftime("%H:%M"), "img": run["crops"][key]}
                if p["stock_value"] != t["stock_value"]:
                    above = p["stock_value"] > t["stock_value"]
                    probs = " · ".join(f"{lvl} {v:.0%}" for lvl, v in zip(STOCK_LEVELS, ps))
                    stock_err.append((0 if above else 1, area, info | {
                        "kind": "Above (−3)" if above else "Below (−2)", "cls": "above" if above else "below",
                        "line": f"real <b>{t['stock_level']}</b> → predicted <b>{p['stock_level']}</b>", "probs": probs}))
                if p["planogram_compliant"] != t["planogram_compliant"]:
                    missed = not t["planogram_compliant"]
                    comp_err.append((0 if missed else 1, area, info | {
                        "kind": "Missed wrong product (−2)" if missed else "False alarm (−2)",
                        "cls": "above" if missed else "below",
                        "line": f"real stock <b>{t['stock_level']}</b> · P(right product) {pc:.0%}"
                                + (" · album flagged" if flagged else ""), "probs": ""}))

    def cards(errors):
        return "".join(
            f"<figure class='card {e['cls']}'><img src='{thumbnail(e['img'])}' alt=''>"
            f"<figcaption><span class='kind'>{e['kind']}</span><b>{html.escape(e['product'])}</b>"
            f"<span>{e['area']} · {e['time']}</span><span>{e['line']}</span>"
            f"<span class='muted'>{e['probs']}</span></figcaption></figure>"
            for _, _, e in sorted(errors, key=lambda x: (x[0], x[1], x[2]["time"])))

    return (f"<section class='run' data-run='{run['name']}'>{summary}"
            f"<h2>Stock errors <small>{len(stock_err)}</small></h2><div class='grid'>{cards(stock_err)}</div>"
            f"<h2>Compliance errors <small>{len(comp_err)}</small></h2><div class='grid'>{cards(comp_err)}</div></section>")


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Model errors</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{margin:0;padding:24px;background:#141513;color:#f8f7f4;font:14px/1.45 system-ui,sans-serif}
h1{font-weight:400;margin:0 0 4px} h2{font-weight:500;margin:32px 0 12px} h2 small{color:#a5a99f;font-weight:400}
.muted{color:#a5a99f} .tabs{display:flex;gap:8px;margin:16px 0 8px}
.tabs button{background:#232521;color:#f8f7f4;border:1px solid #3d4139;border-radius:4px;padding:8px 14px;cursor:pointer;font:inherit}
.tabs button.on{background:#f8fd8f;color:#1d1d1d;border-color:#f8fd8f}
.run{display:none}.run.on{display:block}
.sum{border-collapse:collapse;margin-top:8px}.sum td,.sum th{padding:4px 14px 4px 0;text-align:left}
.sum td:nth-child(n+2),.sum th:nth-child(n+2){text-align:right}.sum .tot td{border-top:1px solid #3d4139;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));gap:10px}
.card{margin:0;background:#1b1c1a;border:1px solid #2f322c;border-radius:6px;overflow:hidden}
.card.above{border-top:3px solid #ff8a7a}.card.below{border-top:3px solid #ffc56e}
.card img{display:block;width:100%;aspect-ratio:1;cursor:zoom-in}
figcaption{display:flex;flex-direction:column;gap:2px;padding:8px 10px 10px;font-size:12.5px}
figcaption b{font-size:14px}.kind{font-size:12px;color:#a5a99f}
#big{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;place-items:center}#big img{width:min(80vw,80vh)}
</style></head><body><h1>Model errors</h1>
<p class="muted">Model: __MODEL__. Red = costlier direction (stock above / missed wrong product), amber = stock below / false alarm.</p>
<div class="tabs">__TABS__</div>__SECTIONS__<div id="big"><img alt=""></div>
<script>
const show = (name) => { document.querySelectorAll(".run").forEach((s) => s.classList.toggle("on", s.dataset.run === name));
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.textContent === name)); };
document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => show(b.textContent));
show(document.querySelector(".tabs button").textContent);
const big = document.getElementById("big");
document.querySelectorAll(".card img").forEach((i) => i.onclick = () => { big.firstChild.src = i.src; big.style.display = "grid"; });
big.onclick = () => big.style.display = "none";
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-run", default="hgb_handcrafted+radio+override-proto", help="name in data/predictions/")
    ap.add_argument("--out", type=Path, default=ROOT / "data/analysis/errors.html")
    args = ap.parse_args()

    labels = load_annotations([ROOT / "annotations/annotations.json"])
    names = load_display_names(ROOT / "analysis/product_display.csv")
    products = {a.area_id: a.product_id for cam in load_planogram(ROOT / "planogram.json").values() for a in cam}
    preds = ROOT / "data/predictions"
    runs = [load_run("CV · all photos", preds / f"{args.cv_run}.json", ROOT / "data/areas",
                     cv_probs(preds / f"{args.cv_run}.proba.npz"))]
    forward = ROOT / "data/demo/runs/current"
    if (forward / "predictions.json").exists():
        runs.append(load_run("Forward · newest photos", forward / "predictions.json", forward / "areas",
                             forward_probs(forward / "details.json")))
    else:
        print("no demo run found (upload data/demo/photos in app/server.py --demo): forward section skipped")

    page = (PAGE.replace("__MODEL__", html.escape(args.cv_run))
            .replace("__TABS__", "".join(f"<button>{r['name']}</button>" for r in runs))
            .replace("__SECTIONS__", "".join(section(r, labels, names, products) for r in runs)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
