"""Turn predictions into the staff to-do list that the app shows (see app/server.py).

Only the newest photo of each camera describes the shelf now; earlier photos only tell since when an
area has been in its current state. Each area with a problem becomes one task:
- empty / half stock -> "restock" (status empty / low);
- planogram_compliant false -> "wrong_product" (status wrong; it wins over the stock level);
- a task whose confidence is not high becomes "check" (status unchanged): the model is unsure, so staff
  should take a look rather than act on it blindly.
Tasks are sorted with no invented weights: empty, wrong product, half; within each, sure tasks before
"check" ones (an unsure empty shelf still comes before a sure half one), then the oldest problem first.
"""
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from shelfmon.model import confidence_level

TZ = ZoneInfo("Europe/Lisbon")
STATUS_ORDER = {"empty": 0, "wrong": 1, "low": 2}
STATUS_LABELS = {"empty": "Empty", "low": "Half", "wrong": "Wrong product", "ok": "OK"}
TYPE_LABELS = {"restock": "Restock", "wrong_product": "Wrong product", "check": "Please check"}
COUNT_WORDS = {"empty": "empty", "low": "half", "wrong": "wrong product"}
STOCK_VALUES = {"empty": 0, "half": 0.5, "full": 1}


def load_display_names(path: Path) -> dict[str, tuple[str, str]]:
    """product id -> (display name, pack), from the hand-checked analysis/product_display.csv."""
    with open(path, encoding="utf-8") as f:
        return {r["product_id"]: (r["display_name"], r["pack"]) for r in csv.DictReader(f)}


def capture_time(image_file: str) -> datetime:
    """Photos are named by their Unix capture time."""
    return datetime.fromtimestamp(int(Path(image_file).stem), TZ)


def area_status(entry: dict) -> str:
    if not entry["planogram_compliant"]:
        return "wrong"
    return {"empty": "empty", "half": "low", "full": "ok"}[entry["stock_level"]]


def apply_corrections(predictions: dict, corrections: dict) -> dict:
    """Predictions with staff corrections (same official format) on top."""
    out = {cam: {img: dict(areas) for img, areas in imgs.items()} for cam, imgs in predictions.items()}
    for cam, imgs in corrections.items():
        for img, areas in imgs.items():
            if img in out.get(cam, {}):
                out[cam][img].update(areas)
    return out


def correction_entry(current: dict, stock_level: str | None = None, planogram_compliant: bool | None = None) -> dict:
    """Official-format entry: the current one with the fields staff corrected."""
    entry = dict(current)
    if stock_level is not None:
        if stock_level not in STOCK_VALUES:
            raise ValueError(f"stock_level must be one of {list(STOCK_VALUES)}")
        entry |= {"stock_level": stock_level, "stock_value": STOCK_VALUES[stock_level]}
    if planogram_compliant is not None:
        entry["planogram_compliant"] = bool(planogram_compliant)
    return entry


def latest_images(predictions: dict) -> dict[str, str]:
    """camera id -> file name of its newest photo."""
    return {cam: max(imgs, key=lambda f: int(Path(f).stem)) for cam, imgs in predictions.items() if imgs}


def _counts_phrase(statuses: list[str]) -> str:
    counts = Counter(statuses)
    return ", ".join(f"{counts[s]} {COUNT_WORDS[s]}" for s in STATUS_ORDER if counts[s])


def build_dashboard(planogram: dict, predictions: dict, details: dict, thresholds: dict, names: dict,
                    corrections: dict | None = None, done: set[str] | None = None,
                    image_url=lambda cam, img: "", frame_size=(1280, 720)) -> dict:
    """Everything the one-screen app shows. thresholds: the model bundle's "confidence" entry."""
    corrections, done = corrections or {}, done or set()
    effective = apply_corrections(predictions, corrections)
    latest = latest_images(effective)

    def name(product_id: str, fallback: str) -> tuple[str, str]:
        return names.get(product_id, (fallback.title(), ""))

    cameras, tasks = [], []
    for cam in sorted(latest, key=int):
        img = latest[cam]
        history = sorted(effective[cam], key=lambda f: int(Path(f).stem))
        when = capture_time(img)
        cam_areas = []
        for area in planogram[cam]:
            entry = effective[cam][img][area.area_id]
            status = area_status(entry)
            display, pack = name(area.product_id, area.product_name)
            number = int(area.area_id.rsplit("_", 1)[1])
            cam_areas.append({"area_id": area.area_id, "number": number, "product": display, "pack": pack,
                              "status": status, "quad": [[round(float(x)), round(float(y))] for x, y in area.quad]})
            if status == "ok":
                continue

            since_img = img
            for earlier in reversed(history[:-1]):
                if area_status(effective[cam][earlier][area.area_id]) != status:
                    break
                since_img = earlier
            since = capture_time(since_img)

            corrected = area.area_id in corrections.get(cam, {}).get(img, {})
            kind = "wrong_product" if status == "wrong" else "restock"
            d = details.get(cam, {}).get(img, {}).get(area.area_id)
            conf = None  # P(task needed); None when staff set this area by hand
            if not corrected and d is not None:
                conf = d.get("p_wrong", 1 - d["p_compliant"]) if kind == "wrong_product" \
                    else d["p_stock"][0] + d["p_stock"][1]
            confidence = "high" if conf is None else confidence_level(conf, thresholds[kind])

            tasks.append({
                "task_id": area.area_id,
                "type": kind if confidence == "high" else "check",
                "status": status,
                "status_label": STATUS_LABELS[status],
                "product": display,
                "pack": pack,
                "product_id": area.product_id,
                "product_raw": area.product_name,
                "stock_level": entry["stock_level"],
                "camera_id": cam,
                "area_id": area.area_id,
                "area_number": number,
                "since": since.strftime("%H:%M"),
                "since_ts": int(since.timestamp()),
                "confidence": confidence,
                "confidence_pct": None if conf is None else round(100 * conf),  # None: set by staff
                "done": area.area_id in done,
            })
        cameras.append({"camera_id": cam, "photo_time": when.strftime("%H:%M"),
                        "image_url": image_url(cam, img), "width": frame_size[0], "height": frame_size[1],
                        "areas": cam_areas})

    tasks.sort(key=lambda t: (STATUS_ORDER[t["status"]], t["type"] == "check", t["since_ts"],
                              int(t["camera_id"]), t["area_number"]))
    by_id = {t["task_id"]: t for t in tasks}

    def group(key: str, title: str, ids: list[str]) -> dict:
        open_ids = [i for i in ids if not by_id[i]["done"]]
        phrase = _counts_phrase([by_id[i]["status"] for i in open_ids]) or "All done"
        return {"key": key, "title": title, "subtitle": phrase, "task_ids": ids}

    def grouped(key_of) -> list[list[str]]:
        groups = {}
        for t in tasks:  # already sorted, so groups come out ordered by their most urgent task
            groups.setdefault(key_of(t), []).append(t["task_id"])
        return list(groups.items())

    location = [group(cam, f"Camera {cam}", ids) for cam, ids in grouped(lambda t: t["camera_id"])]
    product = []
    for pid, ids in grouped(lambda t: t["product_id"]):
        t0 = by_id[ids[0]]
        product.append(group(pid, t0["product"] + (f" · {t0['pack']}" if t0["pack"] != "loose" else ""), ids))
    by_type = [group(k, TYPE_LABELS[k], ids) for k, ids in grouped(lambda t: t["type"])]

    # Shelf KPIs as staff leave it: a task marked done counts as fixed (restocked / product put back).
    fixed = {t["area_id"] for t in tasks if t["done"]}
    areas = [(a.area_id, effective[cam][latest[cam]][a.area_id]) for cam in latest for a in planogram[cam]]
    n = max(1, len(areas))
    open_tasks = [t for t in tasks if not t["done"]]
    newest = max((capture_time(img) for img in latest.values()), default=None)
    return {
        "store": "Continente Bom Dia S. Romão",
        "photos_until": newest.strftime("%H:%M") if newest else None,
        "date": newest.strftime("%-d %b %Y") if newest else None,
        "summary": {
            "availability_pct": round(100 * sum(e["stock_level"] != "empty" or a in fixed for a, e in areas) / n),
            "compliance_pct": round(100 * sum(e["planogram_compliant"] or a in fixed for a, e in areas) / n),
            "todo": len(open_tasks),
            "done": len(tasks) - len(open_tasks),
        },
        "cameras": cameras,
        "tasks": tasks,
        "groups": {"location": location, "product": product, "type": by_type},
    }
