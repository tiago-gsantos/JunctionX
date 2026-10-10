"""Official challenge scoring: per labelled area, +1/-2 for compliance and +1/-2/-3 for stock level."""
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ID_POINTS = {"id_correct": 1, "id_wrong": -2}
STOCK_POINTS = {"stock_correct": 1, "stock_below": -2, "stock_above": -3}


@dataclass
class ScoreResult:
    total: int
    max: int
    counts: Counter = field(default_factory=Counter)  # id_*/stock_* cases, plus missing and extra areas
    per_camera: dict[str, tuple[int, int]] = field(default_factory=dict)  # camera -> (points, max)


def score_area(pred: dict | None, label: dict) -> tuple[int, int, str, str]:
    """Points for one area. A missing prediction (None) counts as wrong identification and stock above."""
    if pred is None:
        return ID_POINTS["id_wrong"], STOCK_POINTS["stock_above"], "id_wrong", "stock_above"
    id_case = "id_correct" if pred["planogram_compliant"] == label["planogram_compliant"] else "id_wrong"
    p, t = float(pred["stock_value"]), float(label["stock_value"])
    stock_case = "stock_correct" if p == t else "stock_below" if p < t else "stock_above"
    return ID_POINTS[id_case], STOCK_POINTS[stock_case], id_case, stock_case


def score(predictions: dict, labels: dict) -> ScoreResult:
    """Score predictions against labels, both in the official {camera: {image: {area_id: {...}}}} format."""
    counts = Counter()
    per_camera = defaultdict(lambda: [0, 0])
    for cam, images in labels.items():
        for image, areas in images.items():
            pred_areas = predictions.get(cam, {}).get(image, {})
            for area_id, label in areas.items():
                pred = pred_areas.get(area_id)
                id_pts, stock_pts, id_case, stock_case = score_area(pred, label)
                counts[id_case] += 1
                counts[stock_case] += 1
                counts["missing"] += pred is None
                per_camera[cam][0] += id_pts + stock_pts
                per_camera[cam][1] += 2
    counts["extra"] = sum(
        1 for cam, images in predictions.items() for image, areas in images.items()
        for area_id in areas if area_id not in labels.get(cam, {}).get(image, {})
    )
    total = sum(points for points, _ in per_camera.values())
    max_points = sum(m for _, m in per_camera.values())
    return ScoreResult(total, max_points, counts, {cam: tuple(v) for cam, v in sorted(per_camera.items())})


def load_annotations(paths: list[str | Path]) -> dict:
    """Load and merge official-format JSON files; directories contribute every *.json inside them."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    merged = {}
    for f in files:
        for cam, images in json.loads(f.read_text(encoding="utf-8")).items():
            for image, areas in images.items():
                target = merged.setdefault(str(cam), {}).setdefault(image, {})
                for area_id, entry in areas.items():
                    if area_id in target and target[area_id] != entry:
                        raise ValueError(f"conflicting entries for {cam}/{image}/{area_id} (latest from {f})")
                    target[area_id] = entry
    return merged
