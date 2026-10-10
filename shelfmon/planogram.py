import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass
class Area:
    area_id: str
    camera_id: str
    product_id: str
    product_name: str
    quad: np.ndarray  # float32 (4, 2), ordered TL, TR, BR, BL; may lie partly off-image


def load_planogram(path: str | Path) -> dict[str, list[Area]]:
    """Return the planogram areas keyed by camera id."""
    cameras = json.loads(Path(path).read_text())["Frutas e Legumes"]["cameras"]
    return {
        cam: [
            Area(
                area_id=p["area_id"],
                camera_id=cam,
                product_id=p["product_id"],
                product_name=p["product_name"],
                quad=np.array([p["image_coordinates"][c] for c in CORNERS], dtype=np.float32),
            )
            for p in data["products"]
        ]
        for cam, data in cameras.items()
    }


def slugify(name: str) -> str:
    """ASCII, filesystem-safe version of a product name: 'MAÇÃ 75/85 KG' -> 'MACA_75_85_KG'."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")
