# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

JunctionX hackathon entry for the Sensei challenge "Fresh eyes on fresh produce!" (full brief: `Fruits & Vegetables - Shelf Monitoring Solution.pdf`). Only stage 1 (area extraction) exists so far. There are no tests or linters yet.

## Commands

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.14, numpy + opencv-python-headless + pillow
.venv/bin/python scripts/extract_areas.py                            # all cameras -> data/areas/
.venv/bin/python scripts/extract_areas.py --cameras 6215 6383 --size 256 --images <dir> --out <dir>
```

The user's shell is fish, so call `.venv/bin/python` directly rather than `source .venv/bin/activate`.

## Code layout

- `shelfmon/`: reusable library code, shared by the offline scripts and the eventual validation-set inference.
  - `planogram.py`: `load_planogram()` returns `{camera_id: [Area]}`, where `Area.quad` is float32 `[TL, TR, BR, BL]`. `slugify()` gives ASCII, filename-safe product names.
  - `rectify.py`: `rectify(image, quad, size)` homography-warps an area quad to a fronto-parallel rectangle at its native aspect ratio, then letterboxes it into a `size×size` square. Letterbox padding and off-image parts are filled with grey `(114,114,114)`. It also returns a `valid_mask` (pixels that came from inside the frame) and `visible_frac`. Use the mask, not the grey colour, to ignore padding.
  - `draw.py`: `draw_boxes()` draws yellow area outlines, and `draw_annotated()` adds in-place labels plus a legend strip below the frame (label stripes and legend swatches are coloured per product). A label shows the full name where it fits, otherwise just the area number. It prefers a system TTF (Liberation/DejaVu/Arial), because Pillow's bundled font lacks glyphs like `Ã`, and falls back to stripping accents.
- `scripts/extract_areas.py`: stage 1 (~12 s for the whole dataset). Writes one folder per image, `data/areas/<camera_id>/<timestamp>/`, holding `<area_id>_<PRODUCT_SLUG>.jpg` crops plus that frame's `_overview.jpg` (annotated) and `_overview_boxes.jpg` (outlines only). It also writes `data/areas/manifest.csv` (`camera_id, image_file, area_id, product_id, product_name, crop_path, visible_frac, native_w, native_h`). `data/` is gitignored and can be regenerated.
- `scripts/annotate.py` + `annotate.html`: stdlib-only labelling UI (`.venv/bin/python scripts/annotate.py`, then http://localhost:8000). It reads `data/areas/manifest.csv`, so run `extract_areas.py` first. It shows each timestamp's `_overview.jpg` with clickable area polygons and the selected area's rectified crop. On start it fills `annotations/annotations.json` (submission format) with the default `{true, "half", 0.5}` for every area without overwriting existing labels. Checked images are listed in `annotations/annotations.reviewed.json`, and only those count as real labels.

## The task

Ceiling-camera images of the produce section of an autonomous store (Continente Bom Dia S. Romão, Leiria). For each image and each planogram area, the app must output:
1. **planogram_compliant**: whether the product in the area is the one the planogram expects.
2. **stock_level / stock_value**: `empty`/0, `half`/0.5 or `full`/1.

Detection/classification and per-item counting are optional intermediate steps. Only the two fields above are scored. There are **no labels** for train/test: the solution must be unsupervised, zero-shot or self-supervised. A hidden validation set (26 images, 1–2 per camera, 289 areas) is scored when organisers request it. Candidates run their own code on those images, so the pipeline must accept an arbitrary directory of images keyed by camera id.

## Data

- `Frutas e Legumes (train & test)/<camera_id>/<unix_timestamp>.jpg`: 14 cameras, 12–14 images each (181 total), 1280×720. Captured roughly hourly over one day (2025-06-27, ~05:00–19:00 UTC). Train and test are not separated on disk. The path contains spaces, `&` and parentheses, so always quote it.
- `planogram.json`: `{"Frutas e Legumes": {"cameras": {<camera_id>: {"products": [{area_id, product_id, product_name, image_coordinates}]}}}}`. There are 152 areas and 94 distinct product names.
  - `area_id` = `<camera_id>_<n>`. `image_coordinates` gives the 4 corners (`top_left`, `top_right`, `bottom_right`, `bottom_left`) as `[x, y]` pixels. Areas are perspective quadrilaterals, not rectangles.
  - Some coordinates are **negative** (areas partly off-image, e.g. 6383_5–7, 6386_3–4). Clip polygons to the image.
  - The same product can occupy several areas (e.g. BANANA CAT I CNT KG ×6 on camera 6215). Score per area, aggregate per product only for display.
  - The planogram reflects the **start of the day**. Products may be moved or swapped later, which is what non-compliance means.
  - Names are Portuguese. A name ending in `KG` = loose (unit = one fruit). A name with a weight (`500 G`, `EMB. 3KG`, `1.5 KG`) = packaged (unit = one package). Some products look nearly identical (several apple, orange and potato variants), so compliance must work at a generic level (banana, orange, potato).

## Submission format (must match exactly)

```json
{"6215": {"1751052241.jpg": {"6215_2": {"planogram_compliant": true, "stock_level": "half", "stock_value": 0.5}}}}
```

Emit an entry for **every** area of the camera in every image. A missing area scores −5.

## Scoring (drives design trade-offs)

Per area, max +2:
- Compliance: correct +1, wrong −2.
- Stock: correct +1, predicted **below** truth −2, predicted **above** truth −3.
- Missing area: −2 −3.

Overestimating stock is the most expensive error. When stock is uncertain, lean towards the lower level. Most areas are compliant (the PDF's example run has 270/289 correct), so default to `true` unless there's strong evidence of a different product. UI/UX (staff-facing: what to restock or fix first) is judged 1–5 and only breaks ties.
