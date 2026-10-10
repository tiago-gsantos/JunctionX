# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

JunctionX hackathon entry for the Sensei challenge "Fresh eyes on fresh produce!" (full brief: `Fruits & Vegetables - Shelf Monitoring Solution.pdf`). Built so far: area extraction, a dataset summary, the official scorer and the format verifier. There's no linter.

## Commands

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.14; torch (CUDA 13), transformers, timm + open_clip_torch (needed by C-RADIOv4)
.venv/bin/python scripts/extract_areas.py                            # all cameras -> data/areas/
.venv/bin/python scripts/extract_areas.py --cameras 6215 6383 --size 256 --images <dir> --out <dir>
.venv/bin/python scripts/analyze_dataset.py                          # dataset summary -> analysis/ (needs the manifest above)
.venv/bin/python scripts/score.py --pred predictions.json --labels labels/   # official points (labels: files or folders)
.venv/bin/python scripts/verify.py labels/                          # every area present + valid values; exit 1 on errors
.venv/bin/python scripts/embed_crops.py --model siglip2               # crops -> data/features/<model>.npy (siglip2 | dinov2 | dinov3 | radio)
.venv/bin/python scripts/check_embeddings.py                         # label-free checks: nearest neighbour + SigLIP zero-shot
.venv/bin/python scripts/compute_features.py                         # features A+B+C+D -> data/features/handcrafted.csv; builds data/references/ (~1 min)
.venv/bin/python scripts/compute_cv_references.py                    # group C per CV round, references without held-out photos -> data/features/group_c_cv.npz (~1 min; rerun after compute_features)
.venv/bin/python scripts/compute_features.py --manifest <new>/manifest.csv --features-dir <dir> --references data/references/references.npz   # new photos (validation) vs saved references
.venv/bin/python scripts/baseline.py --stock full --out data/predictions/baseline_full.json   # constant answers (+1598)
.venv/bin/python scripts/train_eval.py --model hgb --features handcrafted dinov3   # 7-round time CV, scores + logs to results/LEADERBOARD.md
```

The user's shell is fish, so call `.venv/bin/python` directly rather than `source .venv/bin/activate`.

## Code layout

- `shelfmon/`: reusable library code, shared by the offline scripts and the eventual validation-set inference.
  - `planogram.py`: `load_planogram()` returns `{camera_id: [Area]}`, where `Area.quad` is float32 `[TL, TR, BR, BL]`. `slugify()` gives ASCII, filename-safe product names.
  - `rectify.py`: `rectify(image, quad, size)` homography-warps an area quad to a fronto-parallel rectangle at its native aspect ratio, then letterboxes it into a `size×size` square. Letterbox padding and off-image parts are filled with grey `(114,114,114)`. It also returns a `valid_mask` (pixels that came from inside the frame) and `visible_frac`. Use the mask, not the grey colour, to ignore padding.
  - `draw.py`: `draw_boxes()` draws yellow area outlines, and `draw_annotated()` adds in-place labels plus a legend strip below the frame (label stripes and legend swatches are coloured per product). A label shows the full name where it fits, otherwise just the area number. It prefers a system TTF (Liberation/DejaVu/Arial), because Pillow's bundled font lacks glyphs like `Ã`, and falls back to stripping accents.
  - `scoring.py`: the official points (`score`, `score_area`) and `load_annotations()`, which merges official-format JSON files or folders and rejects conflicting duplicates. A missing area scores −5; extra predicted areas are ignored.
  - `verify.py`: `verify(annotations, planogram, images)` checks an official-format dict. Errors: missing images or areas (−5 each in scoring), unknown cameras, images or areas, bad keys or values, `stock_value` not matching `stock_level`. Warning: an empty area marked non-compliant (team convention: empty ⇒ compliant). Run it on labels before training and on predictions before submitting.
  - `embeddings.py`: frozen pretrained encoders (`BACKBONES` registry, `Encoder.embed_images` / `embed_texts`, `cosine`). siglip2 = `google/siglip2-base-patch16-256` (image + text, 768-d); dinov2 = `facebook/dinov2-base` (768-d, run without centre crop); dinov3 = gated on Hugging Face (needs Meta's approval + `hf auth login`); radio = `nvidia/C-RADIOv4-SO400M` (NVIDIA remote code, upsamples crops to 512 px, 2304-d summary, ~5 min for all crops). Embeddings are saved in manifest row order to `data/features/<model>.npy` plus a `.json` with metadata.
  - `features.py`: handcrafted per-crop measurements on real shelf pixels only (`area_mask()` rebuilds the area's valid mask from its geometry, eroded 2 px). Group A, `container_features()`: `teal_frac` (empty teal crates), `low_colour_frac` (grey/metal trays, shelf covers), `dark_frac` (black trays), `mean_saturation`, `colourfulness`. HSV thresholds were set from the empty crates and trays of camera 6462 at 06:12. Absolute colour values differ by area, and dull products (cauliflower, maracujá, avocado, garlic, pale bananas) read as "low colour" even when full, so compare them within an area. Group B, `texture_features()`: `edge_frac` (Canny), `edge_frac_coarse` (Canny at half resolution, ignores crate holes), `laplacian` (log variance), measured 3 px away from the padding border. Texture separates dull-but-full areas (coarse edges ~0.19–0.29) from empty containers and covers (~0.00–0.14).
  - `references.py`: group C, label-free per-area references built from the area's photos of the day. *typical* = median of each A/B feature, pixel-wise median image, mean embedding; *fullest* = the area's photo with the highest `edge_frac_coarse` (its most stocked moment). Features per crop: `d_typ_*` / `d_full_*` (A/B value minus the reference's), `pix_diff_typ` / `pix_diff_full` (blurred mean colour difference on masked pixels), `emb_dist_typ_<bb>` / `emb_dist_full_<bb>` (1 − cosine, for siglip2 and dinov2). Saved to `data/references/references.npz` + `.json`; validation photos must be compared against these saved references (`--references`), not new ones. Group C flags change, not product identity: an area swapped all day (e.g. 6215_7/9) looks "normal" here.
  - `zeroshot.py`: group D (compliance), SigLIP image–text similarity. Each of the 94 products has an English description in `analysis/product_prompts.csv` (hand-written, editable; variety, colour and packaging matter, because variety swaps were labelled non-compliant). It's placed in 4 supermarket-scene `TEMPLATES` whose text embeddings are averaged; scene templates beat a plain "a photo of …". Features: `sim_expected`; four margins, each the expected product minus its best rival: `margin_camera` (other products on the same camera), `margin_camera_kind` (other *kinds* on the same camera; falls back to `margin_camera` on single-kind cameras 6443 and 6385), `margin_all` (the other 93 products), `margin_all_kind` (other kinds in the whole store); `empty_vs_expected` (best `EMPTY_PROMPTS` match minus expected; positive = looks empty). The product-level margins catch variety swaps but hover around 0 for look-alike varieties. The kind-level margins separate the common swaps more cleanly (a real swap like 6215_7/9 is about −0.1 vs +0.02 to +0.03 for correct areas).
  - `dataset.py`: `load_table()` joins manifest, labels, `handcrafted.csv` and embeddings in manifest order. By default it also loads `group_c_cv.npz`, and `Table.handcrafted_for_round(k)` swaps in round k's group C columns (references built without round k's photos, the same situation as the hidden validation photos). `time_folds()`: round k holds out photos 2k and 2k+1 (by time of day) of every camera (7 rounds), mimicking the hidden set's 1–2 photos per camera.
  - `scoring.py` also has the decision rules: `decide_stock()` picks the level with the highest expected points from P(empty, half, full) (`STOCK_PAYOFF`; it leans low when unsure), and `decide_compliance()` thresholds at 0.5 (the penalties are symmetric).
- `scripts/extract_areas.py`: stage 1 (~12 s for the whole dataset). Writes one folder per image, `data/areas/<camera_id>/<timestamp>/`, holding `<area_id>_<PRODUCT_SLUG>.jpg` crops plus that frame's `_overview.jpg` (annotated) and `_overview_boxes.jpg` (outlines only). It also writes `data/areas/manifest.csv` (`camera_id, image_file, area_id, product_id, product_name, crop_path, visible_frac, native_w, native_h`). `data/` is gitignored and can be regenerated.
- `scripts/annotate.py` + `annotate.html`: stdlib-only labelling UI (`.venv/bin/python scripts/annotate.py`, then http://localhost:8000). It reads `data/areas/manifest.csv`, so run `extract_areas.py` first. It shows each timestamp's `_overview.jpg` with clickable area polygons and the selected area's rectified crop. On start it fills `annotations/annotations.json` (submission format) with the default `{true, "half", 0.5}` for every area without overwriting existing labels. Checked images are listed in `annotations/annotations.reviewed.json`, and only those count as real labels.
- `scripts/analyze_dataset.py`: writes `analysis/DATASET.md` (cameras, frames and times of day, areas, products by generic type, partly visible areas) and `analysis/products.csv`. Both are committed; regenerate them rather than editing by hand. The generic type is the first word of the accent-stripped product name (e.g. `MACA`), with `MODIFIERS`/`ABBREVIATIONS` handling `MINI TOMATE` and `CITR`, and `TWO_WORD_TYPES` keeping `BATATA DOCE`, `COUVE BROCOLO` and `COUVE FLOR` as their own types. The loose/packaged split comes from the `PACK_QTY` regex. `products.csv` also has `generic_en` / `generic_pt` (from `GENERIC_NAMES`), used for zero-shot text prompts; English prompts beat Portuguese ones.
- `scripts/train_eval.py`: models `logreg` / `hgb` (HistGradientBoosting) / `knn` (same-area neighbours) on `handcrafted` and/or embeddings (PCA 64, fitted inside each round). It fits a stock and a compliance model per round, scores the out-of-round predictions with both decision rules, writes `data/predictions/<run>.json` and appends to `results/LEADERBOARD.md`. Group C references are honest by default; `--shared-references` reproduces the older, slightly optimistic protocol (runs tagged `@shared` in the leaderboard). So far, `hgb` on handcrafted + one embedding (radio or dinov3) is best, at about +3,520 to +3,540 of 3,924 (baseline +1,598). Score differences under ~50 points are within noise.

## The task

Ceiling-camera images of the produce section of an autonomous store (Continente Bom Dia S. Romão, Leiria). For each image and each planogram area, the app must output:
1. **planogram_compliant**: whether the product in the area is the one the planogram expects.
2. **stock_level / stock_value**: `empty`/0, `half`/0.5 or `full`/1.

Detection/classification and per-item counting are optional intermediate steps. Only the two fields above are scored. The organisers provide **no labels** for train/test. The team is hand-labelling all area crops in the official format (convention: empty ⇒ compliant = true, except a lone wrong item left in an otherwise empty area, which is labelled empty + non-compliant), and those labels may be used for training. Labels are in `annotations/annotations.json`; all 181 images are reviewed. A hidden validation set (26 images, 1–2 per camera, 289 areas) is scored when organisers request it. Candidates run their own code on those images, so the pipeline must accept an arbitrary directory of images keyed by camera id.

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
