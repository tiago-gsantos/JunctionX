# Stock Sense

**Fresh eyes on fresh produce:** our JunctionX entry for the Sensei challenge.

Ceiling cameras photograph the fruit and vegetable section of Continente Bom Dia S. Romão. For every planogram area in every photo, Stock Sense answers two questions:
- **Is the right product there?**
- **Is it empty, half or full?**

The staff app turns the answers into a prioritised to-do list.

## Setup

Requires Python 3.14. A CUDA GPU is recommended: everything runs on a 6 GB card.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
hf auth login        # only for DINOv3, which is gated: request access on its Hugging Face page first
```

The trained model and its references are in `models/`, so you can predict and run the app right away. The three pretrained image models (SigLIP2, DINOv3, C-RADIOv4) download automatically the first time they are used.

**No access to DINOv3?** Add `--dinov2` to `predict.py` and `app/server.py`. That uses the same model built with the open DINOv2 instead (`models/dinov2/`). It scores 91.2% vs 91.4% in cross-validation, and needs no login.

## 1. Predict new photos (the validation set)

Put the photos in one folder with a sub-folder per camera. Keep the file names as they are (Unix timestamps):

```
validation/
  6215/1751052241.jpg
  6385/1751052310.jpg
  ...
```

Then run:

```sh
.venv/bin/python scripts/predict.py --images validation --out data/runs/validation
.venv/bin/python scripts/verify.py data/runs/validation/predictions.json --images validation
```

- **`predictions.json`** is the submission file, in the official format.
- **`verify.py`** must end with `OK: every area present and valid.`
- **Time:** about a minute for 26 photos.
- **No DINOv3 access:** add `--dinov2` to the `predict.py` command.

If the true labels are available, score the predictions:

```sh
.venv/bin/python scripts/score.py --pred data/runs/validation/predictions.json --labels <annotations.json>
```

## 2. Run the staff app

```sh
.venv/bin/python app/server.py
```

Open http://localhost:8000 and upload a photo folder (laid out as above), by button or drag and drop.

The app then shows:
- the to-do list, sorted by urgency;
- the camera photo with each area marked;
- **Mark done** and **AI was wrong** buttons;
- **Download results**, which gives the same `predictions.json`.

Useful options:
- `--port 8080` serves on another port.
- `--host 0.0.0.0` lets a tablet on the same network open it.
- `--load <folder>` processes a folder at start-up.
- `--dinov2` uses the DINOv2 model (no DINOv3 access needed).

Run only one server at a time: the models need about 4 GB of GPU memory.

## 3. Run the live demo

The demo uses a separate model, trained without each camera's two newest photos, so the 28 photos you upload are new to it. It needs the rebuild steps in section 4 to have run once (with DINOv3).

```sh
.venv/bin/python scripts/build_demo.py     # once, ~2 min
rm -rf data/demo/runs                      # start with an empty app
.venv/bin/python app/server.py --demo
```

Open http://localhost:8000 and upload the folder `data/demo/photos`. To run the demo again from scratch, stop the server, run `rm -rf data/demo/runs`, then start it again.

## 4. Retrain from scratch (optional)

From the photos in `Frutas e Legumes (train & test)/` and the labels in `annotations/annotations.json`:

```sh
.venv/bin/python scripts/extract_areas.py              # cut out every planogram area (~15 s)
.venv/bin/python scripts/embed_crops.py --model siglip2
.venv/bin/python scripts/embed_crops.py --model dinov3
.venv/bin/python scripts/embed_crops.py --model radio  # the slow one (~5 min)
.venv/bin/python scripts/compute_features.py           # features + references -> models/references.npz (~1 min)
.venv/bin/python scripts/compute_cv_references.py      # references for honest cross-validation (~1 min)
.venv/bin/python scripts/train_eval.py                 # cross-validation score, then models/final.joblib (~20 s)
```

`train_eval.py` prints the cross-validation score, then trains the final model on all labelled photos.

To rebuild the DINOv2 model in `models/dinov2/` (no DINOv3 needed: skip the `dinov3` step above):

```sh
.venv/bin/python scripts/embed_crops.py --model dinov2 --out data/features_dinov2
cp data/features/siglip2.npy data/features/radio.npy data/features_dinov2/
.venv/bin/python scripts/compute_features.py --dino dinov2 --features-dir data/features_dinov2 --save-references models/dinov2/references.npz
.venv/bin/python scripts/compute_cv_references.py --dino dinov2 --features-dir data/features_dinov2
.venv/bin/python scripts/train_eval.py --dino dinov2 --features-dir data/features_dinov2 --name final_dinov2 --out models/dinov2/final.joblib
```

## Other tools

| Command | What it does |
|---|---|
| `.venv/bin/python scripts/error_report.py` | Writes `data/analysis/errors.html`: every stock and compliance error, with crops |
| `.venv/bin/python scripts/annotate.py` | The labelling tool used to make `annotations/` (http://localhost:8000; run `extract_areas.py` first) |
| `.venv/bin/python scripts/analyze_dataset.py` | Rewrites the dataset summary in `analysis/` |

## How it works

1. **Cut out:** each planogram area is straightened into a square crop.
2. **Look:** three pretrained image models (SigLIP2, DINOv3, C-RADIOv4) describe each crop.
3. **Measure:** 36 handcrafted features per crop. They cover the colour of the empty container, texture, the difference from the area's usual and fullest look, and how well the crop matches a text description of its product.
4. **Decide:** gradient-boosted trees predict stock and compliance. A product "album" flags areas that look like a neighbour's product. The stock answer leans low when unsure, because overestimating stock costs the most points.

**Results**, in official points:

| Test | Score |
|---|---|
| Cross-validation on all 181 labelled photos | **91.4%** of the maximum |
| Each camera's newest photos, unseen by the model | **81.4%** |

## Project layout

```
app/          staff app: server.py + static page
scripts/      pipeline steps, training, prediction, scoring, tools
shelfmon/     shared library code
models/       trained model + references (models/dinov2/: the DINOv2 version)
annotations/  hand labels (official format)
analysis/     product names, prompts, dataset summary
planogram.json
```
