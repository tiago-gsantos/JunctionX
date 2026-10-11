# Stock Sense: fresh eyes on fresh produce

JunctionX entry for the Sensei challenge. Ceiling-camera photos of the produce section of Continente Bom Dia S. Romão are checked area by area against the planogram. For every area, the app reports whether the right product is there and whether it is empty, half or full. Staff get a prioritised to-do list.

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The pipeline uses three pretrained image models from Hugging Face (SigLIP2, DINOv3, C-RADIOv4), downloaded on first use. DINOv3 is gated: it needs Meta's approval and `hf auth login` first. A CUDA GPU is recommended (it runs on a 6 GB card).

The trained model (`models/final.joblib`) and its saved "usual shelf" references (`models/references.npz`) are committed, so the app and predictions on new photos work straight after setup. Rebuilding them is optional.

## Rebuild everything from the photos (optional)

The labels are in `annotations/annotations.json`. Intermediate data goes to `data/` (not committed); the model and references go to `models/`.

```sh
.venv/bin/python scripts/extract_areas.py                       # cut every planogram area out of every photo
.venv/bin/python scripts/embed_crops.py --model siglip2         # image embeddings (three models)
.venv/bin/python scripts/embed_crops.py --model dinov3
.venv/bin/python scripts/embed_crops.py --model radio
.venv/bin/python scripts/compute_features.py                    # handcrafted features + "usual shelf" references
.venv/bin/python scripts/compute_cv_references.py               # references for honest cross-validation
.venv/bin/python scripts/train_eval.py                         # cross-validate (+3,585 of 3,924), then train models/final.joblib
```

## Run on new photos (e.g. the validation set)

```sh
.venv/bin/python scripts/predict.py --images <dir> --out data/runs/validation   # <dir>/<camera_id>/<timestamp>.jpg
.venv/bin/python scripts/verify.py data/runs/validation/predictions.json --images <dir>
```

`predictions.json` is the submission file.

## The staff app

```sh
.venv/bin/python app/server.py          # http://localhost:8000, then upload a photo folder
```

For the live demo, a separate model is trained without the 28 photos you upload (needs the rebuild steps above first):

```sh
.venv/bin/python scripts/build_demo.py  # once (~2 min)
rm -rf data/demo/runs                   # start with no data
.venv/bin/python app/server.py --demo   # then upload data/demo/photos
```


## Layout

- `shelfmon/`: library code (planogram, rectification, features, references, zero-shot, product album, model bundle, scoring, verification, staff tasks).
- `scripts/`: pipeline steps, training/evaluation, prediction, scoring and verification, the labelling tool (`annotate.py`), the error report (`error_report.py`) and the dataset summary.
- `models/`: the trained model and its saved references.
- `app/`: the Stock Sense server and its static page.
- `analysis/`: dataset summary, product prompts and display names.
- `annotations/`: the hand labels (official format) used for training.
