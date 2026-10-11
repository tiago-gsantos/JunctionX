# Accuracy experiments

All scores come from the 7-round time CV of `scripts/train_eval.py`, run in memory by `common.py`, out of 3,924. "Newest photos" is the score on each camera's two newest photos (out of 608). CV is optimistic for these, because they have no later neighbours in time. Gains under ~50 CV points count as noise.

| Experiment | CV total | vs current | Compliance | Stock | Newest photos | Verdict |
|---|---|---|---|---|---|---|
| Current model (hgb, handcrafted + radio, album override) | +3,585 | — | +1,926 | +1,659 | +519 | baseline |
| 0. Fullest reference = high texture + low visible container (label-free) | +3,566 | −19 | +1,923 | +1,643 | +508 | not kept: 141/152 references now full (was 133), but no gain |
| 1. + CLIPSeg produce/container features (zero-shot segmentation, single photo) | +3,588 | +3 | +1,926 | +1,662 | +520 | no gain: produce maps look right (0 on empty crates), but the model already captures fill level |
| 2a. Fine-tuned DINOv3 (last 4 blocks, stock + product heads) alone | +3,513 | −72 | +1,839 | +1,674 | +491 | better stock than the current model, weaker compliance (seed 1: +3,516) |
| **2b. Current model blended 50/50 with fine-tuned DINOv3 (seed 0)** | **+3,654** | **+69** | +1,941 | +1,713 | +524 | **gain** |
| 2b. Same, seed 1 | +3,647 | +62 | +1,941 | +1,706 | +516 | confirms the gain |
| 2b. Same, both seeds averaged | +3,643 | +58 | +1,941 | +1,702 | +516 | no extra from averaging |
| 2c. Blend stock only | +3,639 | +54 | +1,926 | +1,713 | +524 | most of the gain is stock |
| 2d. Blend compliance only | +3,600 | +15 | +1,941 | +1,659 | +519 | |

Local VLM benchmark (Qwen3-VL-2B-Instruct, Apache-2.0, fp16, 298 labelled crops, 0.34 s per crop for 2 questions):
- names the right produce on 80% of correct, non-empty crops;
- names a different produce on 72% of swapped crops (28/39);
- gets the fill level right on 66%.

| Experiment | CV total | vs current | Compliance | Stock | Newest photos | Verdict |
|---|---|---|---|---|---|---|
| 3. + VLM features (other kind named, fill level) | +3,543 | −42 | +1,923 | +1,620 | +502 | not kept: the noisy fill answers hurt stock; produce answers add nothing to compliance |
| 3+2. VLM features + fine-tuned blend | +3,647 | +62 | +1,941 | +1,706 | +524 | no better than 2b alone |

Earlier checks (same harness):
- remove DINOv3: +3,562;
- album needs 2 of 3 votes: +3,576;
- no same-day references: +3,426;
- single photo only (no references, no album): +3,402;
- isotonic recalibration of stock probabilities: +1 stock point.

## Conclusion
- **Keep:** the fine-tuned DINOv3 blend (`finetune_cv.py`). It gains about +60 in CV over two seeds, almost all in stock, and is flat on the newest photos. It isn't wired into `predict.py` or the app yet.
- **Not kept:** the reference fix, CLIPSeg segmentation and the local VLM. Their code was removed; the numbers above are the record.
