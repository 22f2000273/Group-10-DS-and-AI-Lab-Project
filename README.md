# Understanding Cross-Domain Generalization in Semantic Segmentation: A Study Using Aerial Rooftop Imagery

**Group 10 — Data Science and AI Lab (DSAI Labs), IIT Madras BS Degree**
Project guide: Prof. M Nagarajan

## Overview

When a building-segmentation model is trained on aerial images from one dataset and tested on another, its accuracy drops sharply. This project measures that drop and investigates what causes it.

We train one standard model — a plain **U-Net trained from scratch** — separately on three rooftop/building datasets (WHU, Inria, Massachusetts) and test every trained model on all three datasets. Because the model, the training settings and the evaluation are kept identical, any difference in scores can be attributed to the datasets themselves, not to the method.

**Research question:** *To what extent can differences in image resolution, appearance, and annotation boundaries help explain cross-dataset performance degradation in rooftop segmentation, and what limitations remain in attributing the degradation to individual factors?*

## Objectives

1. **Measure** the cross-dataset performance drop with a fixed model and a fixed protocol (the 3 × 3 train/test transfer matrix), and investigate how sensitive the scores are to three dataset properties: image resolution, appearance, and annotation boundaries.
2. **(Ambitious)** Attempt to separate and quantify the contribution of each individual factor, while stating clearly what cannot be attributed from the available evidence.

## Datasets

All datasets were prepared in Milestone 2 as 512 × 512 PNG tiles (23,372 tiles in total). Splits were made at the source-image level before tiling, so no source image contributes tiles to two different splits. Labels were standardised to building = 1, background = 0 **without redrawing any label**, and image resolution was deliberately **not** normalised, because resolution is one of the factors under study.

| Dataset | Tiles (train / val / test) | Resolution | Source |
|---|---|---|---|
| WHU (aerial, Christchurch) | 8,188 (4,736 / 1,036 / 2,416) | 0.3 m | https://gpcv.whu.edu.cn/data/building_dataset.html (CC BY 4.0) |
| Inria Aerial Image Labeling | 14,580 (10,530 / 2,025 / 2,025) | 0.3 m | https://project.inria.fr/aerialimagelabeling/ |
| Massachusetts Buildings | 604 (548 / 16 / 40) | 1 m | https://www.cs.toronto.edu/~vmnih/data |

The AIRS dataset is held as a deferred, test-only set and has not been used so far. The datasets themselves are **not** stored in this repository (several GB); they are kept on the team server. Only the preparation code, the tile metadata and the results are maintained here.

## Method

- **Model:** standard U-Net (base width 64; encoder channels 64–128–256–512–1024; two 3 × 3 convolutions with BatchNorm and ReLU per block; 2 × 2 transposed-convolution upsampling with skip connections), **31,037,633 parameters**, trained from scratch. No pre-trained weights in the baseline.
- **Loss:** BCE + Dice. Prediction threshold fixed at 0.5.
- **Frozen protocol (Protocol A):** AdamW (learning rate 1 × 10⁻³, weight decay 1 × 10⁻⁴), cosine schedule, batch size 8, full 512 × 512 tiles, maximum 50 epochs, early stopping with patience 10 on validation IoU, augmentation = flips and 90° rotations (training split only), seed 42. The settings were fixed before the first run and are identical for every run — only the training dataset changes.
- **Metrics:** IoU (primary), precision, recall, F1, accumulated over the full test split. Pixel accuracy is not reported (buildings cover only ~11–19% of pixels, so it would be misleading).
- **Relative Transfer Drop (RTD):** RTD = (in-domain IoU − cross-dataset IoU) / in-domain IoU × 100% — the percentage of its own score that a model loses on another dataset.
- **Controls (Milestone 4):** seed repeats (42/43/44) for stability; an **equal-budget control (Protocol B)** in which WHU and Inria are stopped at exactly 3,450 optimiser steps (the number the Massachusetts baseline completed), separating the effect of training amount; and a **subset-size control** (WHU and Inria trained on random 548-tile subsets, three draws each), separating the effect of dataset size.
- **Supplementary comparisons (kept separate from the main matrix):** a U-Net with an ImageNet pre-trained ResNet-34 encoder (segmentation-models-pytorch), a focal-loss variant (focal γ = 2.0 + Dice), and inference-latency measurements.
- **Validity rules:** identical settings for every run; model selection on the validation split only, never on test; test splits evaluated once, after training; splits fixed at source-image level.

## Repository structure

```
src/
  data/         dataset loading and data checks (dataset.py, check_data.py)
  models/       U-Net model definition (model.py)
  evaluation/   training, evaluation, metrics and matrix building
                (train.py, evaluate.py, metrics.py, make_matrix.py)
tests/          automated tests (pytest, 12 tests)
evidence/       smoke-test evidence (logs, configs, matrices) proving the pipeline runs end to end
training/       Milestone 4 training record:
  results_tracker.csv   every run, its settings and its test scores
  summaries/            one summary file per run + the subset-control summary
  matrices/             baseline and equal-budget IoU / RTD matrices
  logs/                 per-run training logs
  plots/                training curves and result figures (with PLOTS.md index)
  optional/             ResNet-34 variant, focal-loss comparison and latency results
docs/           milestone reports (Milestone 1–3 PDFs; Milestone 4 report to be added)
notebooks/      EDA.ipynb and Preprocessing.ipynb from Milestone 2
```

Trained checkpoints (`best.pt`, ~119 MB each) exceed GitHub file limits and are kept on the team server only; every run's exact configuration (`config.json`) and per-epoch record (`log.csv`) are preserved with the run outputs on the server, and all reported numbers are reproduced in `training/`.

## Setup

```bash
# 1. Install a PyTorch build matching your CUDA version first, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Install the remaining dependencies
pip install -r requirements.txt
# (segmentation-models-pytorch is needed only for the optional ResNet-34 variant)
```

Expected data layout (read-only; the code never writes into it):

```
<data_root>/WHU/{train,validation,test}/{images,masks}/*.png
<data_root>/Inria/{train,validation,test}/{images,masks}/*.png
<data_root>/Mass/{train,validation,test}/{images,masks}/*.png
```

## How to run

**Smoke test** — proves the full pipeline (data checks → training → evaluation → matrix) end to end:

```bash
bash smoke_test.sh                 # uses built-in toy data
bash smoke_test.sh /path/to/data_root   # or check the real prepared data
```

Expected result: 15 acceptance checks — 13 pass, 1 warning (metadata cross-check), 1 skipped (optional variant library); `pytest` 12 passed. Evidence from our run is in `evidence/smoke_test/`.

**One baseline training run** (Milestone 4 protocol):

```bash
python src/evaluation/train.py \
  --data_root /path/to/data_root --train_set whu \
  --out runs/baseline/unet_whu_seed42 \
  --arch unet --seed 42 --epochs 50 --batch_size 8 \
  --lr 1e-3 --optimizer adamw --weight_decay 1e-4 \
  --scheduler cosine --patience 10 --loss bce_dice
```

Useful additional flags: `--max_steps N` (equal-budget control), `--subset_size 548 --subset_seed 1` (subset control), `--loss focal_dice` (focal comparison), `--arch smp_unet --pretrained` (ResNet-34 variant), `--resume` (continue from `last.pt`), `--dry_run` (tiny settings, full flow).

Each run writes `config.json` (all settings and library versions), `log.csv` (per-epoch record), `best.pt` / `last.pt` (checkpoints) and `results.json` (test metrics on all three datasets). The transfer matrices are then built with `src/evaluation/make_matrix.py`.

## Key results (Milestone 4)

Test IoU of the baseline models (rows: trained on; columns: tested on):

| Train \ Test | WHU | Inria | Massachusetts |
|---|---|---|---|
| WHU | 0.8624 | 0.3677 | 0.3959 |
| Inria | 0.6968 | 0.7169 | 0.3052 |
| Massachusetts | 0.5687 | 0.1995 | 0.6665 |

Relative Transfer Drop: four of the six cross-dataset pairs lose **54–70%** of their in-domain score. The two exceptions are both tested on WHU (Inria → WHU 2.79%, Massachusetts → WHU 14.67%), which suggests WHU is the easiest of the three test sets; the reason has not been tested yet and is examined in Milestone 5.

What the controls showed:

- **Training amount is not the explanation.** At an equal budget of 3,450 steps, in-domain IoU fell to 0.8015 (WHU) and 0.6314 (Inria), but the cross-dataset drop remained large (e.g. WHU → Inria 0.2453, a 69.4% drop).
- **Dataset size explains only part of it.** With 548-tile subsets, in-domain IoU fell by ≈ 7.6 points (WHU) and ≈ 13.3 points (Inria); even at equal size *and* equal steps, cross-dataset scores stayed far below in-domain scores.
- **Neither pre-training nor the loss changes the picture.** The ResNet-34 variant stayed within ~1.5 IoU points of the baseline in-domain and showed the same large transfer drop; the (corrected) focal loss was tied with the baseline on WHU and Massachusetts and clearly lower on Inria. BCE + Dice therefore remains the final loss.
- **Speed:** the selected checkpoints need ≈ 7.9 ms per 512 × 512 tile (≈ 126 tiles/second, batch size 1).

Full numbers: `training/matrices/` (baseline and equal-budget matrices), `training/summaries/subset_control_summary.txt`, `training/optional/` (variant matrix, focal matrix, latency).

## Progress

| Milestone | Content | Status |
|---|---|---|
| M1 (22 Oct 2026) | Problem formulation and project plan | Complete |
| M2 (29 Oct 2026) | Dataset preparation (23,372 tiles, verified) | Complete |
| M3 (5 Nov 2026) | Model and experiment design; pipeline implemented and smoke-tested | Complete |
| M4 (19 Nov 2026) | Model training: 17 main runs + supplementary comparisons | Complete |
| M5 (26 Nov 2026) | Factor-sensitivity experiments (resolution, appearance, annotation boundaries) and error analysis on the frozen baseline checkpoints — no retraining | In progress |
| M6 (10 Dec 2026) | Final synthesis and reporting | Planned |

## Team

Group 10: Bhawani Dutt, Jaival, Prakhar Mishra, Atharva Sarbhukan, Abhishek Rawat.

## Licence

Code in this repository is released under the MIT Licence (see `LICENSE`). The datasets remain under their own terms: WHU is CC BY 4.0; Inria and Massachusetts state no named licence and are used for research with citation of their sources.
