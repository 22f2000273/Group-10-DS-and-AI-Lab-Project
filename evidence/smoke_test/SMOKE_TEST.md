# Smoke Test Results

## Exact command run
```
cd /home/bcilab/DSAI/repo && bash smoke_test.sh /home/bcilab/DSAI/data_view
```
Full terminal output captured at: `evidence/smoke_test/smoke_test.log` (3,863 bytes, copied from `/tmp/smoke_test_full.log`)

## Date and time
Thu Nov 12 07:16:42 PM IST 2026

## Git commit hash
`c3c9a5863c01cd1dae642f885b563bcd8b99662e` (message: "Add baseline U-Net rooftop segmentation pipeline")

## Python and PyTorch versions
- Python: 3.14.4
- PyTorch: 2.15.0+cu132

## Device used
GPU: NVIDIA L40S (CUDA available: True)

## Settings forced by --dry_run flag
From `train.py` line 163-164:
```
[dry_run] forcing tiny settings: max 8 tiles/split, 2 epochs, batch 2, crop 256
```
Specifically: `--epochs 2`, `--batch_size 2`, `--crop 256`, `--max_tiles 8` per split.

## Files produced by the run (in runs_smoke/)
| File | Contents |
|------|----------|
| runs_smoke/unet_whu/config.json | run configuration (args, torch version, device, param count, timestamp) |
| runs_smoke/unet_whu/log.csv | per-epoch train/val loss, val IoU, val F1, lr, seconds |
| runs_smoke/unet_whu/best.pt | best validation-IoU checkpoint (model only) |
| runs_smoke/unet_whu/last.pt | final checkpoint (model, optimizer, scheduler, scaler, RNG states) |
| runs_smoke/unet_whu/results.json | test metrics on whu/inria/mass |
| runs_smoke/unet_inria/config.json | run configuration |
| runs_smoke/unet_inria/log.csv | per-epoch log |
| runs_smoke/unet_inria/best.pt | best checkpoint |
| runs_smoke/unet_inria/last.pt | final checkpoint |
| runs_smoke/unet_inria/results.json | test metrics on whu/inria/mass |
| runs_smoke/unet_mass/config.json | run configuration |
| runs_smoke/unet_mass/log.csv | per-epoch log |
| runs_smoke/unet_mass/best.pt | best checkpoint |
| runs_smoke/unet_mass/last.pt | final checkpoint |
| runs_smoke/unet_mass/results.json | test metrics on whu/inria/mass |
| runs_smoke/eval1.json | first evaluate.py run (deterministic check) |
| runs_smoke/eval2.json | second evaluate.py run (deterministic check - identical to eval1.json) |
| runs_smoke/matrix_iou.csv | cross-dataset IoU matrix |
| runs_smoke/matrix_rtd.csv | cross-dataset RTD % matrix |
| runs_smoke/matrix.md | markdown version of both matrices |
| runs_smoke/resume_test/config.json | resume test config |
| runs_smoke/resume_test/log.csv | resume test epoch log |
| runs_smoke/resume_test/best.pt | resume test best checkpoint |
| runs_smoke/resume_test/last.pt | resume test final checkpoint |

Copied to `evidence/smoke_test/` (excluding `.pt` checkpoint files):
config.json x3, log.csv x3, results.json x3, matrix_iou.csv, matrix_rtd.csv, matrix.md, eval1.json, eval2.json

## Checkpoint file sizes and load verification
| Checkpoint | Size | Loads OK? |
|------------|------|-----------|
| runs_smoke/unet_whu/best.pt | 119M | YES |
| runs_smoke/unet_whu/last.pt | 356M | YES |
| runs_smoke/unet_inria/best.pt | 119M | YES |
| runs_smoke/unet_inria/last.pt | 356M | YES |
| runs_smoke/unet_mass/best.pt | 119M | YES |
| runs_smoke/unet_mass/last.pt | 356M | YES |
| runs_smoke/resume_test/best.pt | 119M | YES |
| runs_smoke/resume_test/last.pt | 356M | YES |

All 8 checkpoints load successfully with expected keys (`epoch`, `model`, `val_iou` for best; plus `optimizer`, `scheduler`, `scaler`, `patience_counter`, `rng`, `seed` for last).

## Data used
**Real tiles from `/home/bcilab/DSAI/data_view`** (symlinks to `/home/bcilab/DSAI/prepared_dataset/`), NOT auto-generated toy data.

Lines from `smoke_test.sh` that decide this (lines 6-16):
```bash
DATA_ROOT_ARG="${1:-}"
TOY_FLAG=""
if [ -z "$DATA_ROOT_ARG" ]; then
  TOY_DIR="$(mktemp -d)/toy"
  echo "== creating toy data at $TOY_DIR =="
  python src/data/make_toy_data.py --out "$TOY_DIR"
  DATA_ROOT_ARG="$TOY_DIR"
  TOY_FLAG="--toy"
else
  echo "== using provided DATA_ROOT=$DATA_ROOT_ARG =="
fi
```
Since `/home/bcilab/DSAI/data_view` was passed as an argument, `TOY_FLAG` is empty and real data is used.

## Acceptance checks

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | All 9 dataset/split folders exist with images/ and masks/ | PASS | check_data.py output: `[PASS] 1. all 9 dataset/split folders exist (images/ + masks/)` |
| 2 | Tile counts equal expected (WHU 8188, Inria 14580, Mass 604 = 23372) | PASS | check_data.py: `[PASS] 2. tile counts match expected... total=23372` |
| 3 | Every image has same-named mask; sizes 512x512; images 3-channel; masks only {0,1} | PASS | check_data.py: `[PASS] 3. pairing+size+channels+mask values on 1456 tiles` |
| 4 | metadata.csv row count == total tiles | NOT IMPLEMENTED | metadata.csv not present at data_view; check_data.py prints `[WARN] 4. metadata.csv not present` — folder scan is source of truth |
| 5 | Forward pass output shape (B,1,512,512) | PASS | check_data.py: `[PASS] 5. UNet forward pass shape (2,1,512,512) — got (2, 1, 512, 512)` |
| 6 | Parameter count == 31,037,633 | PASS | check_data.py: `[PASS] UNet param count == 31,037,633 — counted 31,037,633` |
| 7 | All image-mask pairs paired, no orphans | PASS | check_data.py: `[PASS] 3. pairing... on 1456 tiles` |
| 8 | Loss is finite | PASS | dry_run train_loss values are finite (e.g., 1.4619, 1.3849) |
| 9 | Checkpoint saved and re-loadable | PASS | best.pt and last.pt saved for all 3 runs; all 8 checkpoints verified loadable |
| 10 | results.json written for all 3 runs | PASS | results.json exists for unet_whu, unet_inria, unet_mass |
| 11 | evaluate.py produces identical JSON on repeat runs | PASS | eval1.json and eval2.json are byte-identical (diff produced no output) |
| 12 | Matrix has all 9 entries | PASS | matrix.md shows all 9 train→test cells populated |
| 13 | Pytest suite passes | PASS | 12 passed in 2.02s |
| 14 | Resume continues epoch counter | PASS | resume test: "resumed from ... next epoch 2", then "epoch 2/2" trained |
| 15 | smp_unet variant | SKIPPED | segmentation_models_pytorch not installed |

No check is "NOT IMPLEMENTED" except metadata.csv cross-check (which is explicitly warn-only per spec: "folder scan is the source of truth").

## Data source quote
From smoke_test.sh lines 6-16 — since `/home/bcilab/DSAI/data_view` was passed as `$1`, the script uses real data:
```
== using provided DATA_ROOT=/home/bcilab/DSAI/data_view ==
```

## --max_steps flag
**Exists** in `src/evaluation/train.py` (added after the first smoke run). Training stops after exactly N optimiser steps, counted across epochs. When set, the scheduler steps per optimiser step (not per epoch) so the schedule covers the step budget. The number of steps completed is recorded as `max_steps` and `steps_completed` in `config.json`, and as a `step` column in `log.csv`.

Tested with `--dry_run --max_steps 5` on WHU: training stopped at exactly 5 optimiser steps (4 in epoch 1, 1 in epoch 2, stopped mid-epoch) — output:
```
max_steps 5 reached after 5 optimiser steps (stopped mid-training at epoch 2)
```
Evidence for this test is in `evidence/smoke_test/max_steps_test/` (config.json and log.csv copied from runs_smoke/max_steps_test/).

## Learning-rate scheduler
Per-epoch by default in `train.py` `build_scheduler()` (`CosineAnnealingLR` / `StepLR` stepped once per epoch). Per-optimiser-step only when `--max_steps` is set — then `T_max` / `step_size` are sized to the step budget.

## Factor-sensitivity code
**NOT IMPLEMENTED.** No code exists for:
- Resolution resampling
- Mask shift / erode / dilate
- Brightness / contrast / saturation transforms at evaluation time

Searched `src/` for: resample, erode, dilate, brightness, contrast, saturation, shift — no matches.

## Ready to commit
All staged/unstaged changes are ready. `evidence/smoke_test/` contains the run artifacts. Each `runs_smoke/unet_*` folder holds best.pt (119M) + last.pt (356M) — 474M is the folder total. Checkpoints and run folders are gitignored (`runs_smoke/`, `*.pt`); only `evidence/` and the code changes are committed.
