# M4 Training Plan — flags, proposed protocol, timing

## Phase 0 — setup verification (DONE)
- Latest commit after `git pull`: `3c3d947` ("Add --max_steps flag and smoke-test evidence"), branch `main` synced with origin.
- `pytest tests -q`: **12 passed**.
- Data checks (`src/data/check_data.py --data_root /home/bcilab/DSAI/data_view`): ALL PASSED.
  Tile counts: WHU 8,188 / Inria 14,580 / Mass 604 → total 23,372.
- Disk: 3.1 TB free (≫ 20 GB needed; each run ~475 MB checkpoints).
- GPU: NVIDIA L40S, 45,092 MiB free of 46,068 MiB.

## Flag list — `python src/evaluation/train.py --help` (exact)
```
--data_root DATA_ROOT          (required)
--train_set {whu,inria,mass}   (required)
--out OUT                      (required)
--arch {unet,smp_unet}         (default: unet)
--pretrained                   (smp only: ImageNet encoder weights)
--epochs EPOCHS                (default: 2, placeholder)
--batch_size BATCH_SIZE        (default: 8)
--lr LR                        (default: 1e-3)
--optimizer {adam,adamw,sgd}   (default: adam)
--weight_decay WEIGHT_DECAY    (default: 0.0)
--scheduler {none,cosine,step} (default: none)
--patience PATIENCE            (default: 5)
--crop CROP                    (train-time random crop, multiple of 16)
--amp                          (mixed precision on cuda)
--num_workers NUM_WORKERS      (default: 2)
--seed SEED                    (default: 0)
--resume RESUME                (path to last.pt)
--dry_run                      (8 tiles/split, 2 epochs, batch 2, crop 256)
--max_steps MAX_STEPS          (stop after exactly N optimiser steps, across
                                epochs; scheduler steps per step when set)
```

## PROPOSED baseline protocol (Protocol A, seed 42) — NOT yet approved
| Setting | Value | Fits the flags? |
|---|---|---|
| Model | unet, trained from scratch | yes (`--arch unet`, no `--pretrained`) |
| Loss | BCE + Dice (BCEDice) | yes (hardcoded in train.py) |
| Optimiser | AdamW | yes (`--optimizer adamw`) |
| Learning rate | 1e-3 | yes (`--lr 1e-3`) |
| Weight decay | 1e-4 | yes (`--weight_decay 1e-4`) |
| Scheduler | cosine, stepped per epoch | yes (`--scheduler cosine`, no `--max_steps`) |
| Batch size | 8 | yes (`--batch_size 8`) |
| Input | full 512x512 tiles (no crop) | yes (omit `--crop`) |
| Max epochs | 50 | yes (`--epochs 50`) |
| Early stopping | patience 10 on val IoU | yes (`--patience 10`) |
| Augmentation | h-flip, v-flip, 90° rotations, train split only | yes (built into dataset.py, train only) |
| Seed | 42 | yes (`--seed 42`) |
| Selection | best val IoU checkpoint; threshold 0.5 | yes (built-in) |

Exact command per baseline run (example, WHU):
```
python src/evaluation/train.py --data_root /home/bcilab/DSAI/data_view \
  --train_set whu --out runs/baseline/unet_whu_seed42 --arch unet \
  --optimizer adamw --lr 1e-3 --weight_decay 1e-4 --scheduler cosine \
  --batch_size 8 --epochs 50 --patience 10 --seed 42
```

## Flagged problems (not silently changed)
1. **No `--max_tiles` flag.** The timing probe in the plan asked for
   `--max_tiles 500`; train.py has no such flag (only `--dry_run` limits tiles
   internally). Per your decision, the probe uses `--max_steps 63`
   (63 steps x batch 8 = 504 tiles) instead. No code change.
2. **No `--subset_size` / `--subset_seed` flags.** Needed for Phase 5; they
   will be ADDED to train.py when Phase 5 starts (approved by you), tested
   with `--dry_run` first.

## Phase 4 — equal budget
- B = **3,450 optimiser steps** (from runs/baseline/unet_mass_seed42/config.json,
  `steps_completed` = 3450 = 50 epochs x 69 steps/epoch).
- WHU and Inria retrained with `--max_steps 3450` (all other settings identical
  to Protocol A); the Massachusetts baseline IS the equal-budget Mass run.

## Timing probe (Phase 1, step 6) — measured
- Run: WHU, 1 epoch, `--max_steps 63` (= 504 tiles), batch 8, full 512px,
  AdamW 1e-3 / wd 1e-4, cosine, seed 42. Log: `training/logs/probe_timing.log`.
- Epoch wall time (63 train steps + full WHU val pass): **37.7 s**
- Peak GPU memory: **22,693 MiB** (~22.7 of 46 GB — batch 8 @ 512px fits)
- evaluate.py on full WHU test set (2,416 tiles): **62.6 s** → inference
  ≈ 4.83 batches/s (38.6 tiles/s)

### Derived estimates (arithmetic)
- WHU val pass = 1,036 tiles = 130 batches = 130 / 4.83 ≈ **26.9 s**
- Train-step time = (37.7 − 26.9) / 63 ≈ **0.17 s/step** (incl. warmup)
- Steps/epoch at batch 8: WHU 4736/8 = 592 · Inria 10530/8 = 1,317 · Mass 548/8 = 69
- Test eval per run (all 3 sets = 4,481 tiles) ≈ 62.6 + 52.6 + 1 ≈ **2 min**

| Run | Per-epoch (train+val) | Max wall (50 epochs) + test eval |
|---|---|---|
| WHU | 592×0.17 + 26.9 ≈ 128 s ≈ 2.1 min | ≈ 107 + 2 ≈ **1.8 h** |
| Inria | 1317×0.17 + 52.6 ≈ 277 s ≈ 4.6 min | ≈ 231 + 2 ≈ **3.9 h** |
| Mass | 69×0.17 + 0.4 ≈ 12 s ≈ 0.2 min | ≈ 10 + 2 ≈ **0.2 h** |

Early stopping (patience 10 on val IoU) can shorten these; the numbers above
are the 50-epoch maximum. All 17 planned runs ≈ up to **1–1.5 days** of
sequential GPU time in the worst case.

### Probe quality note
With only 63 optimiser steps the probe model already reached val IoU 0.6134 /
WHU test IoU 0.5947 — the pipeline learns correctly at real scale.

Split counts (folder scan): WHU 4,736/1,036/2,416 · Inria 10,530/2,025/2,025 ·
Mass 548/16/40 (train/val/test). Steps/epoch at batch 8: WHU 592,
Inria 1,317, Mass 69.
