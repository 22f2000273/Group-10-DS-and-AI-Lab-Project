# Milestone 4 — Optional Experiments FINAL REPORT

## Phase 0 — Setup verification (no training)
- git pull: up to date; latest commit `3c3d947` ("Add --max_steps flag and smoke-test evidence")
- pytest: **12 passed** ✅
- `--loss` flag in train.py: did NOT exist; added `--loss {bce_dice,focal_dice}` (default `bce_dice`)
- segmentation-models-pytorch: NOT installed; installed **smp 0.6.0** via pip
- Disk: 3.1 TB free · GPU: NVIDIA L40S, 45 GB free

## Phase 1 — ResNet-34 pre-trained variant (VARIANT — comparison only)
- Dry-run test (`--arch smp_unet --pretrained --dry_run` on WHU): PASSED — full flow complete, **24,436,369 params** (vs baseline U-Net 31,037,633)
- Runs:

| run | best epoch | best val IoU | test whu | test inria | test mass | wall time |
|---|---|---|---|---|---|---|
| variant_mass_seed42 | 42/50 | 0.6620 | 0.5960 | 0.2169 | **0.6808** | ~6 min |
| variant_whu_seed42 | 49/50 | 0.8951 | **0.8593** | 0.4229 | 0.3610 | ~40 min |
| variant_inria_seed42 | 49/50 | 0.7400 | 0.6656 | **0.7224** | 0.3246 | ~40 min |

- Variant diagonal vs baseline diagonal: Mass 0.6808 vs 0.6665 (+), WHU 0.8593 vs 0.8624 (=), Inria 0.7224 vs 0.7169 (+)
- Matrices: `training/optional/resnet_variant_matrix.md` (+ heading: "OPTIONAL VARIANT (ResNet-34 pre-trained encoder) — comparison only, not part of the main baseline matrix.")

## Phase 2 — Alternative loss: focal + Dice
- `--loss {bce_dice,focal_dice}` added to train.py (default `bce_dice`, existing behavior unchanged); `FocalDice` class (gamma 2.0 + same Dice term); only `src/evaluation/train.py` changed; pytest **12 passed**; `--dry_run --loss focal_dice` on WHU completed with finite loss
- **Loss bug found and fixed:** the original focal runs used Dice's eps=1.0 inside the focal log terms — `log(p+1.0)` made the focal term negative and pulled predictions toward 0.31/0.69. Fix: log terms now use `p.clamp_min(1e-7)` / `(1-p).clamp_min(1e-7)`; `dice_loss` untouched (eps=1.0 correct there). Sanity check: perfect → focal 0.000000, wrong → 8.795, 200 random batches never negative. A NaN/inf guard was also added to the training loop. The original `FOCAL_*` runs are **INVALID** (kept on disk, marked in the tracker); the `FOCALFIX_*` re-runs below replace them.

| run | best epoch | best val IoU | test whu | test inria | test mass | wall time |
|---|---|---|---|---|---|---|
| FOCALFIX_mass_seed42 | 47/50 | 0.6550 | 0.5789 | 0.1857 | **0.6683** | ~18 min |
| FOCALFIX_whu_seed42 | 45/50 | 0.8664 | **0.8509** | 0.3262 | 0.3759 | ~2.6 h |
| FOCALFIX_inria_seed42 | 37/50 | 0.7003 | 0.7102 | **0.6859** | 0.3035 | ~5.2 h |

- Diagonal IoU: baseline (BCE+Dice) vs focal+Dice (fixed) — whu 0.8624 → **0.8509** (−0.0115), inria 0.7169 → **0.6859** (−0.0310), mass 0.6665 → **0.6683** (+0.0018)
- With the fix, focal+Dice is **competitive with the baseline** on all three diagonals; the old "loss collapse" conclusion was an artifact of the bug. BCE+Dice stays the default (focal never clearly wins).
- Cross-dataset scores with the fixed focal loss are close to the baseline's (Inria→WHU 0.7102 vs 0.6968; Inria→Mass 0.3035 vs 0.3052); the differences are within seed-to-seed variation, so no cross-domain advantage is claimed.
- Matrices: `training/optional/focal_loss_matrix.md` (includes diagonal comparison table, buggy numbers marked invalid) + csv.

## Phase 3 — Inference latency (measurement only; no training)
- GPU: NVIDIA L40S · batch size: 1 · tile size: 512×512 · 10 warm-up tiles
- Mean ms per tile over up to 100 timed test tiles (Mass timed 30 — its test split has 40 tiles):

| dataset | tiles timed | mean ms/tile | std ms/tile | tiles/sec |
|---|---|---|---|---|
| whu | 100 | 7.90 | 0.02 | 126.7 |
| inria | 100 | 7.91 | 0.01 | 126.5 |
| mass | 30 | 7.93 | 0.02 | 126.1 |

- Saved: `training/optional/latency.md` + `training/optional/latency.py`

## Incidents / failures
- One real incident: the **FocalDice loss bug** (Dice eps=1.0 leaked into the focal log terms) invalidated the first 3 focal runs; fixed, sanity-checked, and re-run as `FOCALFIX_*` (details in Phase 2). All 9 optional runs otherwise completed; Mass variant ~6 min, WHU/Inria variants ~40 min; focal runs ~18 min / 2.6 h / 5.2 h. Separately, the old Inria focal log printed NaN validation loss in several epochs; validation loss is computed with BCE+Dice in all runs and the NaN appeared only in that printed value — validation IoU, checkpoint selection and all test scores were unaffected.

## Files under training/optional/ ready to commit
- PROGRESS.md, optional_tracker.csv
- summaries/VARIANT_*.txt (3), summaries/FOCAL_*.txt (3, INVALID — kept), summaries/FOCALFIX_*.txt (3)
- logs/VARIANT_*.log (3), logs/FOCAL_*.log (3), logs/FOCALFIX_*.log (3)
- resnet_variant_matrix.md, resnet_variant_matrix_iou.csv, resnet_variant_matrix_rtd.csv
- focal_loss_matrix.md, focal_loss_matrix_iou.csv, focal_loss_matrix_rtd.csv
- latency.md, latency.py
- run folders remain in runs/optional/ (checkpoints *.pt — gitignored, stay on server per standing rules)
