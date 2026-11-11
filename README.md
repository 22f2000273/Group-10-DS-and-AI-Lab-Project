# unet_rooftop — baseline pipeline for cross-dataset rooftop segmentation

Fixed plain U-Net (from scratch) trained on **one** dataset at a time
(WHU / Inria / Massachusetts), tested on **all three** — so the measured
drop comes from the data, not the model.

## Purpose
Study of cross-dataset performance drop in binary rooftop segmentation
(DSAI Lab project, Group 10). One identical baseline model per training
dataset; RTD = relative task drop between in-domain and cross-domain IoU.

## Install
```bash
# install the PyTorch build for your CUDA first, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt   # segmentation-models-pytorch optional
```

## Expected data layout (READ-ONLY — the code never writes into it)
```
<data_root>/WHU/{train,validation,test}/{images,masks}/*.png
<data_root>/Inria/{train,validation,test}/{images,masks}/*.png
<data_root>/Mass/{train,validation,test}/{images,masks}/*.png
```
CLI keys: `whu` -> WHU, `inria` -> Inria, `mass` -> Mass.
Images: 512×512 RGB PNG. Masks: same-named single-channel PNG, values {0,1}.
Expected tiles (train/val/test): WHU 4736/1036/2416, Inria 10530/2025/2025,
Mass 548/16/40 — grand total 23,372. The folder scan is the source of truth;
`metadata.csv` (if present) is only cross-checked in `check_data.py`.

## Quick start (proves the pipeline end to end)
```bash
bash smoke_test.sh            # creates toy data automatically, runs all checks
bash smoke_test.sh /path/to/data_root   # or check real data
```

## Real baseline runs
```bash
bash run_all.sh               # trains whu, inria, mass then builds the RTD matrix
```
The hyper-parameters at the top of `run_all.sh` are **PLACEHOLDERS fixed in
Milestone 4** — do not treat them as tuned values.

Single run:
```bash
python train.py --data_root /path/to/data_root --train_set whu \
  --out runs/baseline_unet_whu --arch unet --seed 0 --epochs 50 --batch_size 8
```

## Outputs (in --out)
| file | content |
|------|---------|
| `config.json` | all args + torch version, device, param count, timestamp |
| `log.csv` | epoch, train_loss, val_loss, val_iou, val_f1, lr, seconds |
| `best.pt` | best validation-IoU checkpoint |
| `last.pt` | every-epoch checkpoint (model, optimiser, scheduler, scaler, RNG states) |
| `results.json` | train_set, arch, seed, param_count, best epoch/val IoU, TEST metrics on all three datasets |

## CLI options (train.py)
| flag | default | note |
|------|---------|------|
| `--data_root` | required | prepared dataset root |
| `--train_set` | required | whu / inria / mass |
| `--out` | required | output dir |
| `--arch` | unet | unet / smp_unet |
| `--pretrained` | off | smp only: ImageNet weights (only allowed download) |
| `--epochs` | 2 | smoke placeholder — fixed in Milestone 4 |
| `--batch_size` | 8 | |
| `--lr` | 1e-3 | |
| `--optimizer` | adam | adam / adamw / sgd |
| `--weight_decay` | 0.0 | |
| `--scheduler` | none | none / cosine / step |
| `--patience` | 5 | early stopping on val IoU |
| `--crop` | none | train-time random crop (multiple of 16) |
| `--amp` | off | mixed precision on CUDA |
| `--num_workers` | 2 | |
| `--seed` | 0 | |
| `--resume` | — | continue from last.pt |
| `--dry_run` | off | tiny settings, FULL flow incl. tests on all three |

## The 4 cross-domain validity rules
1. **Identical settings** for every run (same arch, loss, schedule, seed policy) —
   differences must come from the data.
2. **Model selection only on the source validation split** (never on test).
3. **Test splits evaluated once**, after training, by the single shared
   `evaluate_model` path.
4. **Splits fixed at source-image level**; tiles from one source image never
   cross splits.

## Reproducibility
Seeds set for Python/NumPy/torch(+CUDA)/DataLoader workers; `cudnn.benchmark=True`
and optional AMP are enabled for speed, so runs are **near-identical, not
bit-identical**. `evaluate.py` is deterministic (benchmark OFF, eval mode).

## OOM fix
`--batch_size 4` and/or `--crop 256` (validation/test always run at full 512).

## Not implemented yet
Boundary-tolerant IoU, factor experiments, object-level F1, area error —
later milestones. Metrics here are exactly IoU/precision/recall/F1 @ 0.5.
