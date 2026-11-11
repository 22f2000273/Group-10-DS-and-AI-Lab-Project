#!/usr/bin/env bash
# Real baseline runs — PLACEHOLDER values below are fixed in Milestone 4.
# Do NOT run this on real data until the final settings are agreed.
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/home/bcilab/DSAI/data_view}"   # PLACEHOLDER: fixed in Milestone 4
EPOCHS="${EPOCHS:-2}"        # PLACEHOLDER: fixed in Milestone 4
BATCH="${BATCH:-8}"          # PLACEHOLDER: fixed in Milestone 4
LR="${LR:-1e-3}"             # PLACEHOLDER: fixed in Milestone 4
SEED="${SEED:-0}"            # PLACEHOLDER: fixed in Milestone 4

for ds in whu inria mass; do
  echo "=== baseline_unet_${ds} ==="
  python train.py \
    --data_root "$DATA_ROOT" \
    --train_set "$ds" \
    --out "runs/baseline_unet_${ds}" \
    --arch unet \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --optimizer adam \
    --scheduler none \
    --num_workers 4 \
    --amp
done

python make_matrix.py --runs_dir runs --arch unet
