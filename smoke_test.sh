#!/usr/bin/env bash
# Exemplary run proving the pipeline works end to end (acceptance checks 1-4).
set -euo pipefail
cd "$(dirname "$0")"

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

SUMMARY=()

echo ""
echo "== [check 1] check_data.py =="
if python src/data/check_data.py --data_root "$DATA_ROOT_ARG" $TOY_FLAG; then
  SUMMARY+=("1 data checks: PASS")
else
  SUMMARY+=("1 data checks: FAIL"); FAILED=1
fi

echo ""
echo "== [check 2] pytest tests -q =="
if python -m pytest tests -q; then
  SUMMARY+=("2 pytest: PASS")
else
  SUMMARY+=("2 pytest: FAIL"); FAILED=1
fi

echo ""
echo "== [check 3] train.py --dry_run (whu, inria, mass) =="
for ds in whu inria mass; do
  echo "--- dry_run: $ds ---"
  python src/evaluation/train.py --data_root "$DATA_ROOT_ARG" --train_set "$ds" \
    --out "runs_smoke/unet_$ds" --arch unet --dry_run --seed 0
done
SUMMARY+=("3 dry_run x3 + test-on-all-3: PASS")

echo ""
echo "== [check 4] evaluate.py twice on one best.pt -> identical JSON =="
python src/evaluation/evaluate.py --checkpoint runs_smoke/unet_whu/best.pt --data_root "$DATA_ROOT_ARG" \
  --split test --datasets whu --out runs_smoke/eval1.json
python src/evaluation/evaluate.py --checkpoint runs_smoke/unet_whu/best.pt --data_root "$DATA_ROOT_ARG" \
  --split test --datasets whu --out runs_smoke/eval2.json
if diff -q runs_smoke/eval1.json runs_smoke/eval2.json >/dev/null; then
  SUMMARY+=("4 evaluate twice identical: PASS")
else
  echo "JSON files differ:"; diff runs_smoke/eval1.json runs_smoke/eval2.json || true
  SUMMARY+=("4 evaluate twice identical: FAIL"); FAILED=1
fi

echo ""
echo "== [extra] make_matrix.py on dry-run outputs =="
python src/evaluation/make_matrix.py --runs_dir runs_smoke --arch unet

echo ""
echo "== [extra] metrics sanity: all-background prediction -> IoU 0 =="
python - <<'EOF'
import torch, sys
sys.path.insert(0, ".")
from src.evaluation.metrics import ConfusionAccumulator
acc = ConfusionAccumulator()
target = torch.zeros(1, 1, 4, 4); target[0, 0, :2, :2] = 1.0
logits = torch.full((1, 1, 4, 4), -8.0)   # predicts background everywhere
acc.update(logits, target)
m = acc.compute()
assert m["iou"] == 0.0, m
print(f"all-background prediction on non-empty target -> IoU {m['iou']:.1f} (as expected)")
EOF
SUMMARY+=("metrics sanity (IoU 0): PASS")

echo ""
echo "== [extra] resume test: 1 epoch then resume to 2 =="
python src/evaluation/train.py --data_root "$DATA_ROOT_ARG" --train_set whu --out runs_smoke/resume_test \
  --arch unet --seed 0 --epochs 1 --batch_size 2 --num_workers 0
python src/evaluation/train.py --data_root "$DATA_ROOT_ARG" --train_set whu --out runs_smoke/resume_test \
  --arch unet --seed 0 --epochs 2 --batch_size 2 --num_workers 0 \
  --resume runs_smoke/resume_test/last.pt
EPOCH=$(tail -n 1 runs_smoke/resume_test/log.csv | cut -d, -f1)
if [ "$EPOCH" = "2" ]; then
  SUMMARY+=("resume continues epoch counter (-> epoch $EPOCH): PASS")
else
  SUMMARY+=("resume epoch counter: FAIL (last epoch $EPOCH)"); FAILED=1
fi

echo ""
echo "== [extra] smp_unet variant (if installed, NO --pretrained) =="
if python -c "import segmentation_models_pytorch" 2>/dev/null; then
  python src/evaluation/train.py --data_root "$DATA_ROOT_ARG" --train_set whu --out runs_smoke/smp_whu \
    --arch smp_unet --dry_run --seed 0
  SUMMARY+=("smp_unet dry-run: PASS")
else
  SUMMARY+=("smp_unet dry-run: SKIPPED (segmentation_models_pytorch not installed)")
fi

echo ""
echo "================ SMOKE TEST SUMMARY ================"
for s in "${SUMMARY[@]}"; do echo "  $s"; done
echo "===================================================="
if [ -n "${FAILED:-}" ]; then
  echo "FAILED (one or more checks above)"
  exit 1
fi
echo "ALL PASSED"
