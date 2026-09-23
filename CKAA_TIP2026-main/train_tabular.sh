#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_ROOT="$(dirname "$ROOT")"

python "$ROOT/tools/prepare_tabular_dataset.py" \
  --raw-root "$RAW_ROOT" \
  --output-root "$ROOT/A_CLData/tabular_ckaa"

python "$ROOT/train_eval.py" \
  -d tabular \
  -t 5 \
  -m vit_base_patch16_224.augreg_in21k \
  -b 32 \
  -e 12 \
  -jt 0 \
  -je 0 \
  -et 1 \
  --eval_batch_size 64 \
  --temperature 28.0 \
  --lr 0.005 \
  --lr_scale 0.2 \
  --lr_scale_patterns patch_embed \
  --null_eta1 0.999 \
  --null_eta2 0.999 \
  -tc 2.0 \
  --eval-tool adapter \
  --eval-trained-task-router true \
  --eval-local-task-head true \
  --eval-task-weight false \
  --freeze-shared-prompts-after-first true \
  --prototype-confidence-weight 0.0 \
  --seed 2024 \
  --save-model true \
  --save-model-name model_tabular_5s_2cls_final \
  --logs-dir logs \
  -sf tabular_5s_2cls_e12_final \
  "$@"
