#!/usr/bin/env bash
# Inference + Test-Time Adaptation -> out/submission_tta.csv (xem docs/test_time_adaptation.md).
# Usage:
#   bash scripts/clip_linear/infer_tta.sh                      # MODE=ttaug (mặc định, an toàn)
#   MODE=tent TENT_STEPS=1 TENT_LR=1e-4 bash scripts/clip_linear/infer_tta.sh
#   MODE=ttaug+tent CKPT=out/clip_linear/<run>/best.pt bash scripts/clip_linear/infer_tta.sh
#
# ⚠️ Metric rank-based: chỉ TTAug & TENT có tác dụng (score-norm vô hiệu). Luôn kiểm Val-OOD
#    trước khi nộp TENT (rủi ro collapse).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
PY="${PY:-python}"

CKPT="${CKPT:-$(ls -t "$REPO"/out/clip_linear/*/best.pt 2>/dev/null | grep -v '/loto_' | head -1)}"
[ -z "$CKPT" ] && { echo "No checkpoint found. Train first or set CKPT=..."; exit 1; }
TEST_DIR="${TEST_DIR:-public_test}"
OUT="${OUT:-$REPO/out/submission_tta.csv}"
MODE="${MODE:-ttaug}"; TTA_SCALES="${TTA_SCALES:-0.85}"
TENT_STEPS="${TENT_STEPS:-1}"; TENT_LR="${TENT_LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"; GPU="${GPU:-0}"

EXTRA_ARGS=("$@")
[ "${ALIGN:-1}" = "1" ] && EXTRA_ARGS+=(--align-submission "$REPO/DATA/sample_submission.csv")

echo "==> ckpt=$CKPT"
echo "==> mode=$MODE  test_dir=DATA/$TEST_DIR  out=$OUT  align=${ALIGN:-1}"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/infer_tta.py \
    --ckpt "$CKPT" --test-dir "$TEST_DIR" --out "$OUT" \
    --mode "$MODE" --tta-scales "$TTA_SCALES" \
    --tent-steps "$TENT_STEPS" --tent-lr "$TENT_LR" \
    --batch-size "$BATCH_SIZE" "${EXTRA_ARGS[@]}"

echo "==> submit: $PY -m kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai -f $OUT -m '$MODE'"
