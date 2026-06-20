#!/usr/bin/env bash
# Inference with the SEMI-WEAK multi-task model -> out/submission_semiweak.csv (Kaggle-ready).
# Aligns to the full sample_submission id list by default (Kaggle requires all 142,818 rows).
#
# Usage:
#   bash scripts/semiweak/infer.sh                                  # newest semiweak best.pt
#   CKPT=out/semiweak/<run>/best.pt bash scripts/semiweak/infer.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
PY="${PY:-python}"

CKPT="${CKPT:-$(ls -t "$REPO"/out/semiweak/*/best.pt 2>/dev/null | head -1)}"
[ -z "$CKPT" ] && { echo "Không tìm thấy checkpoint semiweak. Train trước hoặc set CKPT=..."; exit 1; }
TEST_DIR="${TEST_DIR:-public_test}"
OUT="${OUT:-$REPO/out/submission_semiweak.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"; GPU="${GPU:-0}"
ALIGN="${ALIGN:-1}"

ARGS=(--ckpt "$CKPT" --test-dir "$TEST_DIR" --out "$OUT" --batch-size "$BATCH_SIZE")
[ "$ALIGN" = "1" ] && ARGS+=(--align-submission "$REPO/DATA/sample_submission.csv")

echo "==> ckpt=$CKPT"
echo "==> test_dir=DATA/$TEST_DIR  out=$OUT  align=$ALIGN"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/infer_semiweak.py "${ARGS[@]}"
echo "==> nộp: $PY -m kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai -f $OUT -m 'semiweak edgenext'"
