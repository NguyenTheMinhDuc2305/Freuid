#!/usr/bin/env bash
# Run inference on the test set -> out/submission.csv (ready for Kaggle).
# Usage:
#   bash scripts/clip_linear/infer.sh                                   # uses newest best.pt
#   CKPT=out/clip_linear/<run>/best.pt bash scripts/clip_linear/infer.sh
#   HARD_LABEL=1 bash scripts/clip_linear/infer.sh                      # write 0/1 instead of score
#
# Metric is AuDET / APCER@1%BPCER (threshold-swept) -> default output is the
# fraud PROBABILITY, which is what those metrics expect.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PY:-python}"

# newest full-data best.pt unless CKPT is given (exclude loto_* diagnostic folds)
CKPT="${CKPT:-$(ls -t "$REPO"/out/clip_linear/*/best.pt 2>/dev/null | grep -v '/loto_' | head -1)}"
[ -z "$CKPT" ] && { echo "No checkpoint found. Train first or set CKPT=..."; exit 1; }
TEST_DIR="${TEST_DIR:-public_test}"
OUT="${OUT:-$REPO/out/submission.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPU="${GPU:-0}"

EXTRA_ARGS=("$@")
[ "${HARD_LABEL:-0}" = "1" ] && EXTRA_ARGS+=(--hard-label)
# Kaggle requires ALL sample_submission ids (142,818) — align by default so the
# output is submittable. Set ALIGN=0 to emit only the predicted images.
if [ "${ALIGN:-1}" = "1" ]; then
    EXTRA_ARGS+=(--align-submission "$REPO/DATA/sample_submission.csv")
fi

echo "==> ckpt=$CKPT"
echo "==> test_dir=DATA/$TEST_DIR  out=$OUT  align=${ALIGN:-1}"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/infer.py \
    --ckpt "$CKPT" \
    --test-dir "$TEST_DIR" \
    --out "$OUT" \
    --batch-size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"

echo "==> submit with:"
echo "    $PY -m kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai -f $OUT -m 'clip-linear'"
