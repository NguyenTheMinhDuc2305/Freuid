#!/usr/bin/env bash
# Fine-tune TruFor (forensic framework) cho DETECTION trên FREUID — nhãn ảnh, KHÔNG mask.
# Đông Noiseprint++; train encoder+detection head; LR thấp + early-stop. -> forensic member ensemble.
#
# Usage:
#   bash scripts/trufor/train.sh                 # encoder_head @512
#   bash scripts/trufor/train.sh --smoke         # 1-epoch error check
#   TRAINABLE=head bash scripts/trufor/train.sh  # chỉ detection head (nhẹ, có thể yếu)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
PY="${PY:-python}"

IMG_SIZE="${IMG_SIZE:-512}"; BATCH_SIZE="${BATCH_SIZE:-8}"; LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-15}"; TRAINABLE="${TRAINABLE:-encoder_head}"; NUM_WORKERS="${NUM_WORKERS:-8}"
GPU="${GPU:-0}"; TS="$(date +%Y%m%d_%H%M%S)"; mkdir -p "$REPO/logs"
RUN="trufor_${TRAINABLE}_${TS}"

echo "==> run=$RUN  img=$IMG_SIZE batch=$BATCH_SIZE lr=$LR trainable=$TRAINABLE gpu=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL "$PY" src/train_trufor.py \
    --run-name "$RUN" --img-size "$IMG_SIZE" --batch-size "$BATCH_SIZE" --lr "$LR" \
    --epochs "$EPOCHS" --trainable "$TRAINABLE" --num-workers "$NUM_WORKERS" "$@" \
    2>&1 | tee "$REPO/logs/${RUN}.log"
echo "==> best: out/trufor/${RUN}/best.pt"
echo "==> infer: CKPT=out/trufor/${RUN}/best.pt $PY src/infer_trufor.py --ckpt out/trufor/${RUN}/best.pt --out out/submission_trufor.csv"
