#!/usr/bin/env bash
# Semi-weakly-supervised JOINT training: one model trained on the COMBINED stream
#   FREUID (label only) + IDNet (label+mask) + FantasyID (label+mask).
# Mask loss applies only where a mask exists; a held-out FREUID val is scored with
# the competition metric (APCER@1%BPCER) each epoch.
#
# Usage:
#   bash scripts/semiweak/train.sh                      # auto: all IDNet + FantasyID
#   FREUID_VAL_TYPE="MAURITIUS/ID" bash scripts/semiweak/train.sh   # OOD val (unseen type)
#   IDNET_COUNTRIES="EST ESP GRC" bash scripts/semiweak/train.sh    # restrict IDNet
#   USE_FANTASY=0 bash scripts/semiweak/train.sh        # IDNet only
#
# Output: out/semiweak/<run>/best.pt  (chosen by best FREUID-val AUC)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
PY="${PY:-python}"
[ -f "$REPO/scripts/.wandb_env" ] && source "$REPO/scripts/.wandb_env"

# Model: CLIP ViT-L (LN-tuning, foundation) + EdgeDoc-style 2 heads (detect+mask).
# Override BACKBONE=edgenext_xx_small for the tiny EdgeNeXt variant.
# Defaults: λ=3, lr 3e-4, wd 5e-4, 20 epochs + early stopping.
BACKBONE="${BACKBONE:-vit_large_patch14_clip_224.openai}"
EPOCHS="${EPOCHS:-20}"; BATCH_SIZE="${BATCH_SIZE:-24}"; IMG_SIZE="${IMG_SIZE:-384}"
LR="${LR:-3e-4}"; MASK_WEIGHT="${MASK_WEIGHT:-3.0}"; PATIENCE="${PATIENCE:-5}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"   # chống overconfident (0=tắt)
UNFREEZE_BLOCKS="${UNFREEZE_BLOCKS:-4}"       # CLIP: mở băng 2 block cuối cho mask (0=chỉ LN)
NUM_WORKERS="${NUM_WORKERS:-8}"; IDNET_CAP="${IDNET_CAP:-2000}"
USE_IDNET="${USE_IDNET:-1}"; USE_FANTASY="${USE_FANTASY:-1}"
GPU="${GPU:-0}"; WANDB_MODE="${WANDB_MODE:-online}"
RUN_NAME="semiweak_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$REPO/logs"

# ensure the fixed leakage-free FREUID split exists (val = ONLY FREUID samples)
if [ ! -f "$REPO/DATA/freuid_val.csv" ]; then
    echo "==> tạo split FREUID cố định..."; "$PY" src/data_prep/make_freuid_split.py
fi

ARGS=(--run-name "$RUN_NAME" --backbone "$BACKBONE" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
      --img-size "$IMG_SIZE" --lr "$LR" --mask-weight "$MASK_WEIGHT" --patience "$PATIENCE"
      --label-smoothing "$LABEL_SMOOTHING" --unfreeze-blocks "$UNFREEZE_BLOCKS"
      --num-workers "$NUM_WORKERS" --idnet-cap-per-country "$IDNET_CAP"
      --use-idnet "$USE_IDNET" --use-fantasy "$USE_FANTASY" --wandb-mode "$WANDB_MODE")
[ -n "${FREUID_VAL_TYPE:-}" ] && ARGS+=(--freuid-val-type "$FREUID_VAL_TYPE")
[ -n "${IDNET_COUNTRIES:-}" ] && ARGS+=(--idnet-countries $IDNET_COUNTRIES)

echo "==> semi-weak: img=$IMG_SIZE bs=$BATCH_SIZE ep=$EPOCHS λ=$MASK_WEIGHT patience=$PATIENCE idnet_cap=$IDNET_CAP"
echo "==> FREUID val: ${FREUID_VAL_TYPE:-fixed leakage-free split (DATA/freuid_val.csv)}"
CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL "$PY" src/train_semiweak.py "${ARGS[@]}" \
    2>&1 | tee "$REPO/logs/${RUN_NAME}.log"
echo "==> done. weight: out/semiweak/${RUN_NAME}/best.pt"
