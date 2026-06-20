#!/usr/bin/env bash
# Train the classification-only CLIP-linear forgery detector.
# Usage:
#   cp scripts/.wandb_env.example scripts/.wandb_env   # once: paste key from wandb.ai/authorize
#   bash scripts/clip_linear/train.sh                  # full training (logs to your W&B account)
#   bash scripts/clip_linear/train.sh --smoke          # quick 1-epoch error check
#   LEAVE_OUT="EGYPT/DL" bash scripts/clip_linear/train.sh   # leave-one-type-out eval
#   WANDB_MODE=offline bash scripts/clip_linear/train.sh     # no upload (local only)
#
# Logging: Weights & Biases (view loss curves on the web at the printed run URL).
# Local copy: logs/<run>.log (full stdout+stderr); checkpoints in out/clip_linear/<run>/.
# Crashes are also surfaced in the W&B run (alert + summary) by src/train.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PY:-python}"

# W&B auth: load WANDB_API_KEY from an untracked secrets file if present, so the
# run authenticates automatically (no manual `wandb login`). Keep the key OUT of
# this tracked script — put it in scripts/.wandb_env (see scripts/.wandb_env.example).
[ -f "$REPO/scripts/.wandb_env" ] && source "$REPO/scripts/.wandb_env"

# ---- hyperparameters (override via env) ----
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-512}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEAVE_OUT="${LEAVE_OUT:-}"
GPU="${GPU:-0}"
AUG="${AUG:-default}"     # default | p1p3 (print-capture + resolution/quality aug)
# Backbone swappable: CLIP (mặc định) | DINOv2 (vit_large_patch14_dinov2.lvd142m, IMG_SIZE bội số 14 vd 518)
#                     | ConvNeXt (convnext_large.fb_in22k_ft_in1k, mọi IMG_SIZE)
BACKBONE="${BACKBONE:-vit_large_patch14_clip_224.openai}"
# W&B (override via env): WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE=online|offline|disabled
WANDB_PROJECT="${WANDB_PROJECT:-fraud-id}"
WANDB_MODE="${WANDB_MODE:-online}"

# tag backbone vào tên run -> out/clip_linear/clip_linear_<tag>_<ts>/ (DINOv2/CLIP/ConvNeXt KHÔNG lẫn nhau)
# RUN_TAG override (vd forensic_convnext) để phân biệt cùng-backbone khác-approach.
BB_TAG="$(echo "$BACKBONE" | grep -oE 'dinov3|dinov2|convnextv2|convnext|swin|eva02|aimv2|siglip2|siglip|clip' | head -1)"; BB_TAG="${BB_TAG:-other}"
RUN_TAG="${RUN_TAG:-$BB_TAG}"
RUN_NAME="clip_linear_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/logs"
LOG_FILE="$REPO/logs/${RUN_NAME}.log"

EXTRA_ARGS=("$@")
[ -n "$LEAVE_OUT" ] && EXTRA_ARGS+=(--leave-out-type "$LEAVE_OUT")
[ -n "${WANDB_ENTITY:-}" ] && EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")

echo "==> run=$RUN_NAME  backbone=$BACKBONE  gpu=$GPU  wandb_mode=$WANDB_MODE  log=$LOG_FILE"

# tee everything (stdout+stderr) to the log file; train.py also logs errors to W&B
CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL "$PY" src/train.py \
    --run-name "$RUN_NAME" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --img-size "$IMG_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --aug "$AUG" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-mode "$WANDB_MODE" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

echo "==> done. log saved to $LOG_FILE"
