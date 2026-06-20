#!/usr/bin/env bash
# Run ONE data-centric generalization experiment with standardized saving.
# Each run -> out/clip_linear/<exp>/{best.pt,last.pt,config.json,val_predictions.csv}
# and one row appended to out/experiments/registry.csv (compare via
# `python src/summarize_experiments.py`).
#
# Validation: pass OOD_TYPE to validate on a HELD-OUT document type (reflects the
# private test of unseen types). Leave empty for the (leaky) random split.
#
# Usage:
#   EXP=baseline           OOD_TYPE="MAURITIUS/ID" bash scripts/experiment/run.sh
#   EXP=p1p3   AUG=p1p3    OOD_TYPE="MAURITIUS/ID" bash scripts/experiment/run.sh
#   EXP=sbi    SBI=0.3     OOD_TYPE="MAURITIUS/ID" bash scripts/experiment/run.sh
#   EXP=p1p3_sbi AUG=p1p3_fourier SBI=0.3 OOD_TYPE="MAURITIUS/ID" bash scripts/experiment/run.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
PY="${PY:-python}"
[ -f "$REPO/scripts/.wandb_env" ] && source "$REPO/scripts/.wandb_env"

EXP="${EXP:-exp}"
AUG="${AUG:-default}"          # default | p1p3 | fourier | p1p3_fourier
SBI="${SBI:-0.0}"             # P(real -> self-blended fake), e.g. 0.3
OOD_TYPE="${OOD_TYPE:-}"      # e.g. "MAURITIUS/ID" -> OOD validation; "" -> random
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-512}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU="${GPU:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-fraud-id-gen}"
WANDB_MODE="${WANDB_MODE:-online}"

RUN_NAME="exp_${EXP}_img${IMG_SIZE}"
mkdir -p "$REPO/logs"
LOG="$REPO/logs/${RUN_NAME}.log"
ARGS=(--run-name "$RUN_NAME" --exp-name "$EXP" --aug "$AUG" --sbi-prob "$SBI"
      --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --img-size "$IMG_SIZE" --lr "$LR"
      --num-workers "$NUM_WORKERS" --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
[ -n "$OOD_TYPE" ] && ARGS+=(--leave-out-type "$OOD_TYPE")

echo "==> exp=$EXP aug=$AUG sbi=$SBI ood=${OOD_TYPE:-none} img=$IMG_SIZE ep=$EPOCHS"
CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL "$PY" src/train.py "${ARGS[@]}" 2>&1 | tee "$LOG"
"$PY" src/summarize_experiments.py
echo "==> done. compare: docs/reports/experiments.md"
