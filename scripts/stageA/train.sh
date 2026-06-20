#!/usr/bin/env bash
# Stage-A: pretrain the multi-task forgery model (detection + localization) on IDNet.
# TURNKEY: auto-detects which IDNet countries are downloaded and, if >=2 exist,
# holds one out for OOD validation (mirrors FREUID's unseen-type goal). Just run:
#
#   bash scripts/stageA/train.sh
#
# Optional overrides (env):
#   VAL_COUNTRY=GRC          force which country is the OOD validation set
#   COUNTRIES="EST ESP FIN"  restrict to a subset of countries
#   EPOCHS=10 BATCH_SIZE=32 IMG_SIZE=384 LR=2e-4 MASK_WEIGHT=1.0 NUM_WORKERS=8
#   GPU=0  WANDB_MODE=online|offline|disabled
#
# Output: out/stageA/<run>/best.pt  (full model -> init for Stage B / finetune FREUID)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
PY="${PY:-python}"
[ -f "$REPO/scripts/.wandb_env" ] && source "$REPO/scripts/.wandb_env"

# ---- discover downloaded countries (those with a .done marker) ----
EXTRACT="$REPO/DATA/IDNet/extracted"
mapfile -t AVAIL < <(ls "$EXTRACT"/.*.done 2>/dev/null | sed 's#.*/\.##; s/\.done//' | sort)
if [ "${#AVAIL[@]}" -eq 0 ]; then
    echo "!! Chưa có nước IDNet nào tải xong trong $EXTRACT"
    echo "   Chạy tải trước: $PY src/data_prep/download_idnet.py"
    exit 1
fi
# allow restricting via COUNTRIES env
if [ -n "${COUNTRIES:-}" ]; then
    USE=($COUNTRIES)
else
    USE=("${AVAIL[@]}")
fi

# ---- pick OOD validation country: env override, else hold out the last one ----
if [ -z "${VAL_COUNTRY:-}" ] && [ "${#USE[@]}" -ge 2 ]; then
    VAL_COUNTRY="${USE[-1]}"
fi

EPOCHS="${EPOCHS:-10}"; BATCH_SIZE="${BATCH_SIZE:-32}"; IMG_SIZE="${IMG_SIZE:-384}"
LR="${LR:-2e-4}"; MASK_WEIGHT="${MASK_WEIGHT:-1.0}"; NUM_WORKERS="${NUM_WORKERS:-8}"
GPU="${GPU:-0}"; WANDB_MODE="${WANDB_MODE:-online}"
RUN_NAME="stageA_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$REPO/logs"
LOG="$REPO/logs/${RUN_NAME}.log"

echo "==> Nước có sẵn:     ${AVAIL[*]}"
echo "==> Dùng để train:   ${USE[*]}"
echo "==> OOD validation:  ${VAL_COUNTRY:-(random split — chỉ 1 nước)}"
echo "==> img=$IMG_SIZE batch=$BATCH_SIZE epochs=$EPOCHS  log=$LOG"

ARGS=(--run-name "$RUN_NAME" --countries "${USE[@]}" --epochs "$EPOCHS"
      --batch-size "$BATCH_SIZE" --img-size "$IMG_SIZE" --lr "$LR"
      --mask-weight "$MASK_WEIGHT" --num-workers "$NUM_WORKERS" --wandb-mode "$WANDB_MODE")
[ -n "${VAL_COUNTRY:-}" ] && ARGS+=(--val-country "$VAL_COUNTRY")

CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL "$PY" src/pretrain_stageA.py "${ARGS[@]}" \
    2>&1 | tee "$LOG"
echo "==> done. weight: out/stageA/${RUN_NAME}/best.pt  (dùng cho Stage B)"
