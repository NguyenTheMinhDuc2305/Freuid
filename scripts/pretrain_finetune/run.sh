#!/usr/bin/env bash
# 2-STAGE (tách riêng, KHÔNG gộp):
#   Stage A — PRETRAIN trên data NGOÀI (IDNet + FantasyID), FREUID chỉ để theo dõi.
#   Stage B — FINETUNE trên FREUID ONLY, khởi tạo từ checkpoint Stage A.
# Khác với semiweak (gộp 1 lượt): bước cuối chuyên hoá lại về FREUID -> hồi phục điểm public,
# init từ pretrain giúp generalize (private).
#
# Usage:
#   bash scripts/pretrain_finetune/run.sh
#   EPOCHS_A=8 EPOCHS_B=15 IDNET_CAP=4000 bash scripts/pretrain_finetune/run.sh
#   SKIP_A=out/semiweak/<runA>/last.pt bash scripts/pretrain_finetune/run.sh   # dùng lại Stage A cũ
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
PY="${PY:-python}"
[ -f "$REPO/scripts/.wandb_env" ] && source "$REPO/scripts/.wandb_env"

BACKBONE="${BACKBONE:-vit_large_patch14_clip_224.openai}"
IMG_SIZE="${IMG_SIZE:-384}"; BATCH_SIZE="${BATCH_SIZE:-24}"
LR="${LR:-3e-4}"; MASK_WEIGHT="${MASK_WEIGHT:-3.0}"; LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
UNFREEZE_BLOCKS="${UNFREEZE_BLOCKS:-2}"; NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS_A="${EPOCHS_A:-8}"; EPOCHS_B="${EPOCHS_B:-15}"
IDNET_CAP="${IDNET_CAP:-4000}"   # pretrain dùng NHIỀU IDNet hơn
GPU="${GPU:-0}"; WANDB_MODE="${WANDB_MODE:-online}"; TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/logs"
[ ! -f "$REPO/DATA/freuid_val.csv" ] && "$PY" src/data_prep/make_freuid_split.py

COMMON=(--backbone "$BACKBONE" --img-size "$IMG_SIZE" --batch-size "$BATCH_SIZE"
        --lr "$LR" --mask-weight "$MASK_WEIGHT" --label-smoothing "$LABEL_SMOOTHING"
        --unfreeze-blocks "$UNFREEZE_BLOCKS" --num-workers "$NUM_WORKERS"
        --wandb-mode "$WANDB_MODE")

# ---------- Stage A: pretrain on EXTERNAL only ----------
if [ -n "${SKIP_A:-}" ]; then
    CKPT_A="$SKIP_A"; echo "==> dùng lại Stage A: $CKPT_A"
else
    RUN_A="pf_A_${TS}"
    echo "==> STAGE A (pretrain external, FREUID excluded): $RUN_A"
    CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL "$PY" src/train_semiweak.py "${COMMON[@]}" \
        --run-name "$RUN_A" --use-freuid 0 --use-idnet 1 --use-fantasy 1 \
        --idnet-cap-per-country "$IDNET_CAP" --epochs "$EPOCHS_A" --patience 999 \
        --wandb-project fraud-id-pretrain 2>&1 | tee "$REPO/logs/${RUN_A}.log"
    CKPT_A="$REPO/out/semiweak/${RUN_A}/last.pt"     # dùng model pretrain đầy đủ
fi
[ -f "$CKPT_A" ] || { echo "!! không thấy checkpoint Stage A: $CKPT_A"; exit 1; }

# ---------- Stage B: finetune on FREUID only ----------
RUN_B="pf_B_${TS}"
echo "==> STAGE B (finetune FREUID-only, init từ Stage A): $RUN_B"
CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL "$PY" src/train_semiweak.py "${COMMON[@]}" \
    --run-name "$RUN_B" --use-idnet 0 --use-fantasy 0 --use-freuid 1 \
    --init-from "$CKPT_A" --epochs "$EPOCHS_B" --patience 5 \
    --wandb-project fraud-id-semiweak 2>&1 | tee "$REPO/logs/${RUN_B}.log"

echo "==> XONG. Stage A: $CKPT_A"
echo "==> Final (nộp): out/semiweak/${RUN_B}/best.pt"
echo "==> infer: CKPT=out/semiweak/${RUN_B}/best.pt bash scripts/semiweak/infer.sh"
