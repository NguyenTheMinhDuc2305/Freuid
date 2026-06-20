#!/usr/bin/env bash
# Sinh sẵn các BASELINE Test-Time Adaptation cho 1 checkpoint -> nhiều file submission để so.
# Bám lưới ablation trong docs/test_time_adaptation.md (R0/R1/R2/R4/R5).
# Metric rank-based -> chỉ TTAug & TENT đổi được điểm; score-norm vô hiệu.
#
# Usage:
#   CKPT=out/clip_linear/clip_linear_dinov2_<ts>/best.pt bash scripts/clip_linear/tta_baselines.sh
#   PREFIX=dino CKPT=... bash scripts/clip_linear/tta_baselines.sh   # đặt tên file out
#
# ⚠️ Mỗi cấu hình PHẢI kiểm trên Val-OOD trước khi nộp (TENT có rủi ro collapse) — xem plan §3.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
CKPT="${CKPT:-$(ls -t "$REPO"/out/clip_linear/*/best.pt 2>/dev/null | grep -v '/loto_' | head -1)}"
[ -z "$CKPT" ] && { echo "No checkpoint. set CKPT=..."; exit 1; }
PREFIX="${PREFIX:-$(basename "$(dirname "$CKPT")")}"
BATCH_SIZE="${BATCH_SIZE:-24}"; GPU="${GPU:-0}"
OUTDIR="$REPO/out/tta_${PREFIX}"; mkdir -p "$OUTDIR"

# name | MODE | TTA_SCALES | TENT_STEPS   (rỗng = bỏ; encode hết qua env, không truyền cờ trùng)
RUNS=(
  "R0_none|none||"
  "R1_ttaug_flip|ttaug|||"
  "R2_ttaug_flip_scale|ttaug|0.85,1.15|"
  "R4_tent1|ttaug+tent|0.85,1.15|1"
  "R5_tent3|ttaug+tent|0.85,1.15|3"
)
echo "==> ckpt=$CKPT  -> $OUTDIR/"
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name mode scales steps <<< "$spec"
  out="$OUTDIR/submission_${name}.csv"
  echo "==> [$name] mode=$mode scales='${scales}' tent_steps='${steps:-–}'"
  CKPT="$CKPT" OUT="$out" MODE="$mode" TTA_SCALES="$scales" TENT_STEPS="${steps:-1}" \
    BATCH_SIZE="$BATCH_SIZE" GPU="$GPU" \
    bash scripts/clip_linear/infer_tta.sh 2>&1 | grep -aE 'score:|wrote|entropy|COLLAPSE' || true
done
echo "==> XONG. So sánh các file trong $OUTDIR/ (đo trên Val-OOD trước khi nộp)."
