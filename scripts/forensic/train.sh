#!/usr/bin/env bash
# Forensic baseline — SRM noise-residual front-end + backbone linear-probe.
# Ăn NOISE RESIDUAL thay vì RGB -> nhìn "bất nhất nhiễu" ở đường nối forgery, KHÔNG học thuộc
# template (chống overfit) -> tương quan THẤP với CLIP/DINO/ConvNeXt -> đẩy ENSEMBLE + tốt private.
#
# Dùng chung pipeline (src/train.py) qua cờ --forensic 1. Backbone CNN (ConvNeXt V2) hợp tín hiệu
# tầng thấp. Lưu tách: out/clip_linear/clip_linear_<tag>_<ts>/ (tag theo backbone).
#
# Usage:
#   bash scripts/forensic/train.sh                 # ConvNeXt V2 + SRM @512
#   bash scripts/forensic/train.sh --smoke         # 1-epoch error check
#   BACKBONE=vit_large_patch14_clip_224.openai IMG_SIZE=512 bash scripts/forensic/train.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export BACKBONE="${BACKBONE:-convnextv2_large.fcmae_ft_in22k_in1k}"
export IMG_SIZE="${IMG_SIZE:-512}"
export BATCH_SIZE="${BATCH_SIZE:-24}"
# tách tên run khỏi convnext-semantic: out/clip_linear/clip_linear_forensic_<bb>_<ts>/
_FBB="$(echo "$BACKBONE" | grep -oE 'dinov3|dinov2|convnext|swin|clip' | head -1)"
export RUN_TAG="forensic_${_FBB:-other}"

# bật SRM front-end; tái dùng toàn bộ pipeline linear-probe (W&B, tagging, leakage-free val)
exec bash "$REPO/scripts/clip_linear/train.sh" --forensic 1 "$@"
