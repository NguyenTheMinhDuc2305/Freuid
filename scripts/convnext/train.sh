#!/usr/bin/env bash
# ConvNeXt linear-probe — backbone CNN (đa dạng với ViT CLIP/DINO) để thêm vào ENSEMBLE.
#
# ConvNeXt = CNN hiện đại hoá (depthwise conv kernel lớn + LayerNorm + GELU). Dùng CHUNG
# pipeline linear-probe (src/train.py): ĐÓNG BĂNG backbone + tune LayerNorm + 1 linear head
# -> chống overfit, giữ feature pretrain mạnh, hợp để ensemble.
#
# Mặc định ConvNeXt V2 (FCMAE = pretrain self-supervised, nhạy texture hơn bản supervised V1).
# Lưu tách: out/clip_linear/clip_linear_convnext_<ts>/  (tag tự gắn theo backbone).
#
# Usage:
#   bash scripts/convnext/train.sh                 # ConvNeXt V2 @512
#   bash scripts/convnext/train.sh --smoke         # 1-epoch error check
#   BACKBONE=convnext_large.fb_in22k_ft_in1k bash scripts/convnext/train.sh   # đổi sang V1
#   IMG_SIZE=384 BATCH_SIZE=32 bash scripts/convnext/train.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ConvNeXt nhận MỌI img_size (resolution-agnostic) và nhẹ hơn ViT-L/14@512 -> batch lớn hơn.
export BACKBONE="${BACKBONE:-convnextv2_large.fcmae_ft_in22k_in1k}"
export IMG_SIZE="${IMG_SIZE:-512}"
export BATCH_SIZE="${BATCH_SIZE:-24}"

# tái dùng toàn bộ pipeline linear-probe (W&B, tagging backbone, logging, leakage-free val)
exec bash "$REPO/scripts/clip_linear/train.sh" "$@"
