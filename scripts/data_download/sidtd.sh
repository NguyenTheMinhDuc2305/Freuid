#!/usr/bin/env bash
# SIDTD — forged ID (crop-replace portrait/text) dựng trên MIDV-2020.
#   kind=templates : ảnh ID phẳng scan (genuine + forged)
#   kind=clips     : frame VIDEO CHỤP -> domain ảnh chụp vật lý (gần private FREUID)
# Tải tự động qua package SIDTD (server tc11.cvc.uab.es). Nhãn: genuine/forged.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
OUT="${OUT:-$REPO/DATA/external}"; mkdir -p "$OUT"
KIND="${1:-templates}"   # templates | clips
SRC="$OUT/SIDTD_Dataset"

if [ ! -d "$SRC/.git" ]; then
    echo "==> clone repo SIDTD"
    git clone --depth 1 https://github.com/Oriolrt/SIDTD_Dataset "$SRC"
fi
cd "$SRC"
echo "==> cài package (Dataloader)"
"$PY" -m pip install -e . >/dev/null 2>&1 || "$PY" setup.py install >/dev/null 2>&1 || true

echo "==> tải SIDTD kind=$KIND (server tc11, có thể lâu)"
"$PY" SIDTD/data/DataLoader/Loader_Modules.py \
      --dataset SIDTD --kind "$KIND" --download_static
echo "==> XONG. Dữ liệu nằm trong cây package SIDTD/data/ (xem log dòng 'dataset_path')."
echo "    đổi kind: bash $0 clips   # lấy thêm frame ảnh chụp"
