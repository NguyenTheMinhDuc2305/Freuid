#!/usr/bin/env bash
# MIDV-2020 + MIDV-DM (SmartEngines FTP, KHÔNG cần đăng ký).
#   MIDV-2020 : 1000 mock ID, mỗi cái có photo + scan + video clip -> domain ẢNH CHỤP/SCAN vật lý
#               (đúng phần private FREUID thiếu). Bản thân MIDV-2020 là GENUINE.
#   MIDV-DM   : Document Manipulation Detection + Localization, CÓ MASK -> hợp Stage A multi-task.
# FMIDV (7 forgery/ảnh trên MIDV-2020) phải XIN QUA EMAIL l3i-pn@univ-lr.fr -> không script được.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$REPO/DATA/external}"; mkdir -p "$OUT"; cd "$OUT"
WHICH="${1:-dm}"   # 'dm' = MIDV-DM (mask, nhẹ, nên lấy trước) | '2020' = MIDV-2020 (lớn, ảnh chụp)

# wget mirror FTP (resume + giữ cây thư mục). FTP smartengines không auth.
case "$WHICH" in
  dm)
    echo "==> MIDV-DM (manipulation + mask) -> $OUT/midv-dm"
    wget -c -r -np -nH --cut-dirs=1 -R "index.html*" \
         "ftp://smartengines.com/midv-dm/" -P "$OUT/midv-dm"
    echo "==> XONG MIDV-DM. mask localization kèm theo (xem README trong thư mục)." ;;
  2020)
    echo "==> MIDV-2020 (photo/scan/clip, ~lớn) -> $OUT/midv-2020"
    echo "    (để giảm dung lượng có thể tải riêng subfolder photo/ scan/ thay vì cả bộ)"
    wget -c -r -np -nH --cut-dirs=1 -R "index.html*" \
         "ftp://smartengines.com/midv-2020/" -P "$OUT/midv-2020"
    echo "==> XONG MIDV-2020." ;;
  *) echo "dùng: bash $0 [dm|2020]"; exit 1 ;;
esac
