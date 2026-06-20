#!/usr/bin/env bash
# DocXPand-25k — 24,994 synthetic ID images, 9 layouts (cards/permits/passports).
# License: CC-BY-NC-SA 4.0 (NonCommercial!). Chủ yếu là ID GENUINE -> dùng làm
# NGUỒN TEMPLATE cho SBI (tự blend chân dung sinh fake), KHÔNG dùng làm fake trực tiếp.
# ~ vài GB, 12 part.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$REPO/DATA/external/DocXPand-25k}"; mkdir -p "$OUT"; cd "$OUT"
BASE="https://github.com/QuickSign/docxpand/releases/download/v1.0.0"

echo "==> tải 12 part vào $OUT (resume được, chạy lại nếu đứt mạng)"
for i in $(seq -w 0 11); do
    f="DocXPand-25k.tar.gz.$i"
    wget -c "$BASE/$f" -O "$f"
done

echo "==> giải nén (cat 12 part | tar)"
cat DocXPand-25k.tar.gz.* | tar xzvf -
echo "==> XONG. ảnh ở $OUT/images/ , field ở $OUT/fields/ , label JSON kèm theo."
echo "    code đọc: class DocFakerDataset (repo QuickSign/docxpand)."
