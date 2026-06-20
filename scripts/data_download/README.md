# Tải dataset ngoài (đa dạng hoá cho Stage A pretrain)

> ⚠️ Dùng các bộ này **chỉ ở Stage A** (`scripts/pretrain_finetune/run.sh`), **Stage B vẫn finetune FREUID-only**.
> Gộp thẳng vào training đã làm public TỆ đi (xem `docs/timeline.md`). Giá trị của data ngoài là **private** (2 type lạ + ảnh chụp).
> Kiểm tra **luật FREUID có cho dùng data ngoài không** trước khi nộp. DocXPand là **NonCommercial**.

| Script | Bộ | Bổ sung gì | Cần đăng ký? |
|---|---|---|---|
| `midv_smartengines.sh dm` | **MIDV-DM** | manipulation + **MASK** localization → hợp multi-task | Không (FTP mở) |
| `midv_smartengines.sh 2020` | **MIDV-2020** | **ảnh chụp/scan vật lý** (genuine) | Không (FTP mở) |
| `sidtd.sh templates` / `sidtd.sh clips` | **SIDTD** | forged crop-replace; `clips` = frame **chụp** | Không (server tc11) |
| `docxpand.sh` | **DocXPand-25k** | 9 layout genuine → **nguồn SBI** | Không (GitHub release) |
| — (thủ công) | **FMIDV** | 7 forgery/ảnh trên MIDV-2020 | **Email** l3i-pn@univ-lr.fr |

Tất cả tải về `DATA/external/`. Script đều **resume được** (chạy lại nếu đứt mạng).

## Ưu tiên
1. `bash scripts/data_download/midv_smartengines.sh dm` — nhẹ, có mask, hợp Stage A nhất.
2. `bash scripts/data_download/sidtd.sh clips` — bù domain ảnh chụp + có nhãn forged.
3. `bash scripts/data_download/docxpand.sh` — chỉ khi làm nhánh SBI.

Sau khi tải xong → viết loader trong `src/dataset/combined.py` (giống IDNet/FantasyID) rồi bật ở Stage A.
