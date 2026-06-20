# FREUID Challenge 2026 — Training code

Code train + load dataset cho bài phát hiện ID giả (FREUID, IJCAI-ECAI 2026).
Repo này **chỉ chứa code** — KHÔNG có dataset/weight (chuyển riêng). Best public hiện tại: **0.0506** (ensemble CLIP+DINO).

## 1. Cài môi trường
```bash
conda create -n fraud python=3.12 -y && conda activate fraud
# torch khớp CUDA của server (xem requirements.txt):
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 2. Đặt DATA
Mặc định code đọc `<repo>/DATA`. Trên server khác, copy data vào đó **hoặc** trỏ env:
```bash
export DATA_ROOT=/đường/dẫn/tới/DATA
```
`DATA/` cần có:
```
DATA/train_labels.csv         # id,image_path,label,is_digital,type
DATA/train/*.jpeg             # ảnh train (image_path trỏ tới đây)
DATA/public_test/*.jpeg       # ảnh test public (để infer)
DATA/sample_submission.csv    # 142,818 id (để align submission)
```
Tạo split leakage-free (group theo aHash template) — chạy 1 lần:
```bash
python -m src.data_prep.make_freuid_split     # -> DATA/freuid_{train,val}.csv
```

## 3. (tuỳ chọn) W&B
```bash
cp scripts/.wandb_env.example scripts/.wandb_env   # dán key vào; bỏ trống = không log
# hoặc tắt hẳn: export WANDB_MODE=disabled
```

## 4. Train (linear-probe: backbone ĐÓNG BĂNG + LN-tune + 1 linear head)
`PY` mặc định `python`; đổi bằng `export PY=/path/to/python`. Backbone đổi qua `BACKBONE`/`IMG_SIZE`.
```bash
# CLIP ViT-L @512  (nền tốt nhất hiện tại, public 0.0566)
BACKBONE=vit_large_patch14_clip_224.openai IMG_SIZE=512 bash scripts/clip_linear/train.sh

# DINOv2 @518
BACKBONE=vit_large_patch14_dinov2.lvd142m IMG_SIZE=518 bash scripts/clip_linear/train.sh

# EVA-02 / AIMv2 (semantic mạnh + đa dạng — đang muốn thử cho ensemble)
BACKBONE=eva02_large_patch14_448.mim_in22k_ft_in1k IMG_SIZE=448 bash scripts/clip_linear/train.sh
BACKBONE=aimv2_large_patch14_336.apple_pt_dist     IMG_SIZE=336 bash scripts/clip_linear/train.sh

# ConvNeXt V2 (CNN) / Forensic (SRM noise) — dùng folder riêng
bash scripts/convnext/train.sh
bash scripts/forensic/train.sh
```
Checkpoint: `out/clip_linear/clip_linear_<tag>_<ts>/best.pt` (tag = clip/dinov2/eva02/aimv2/convnext...).
Smoke test nhanh: thêm `--smoke`.

## 5. Infer + ensemble (tạo file nộp)
```bash
# infer 1 model -> submission (TTAug tuỳ chọn: MODE=ttaug)
CKPT=out/clip_linear/<run>/best.pt OUT=$PWD/out/sub_x.csv MODE=none \
  bash scripts/clip_linear/infer_tta.sh

# ensemble nhiều submission (rank-average — đúng cho metric rank-based AuDET/APCER)
python src/ensemble.py --inputs out/sub_clip.csv out/sub_dino.csv --method rank --weights 2,1 \
  --out out/sub_ensemble.csv
```

## 6. Lấy weight về máy chính
Train xong, copy `out/clip_linear/<run>/best.pt` về máy local để infer/nộp. Checkpoint chỉ lưu phần
trainable (LN+head, ~vài MB) — backbone tự tải lại từ timm khi infer.

## Cấu trúc
- `src/train.py` — train linear-probe (backbone-agnostic qua `--backbone`, `--forensic 1` cho SRM).
- `src/dataset/` — load dataset (`dataset.py`), augment (`augment.py`, `sbi.py`), combined/external.
- `src/models/` — `clip_classifier.py` (linear-probe), `forensic_classifier.py` (SRM front-end), ...
- `src/metrics.py` — APCER@1%BPCER / AuDET / AUC. `src/ensemble.py` — rank-average ensemble.
- `scripts/<approach>/` — runner cho từng hướng.

## Ghi chú
- **Path B (TruFor)**: `src/train_trufor.py` cần `third_party/TruFor` (clone riêng từ grip-unina/TruFor, KHÔNG nằm trong repo do nặng + có weight).
- Metric **rank-based** → trong ensemble chỉ rank-average/thứ-tự có tác dụng; chuẩn hoá điểm đơn điệu vô hiệu.
