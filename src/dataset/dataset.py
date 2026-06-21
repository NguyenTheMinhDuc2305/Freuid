"""Dataset loading for the ID-document fraud detection task.

Labels: 0 = real (bona-fide), 1 = fake (face-swap / text-inpaint).

Design notes (see docs/eda_statistical_methods.md):
  - Signal is local & subtle -> keep resolution reasonably high, never flip
    (ID layout/text direction is a real cue, flipping creates fake "anomalies").
  - Shortcut-breaking augs on train: JPEG re-compression, slight blur/jitter.
  - Splits are stratified by (type, label); leave-one-type-out supported to
    measure generalization the way SIDTD/IDNet protocols do.
"""
import os

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# Ảnh cụt/đang tải dở vẫn decode được thay vì raise (an toàn cho run dài).
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Portable: mặc định <repo>/DATA; ghi đè bằng env DATA_ROOT trên server khác.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(_REPO, "DATA"))
TRAIN_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SUBMISSION_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
PUBLIC_TEST_DIR = "public_test"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FraudIDDataset(Dataset):
    """Returns (image_tensor, label, meta_dict). label = -1 for unlabeled test."""

    def __init__(self, df: pd.DataFrame, root_dir: str = DATA_ROOT,
                 transform=None, return_meta: bool = False, sbi_prob: float = 0.0):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.return_meta = return_meta
        self.has_label = "label" in self.df.columns
        self.sbi_prob = sbi_prob          # P(turn a real image into a self-blended fake)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import random
        row = self.df.iloc[idx]
        path = os.path.join(self.root_dir, row["image_path"])
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            # 1 ảnh thiếu/hỏng không được làm chết cả epoch -> nhảy sang ảnh kế.
            # (ensure_data đã verify đủ ảnh; đây chỉ là lưới an toàn.)
            if not getattr(self, "_warned_missing", False):
                print(f"[FraudIDDataset] WARN ảnh lỗi/thiếu: {path} ({e}); bỏ qua.")
                self._warned_missing = True
            return self.__getitem__((idx + 1) % len(self.df))
        label = int(row["label"]) if self.has_label else -1
        # SBI: convert a REAL card into a synthetic fake (label 0 -> 1) before transform
        if self.sbi_prob > 0 and label == 0 and random.random() < self.sbi_prob:
            import numpy as np
            from src.dataset.sbi import self_blend
            img = Image.fromarray(self_blend(np.asarray(img)))
            label = 1
        if self.transform is not None:
            img = self.transform(img)
        if self.return_meta:
            meta = {"id": row["id"], "type": row.get("type", ""),
                    "image_path": row["image_path"]}
            return img, label, meta
        return img, label


def build_transforms(img_size: int = 512, train: bool = True,
                     normalize: bool = True, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                     aug: str = "default"):
    """Build a PIL-in transform.

    aug="default": light torchvision v2 aug (train) / resize-only (val).
    aug="p1p3"   : print-and-capture + resolution/quality aug (train only),
                   see src/dataset/augment.py. Falls back to default for val.
    mean/std default to ImageNet; pass CLIP stats for a CLIP backbone.
    """
    if train and aug in ("p1p3", "fourier", "p1p3_fourier"):
        from src.dataset.augment import AlbuWrapper, build_aug
        return AlbuWrapper(build_aug(aug, img_size, mean, std))   # normalizes internally

    t = [v2.ToImage(), v2.Resize((img_size, img_size), antialias=True)]
    if train:
        t += [
            v2.RandomApply([v2.JPEG(quality=(50, 95))], p=0.5),   # break re-save shortcuts
            v2.RandomApply([v2.ColorJitter(0.1, 0.1, 0.05)], p=0.3),
            v2.RandomApply([v2.GaussianBlur(3)], p=0.2),
        ]
    t += [v2.ToDtype(torch.float32, scale=True)]
    if normalize:
        t += [v2.Normalize(mean, std)]
    return v2.Compose(t)


def load_env():
    """Nạp .env (repo root) vào os.environ CHO RIÊNG process Python này (python-dotenv).
    Dùng load_dotenv thay vì 'source .env' để KHÔNG xuất key ra shell — an toàn trên server chung.
    No-op nếu chưa cài python-dotenv hoặc không có .env."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_REPO, ".env"))
    except Exception:
        pass


def _data_complete(data_root: str, sample: int = 400) -> bool:
    """True nếu train_labels.csv CÓ và (mẫu) ảnh tham chiếu đều tồn tại trên đĩa.
    Bắt trường hợp tải/giải nén DỞ DANG (CSV có nhưng ảnh thiếu) -> tránh chết giữa epoch."""
    labels = os.path.join(data_root, "train_labels.csv")
    if not os.path.exists(labels):
        return False
    import csv
    import random
    try:
        with open(labels, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    if not rows:
        return False
    col = "image_path" if "image_path" in rows[0] else None
    if col is None:                                       # layout lạ -> chỉ cần có thư mục train/
        td = os.path.join(data_root, "train")
        return os.path.isdir(td) and any(os.scandir(td))
    pick = rows if len(rows) <= sample else random.sample(rows, sample)
    missing = [r[col] for r in pick
               if not os.path.exists(os.path.join(data_root, r[col]))]
    if missing:
        print(f"[ensure_data] data CHƯA đủ: {len(missing)}/{len(pick)} ảnh mẫu thiếu "
              f"(vd {missing[0]}) -> sẽ tải/giải nén lại.")
        return False
    return True


def _extract_zips(data_root: str) -> None:
    import glob
    import zipfile
    for z in glob.glob(os.path.join(data_root, "*.zip")):
        print(f"[ensure_data] giải nén {os.path.basename(z)} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(data_root)
        os.remove(z)


def _download(data_root: str, competition: str) -> None:
    import subprocess
    import sys
    print(f"[ensure_data] tải Kaggle '{competition}' -> {data_root} ...")
    subprocess.run([sys.executable, "-m", "kaggle", "competitions", "download",
                    "-c", competition, "-p", data_root], check=True)


def ensure_data(data_root: str = DATA_ROOT,
                competition: str = "the-freuid-challenge-2026-ijcai-ecai") -> str:
    """Đảm bảo DATA đầy đủ; nếu thiếu/DỞ DANG -> tải Kaggle + giải nén, có VERIFY lại.

    Creds Kaggle đọc từ .env (KAGGLE_USERNAME + KAGGLE_KEY) qua load_dotenv, hoặc ~/.kaggle/kaggle.json.
    (pip install kaggle python-dotenv).
    """
    if _data_complete(data_root):
        return data_root
    load_env()                                            # nạp KAGGLE_* từ .env (chỉ process này)
    import glob
    import zipfile
    os.makedirs(data_root, exist_ok=True)
    # Lần trước có thể tải xong nhưng giải nén dở -> thử giải nén zip còn sót trước.
    if glob.glob(os.path.join(data_root, "*.zip")):
        try:
            _extract_zips(data_root)
        except zipfile.BadZipFile:                        # zip tải dở -> bỏ, tải lại bên dưới
            for z in glob.glob(os.path.join(data_root, "*.zip")):
                os.remove(z)
    if not _data_complete(data_root):                     # vẫn thiếu -> tải mới rồi giải nén
        _download(data_root, competition)
        _extract_zips(data_root)
    if not _data_complete(data_root):
        raise FileNotFoundError(
            f"Sau khi tải/giải nén, DATA tại {data_root} VẪN thiếu ảnh. Kiểm tra: "
            f"đĩa còn trống (df -h), Kaggle creds trong .env, đã Join/accept rules competition, "
            f"hoặc đặt DATA_ROOT trỏ tới data có sẵn.")
    print(f"[ensure_data] OK -> {data_root}")
    return data_root


def make_splits(csv_file: str = TRAIN_CSV, val_frac: float = 0.15,
                seed: int = 42, leave_out_type: str | None = None):
    """Stratified train/val split by (type,label).

    leave_out_type: e.g. "EGYPT/DL" -> train on the 4 other types and
    validate on the held-out one (generalization protocol).
    """
    ensure_data(os.path.dirname(csv_file) or DATA_ROOT)   # thử load, thiếu thì tải
    df = pd.read_csv(csv_file)
    if leave_out_type is not None:
        assert leave_out_type in set(df["type"]), f"unknown type {leave_out_type}"
        train_df = df[df["type"] != leave_out_type]
        val_df = df[df["type"] == leave_out_type]
    else:
        from sklearn.model_selection import train_test_split
        strat = df["type"].astype(str) + "_" + df["label"].astype(str)
        train_df, val_df = train_test_split(
            df, test_size=val_frac, random_state=seed, stratify=strat)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_test_df(submission_csv: str = SUBMISSION_CSV,
                  root_dir: str = DATA_ROOT,
                  test_dir: str = PUBLIC_TEST_DIR) -> pd.DataFrame:
    """Submission lists 142k ids but only released images exist on disk —
    keep rows whose file is present; missing ids get a default at submit time."""
    sub = pd.read_csv(submission_csv)
    sub["image_path"] = sub["id"].map(lambda i: f"{test_dir}/{i}.jpeg")
    exists = sub["image_path"].map(
        lambda p: os.path.exists(os.path.join(root_dir, p)))
    return sub[exists].drop(columns=["label"]).reset_index(drop=True)


def build_loaders(img_size: int = 512, batch_size: int = 32,
                  num_workers: int = 8, val_frac: float = 0.15,
                  seed: int = 42, leave_out_type: str | None = None,
                  normalize: bool = True, return_meta: bool = False,
                  mean=IMAGENET_MEAN, std=IMAGENET_STD,
                  train_df=None, val_df=None, aug: str = "default", sbi_prob: float = 0.0):
    if train_df is None or val_df is None:
        train_df, val_df = make_splits(val_frac=val_frac, seed=seed,
                                       leave_out_type=leave_out_type)
    train_ds = FraudIDDataset(train_df, transform=build_transforms(img_size, True, normalize, mean, std, aug),
                              return_meta=return_meta, sbi_prob=sbi_prob)
    val_ds = FraudIDDataset(val_df, transform=build_transforms(img_size, False, normalize, mean, std),
                            return_meta=return_meta)   # never SBI on val
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return train_ld, val_ld


if __name__ == "__main__":
    # smoke test: stratified split
    train_ld, val_ld = build_loaders(img_size=384, batch_size=16, num_workers=4)
    tdf, vdf = make_splits()
    print(f"train={len(tdf)}  val={len(vdf)}")
    print("val label balance:\n", vdf.groupby(["type", "label"]).size().unstack())
    x, y = next(iter(train_ld))
    print("train batch:", x.shape, x.dtype, "labels:", y[:8].tolist())
    x, y = next(iter(val_ld))
    print("val batch:  ", x.shape, "min/max:", round(float(x.min()), 2), round(float(x.max()), 2))

    # smoke test: leave-one-type-out + test set
    tdf, vdf = make_splits(leave_out_type="EGYPT/DL")
    print(f"\nleave-out EGYPT/DL -> train={len(tdf)} val={len(vdf)}",
          "| val types:", vdf['type'].unique().tolist())
    test_df = build_test_df()
    test_ds = FraudIDDataset(test_df, transform=build_transforms(384, False),
                             return_meta=True)
    img, label, meta = test_ds[0]
    print(f"test: {len(test_ds)} imgs | first: {meta['id']} label={label} shape={tuple(img.shape)}")
