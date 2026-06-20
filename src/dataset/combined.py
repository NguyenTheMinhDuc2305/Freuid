"""Combined SEMI-WEAKLY-supervised dataset for joint training.

Unifies three sources into one stream of (image, label, mask, has_mask):
  - FREUID  : label only (has_mask=0)   -> weak supervision (classification)
  - IDNet   : label + bbox->mask (has_mask=1)  full supervision
  - FantasyID: label + "altered" regions->mask (has_mask=1)  full supervision (captured face-swap)

The mask loss is applied only where has_mask=1 (see src/train_semiweak.py), so
FREUID still drives the classifier while the external masks teach localization.
"""
import glob
import json
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True   # ảnh ngoài (IDNet/FantasyID) đôi khi bị cắt cụt -> đừng để 1 file hỏng giết cả train

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "DATA")
IDNET_EXTRACT = os.path.join(DATA, "IDNet", "extracted")
FANTASY_ROOT = os.path.join(DATA, "FantasyID", "FantasyID")
IDNET_FRAUD = {"fraud5_inpaint_and_rewrite": "{c}_inpaint_and_rewrite.json",
               "fraud6_crop_and_replace": "{c}_crop_and_replace.json"}

# ---------- index builders (one row per image, uniform schema) ----------
COLS = ["image_path", "label", "source", "has_mask", "mask_ref"]
# mask_ref encodes how to render the mask: "" (none) | "idnet:<country>:<kind>:<key>"
#                                          | "fantasy:<json_path>"


def _freuid_rows(df):
    return [(os.path.join(DATA, r.image_path), int(r.label), "freuid", 0, "")
            for r in df.itertuples(index=False)]


def _idnet_rows(countries=None, cap_per_country=None):
    rows = []
    countries = countries or [d for d in os.listdir(IDNET_EXTRACT)
                              if os.path.isdir(os.path.join(IDNET_EXTRACT, d))]
    for c in sorted(countries):
        cdir = os.path.join(IDNET_EXTRACT, c)
        pos = glob.glob(os.path.join(cdir, "positive", "*"))
        if cap_per_country:
            pos = pos[:cap_per_country]
        rows += [(p, 0, "idnet", 1, "") for p in pos]
        for fdir in IDNET_FRAUD:
            fs = glob.glob(os.path.join(cdir, fdir, "*"))
            if cap_per_country:
                fs = fs[:cap_per_country]
            for p in fs:
                key = os.path.splitext(os.path.basename(p))[0]
                rows.append((p, 1, "idnet", 1, f"idnet:{c}:{fdir}:{key}"))
    return rows


def _fantasy_rows():
    rows = []
    for p in glob.glob(os.path.join(FANTASY_ROOT, "test", "attack", "**", "*.jpg"),
                       recursive=True):
        j = os.path.splitext(p)[0] + ".json"
        if os.path.exists(j):
            rows.append((p, 1, "fantasy", 1, f"fantasy:{j}"))   # all attacks = fake
    return rows


def build_combined_index(freuid_train_df, use_idnet=True, use_fantasy=True,
                         idnet_countries=None, idnet_cap_per_country=None):
    rows = _freuid_rows(freuid_train_df)
    if use_idnet and os.path.isdir(IDNET_EXTRACT):
        rows += _idnet_rows(idnet_countries, idnet_cap_per_country)
    if use_fantasy and os.path.isdir(FANTASY_ROOT):
        rows += _fantasy_rows()
    return pd.DataFrame(rows, columns=COLS)


# ---------- mask rendering ----------
def _idnet_bbox(entry):
    out = []
    for side in ("des", "src"):
        d = entry.get(side, {})
        bb = d.get("bbox") or d.get("region", {}).get("bbox")
        if bb:
            out.append(bb)
    return out


def _render_mask(mask_ref, h, w, cache):
    m = np.zeros((h, w), np.uint8)
    if not mask_ref:
        return m
    if mask_ref.startswith("png:"):                  # mask Gemini lưu sẵn (src/data_prep/gemini_mask.py)
        p = mask_ref.split(":", 1)[1]
        if os.path.exists(p):
            mp = np.asarray(Image.open(p).convert("L"))
            if mp.shape != (h, w):
                mp = np.asarray(Image.fromarray(mp).resize((w, h)))
            m[mp > 127] = 1
        return m
    if mask_ref.startswith("idnet:"):
        _, country, kind, key = mask_ref.split(":", 3)
        jf = os.path.join(IDNET_EXTRACT, country, "meta", IDNET_FRAUD[kind].format(c=country))
        meta = cache.setdefault(jf, json.load(open(jf)) if os.path.exists(jf) else {})
        for x0, y0, x1, y1 in _idnet_bbox(meta.get(key, {})):
            m[max(0, y0):y1, max(0, x0):x1] = 1
    elif mask_ref.startswith("fantasy:"):
        jf = mask_ref.split(":", 1)[1]
        meta = cache.setdefault(jf, json.load(open(jf)) if os.path.exists(jf) else {})
        for reg in meta.get("regions", []):
            if reg.get("region_attributes", {}).get("region_provenance") == "altered":
                s = reg["shape_attributes"]
                x, y = int(s["x"]), int(s["y"])
                m[max(0, y):y + int(s["height"]), max(0, x):x + int(s["width"])] = 1
    return m


def combined_transform(img_size, mean, std, train=True):
    aug = []
    if train:
        aug = [A.Rotate(limit=4, border_mode=0, fill=0, p=0.3),
               A.ImageCompression(quality_range=(45, 95), p=0.5),
               A.OneOf([A.Downscale(scale_range=(0.5, 0.9), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.3),
               A.RandomBrightnessContrast(0.15, 0.15, p=0.4), A.ISONoise(p=0.15)]
    return A.Compose([A.Resize(img_size, img_size), *aug,
                      A.Normalize(mean=mean, std=std), ToTensorV2()])


class CombinedDataset(Dataset):
    def __init__(self, df, transform, mask_size):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mask_size = mask_size            # (h,w) head resolution
        self._cache = {}                      # json cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        try:
            img = np.asarray(Image.open(r["image_path"]).convert("RGB"))
        except (OSError, ValueError) as e:   # file hỏng hoàn toàn -> nhảy sang sample khác, không crash
            print(f"[skip ảnh hỏng] {r['image_path']}: {e}")
            return self.__getitem__((idx + 1) % len(self.df))
        h, w = img.shape[:2]
        mask = _render_mask(r["mask_ref"], h, w, self._cache)
        out = self.transform(image=img, mask=mask)
        m = out["mask"].float().unsqueeze(0)
        m = torch.nn.functional.interpolate(m.unsqueeze(0), size=self.mask_size, mode="area")[0]
        m = (m > 0.5).float()
        return out["image"], int(r["label"]), m, int(r["has_mask"])


if __name__ == "__main__":
    from src.dataset.dataset import make_splits
    tr, _ = make_splits()
    idx = build_combined_index(tr.sample(2000, random_state=0),
                               idnet_countries=["EST"], idnet_cap_per_country=500)
    print("combined rows:", len(idx))
    print(idx.groupby(["source", "has_mask", "label"]).size())
    ds = CombinedDataset(idx, combined_transform(384, (0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225), True), (96, 96))
    for src in ["freuid", "idnet", "fantasy"]:
        sub = idx[idx.source == src]
        if len(sub):
            i = sub.index[0]
            im, lab, mk, hm = ds[i]
            print(f"{src}: img{tuple(im.shape)} label={lab} mask_sum={int(mk.sum())} has_mask={hm}")
