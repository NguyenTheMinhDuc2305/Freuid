"""IDNet dataset for Stage-A multi-task pretraining (detection + localization).

Layout (per country, e.g. extracted/EST/):
  positive/                       -> bona fide (label 0, empty mask)
  fraud5_inpaint_and_rewrite/     -> fake (label 1), bbox in meta/<C>_inpaint_and_rewrite.json
  fraud6_crop_and_replace/        -> fake (label 1), bbox in meta/<C>_crop_and_replace.json
  meta/*.json                     -> manipulated-region bounding boxes

Returns (image[3,H,W], label, mask[1,H,W]); mask = manipulated region (from bbox).
"""
import glob
import json
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IDNET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "DATA", "IDNet", "extracted")
FRAUD_DIRS = {
    "fraud5_inpaint_and_rewrite": "{c}_inpaint_and_rewrite.json",
    "fraud6_crop_and_replace": "{c}_crop_and_replace.json",
}


def _bbox_of(entry):
    """Return list of [x0,y0,x1,y1] manipulated boxes; handles both meta formats."""
    boxes = []
    for side in ("des", "src"):
        d = entry.get(side, {})
        bb = d.get("bbox") or d.get("region", {}).get("bbox")
        if bb:
            boxes.append(bb)
    return boxes


def build_idnet_index(root=IDNET_ROOT, countries=None):
    """One row per image: image_path, label, country, kind, key."""
    rows = []
    countries = countries or [d for d in os.listdir(root)
                              if os.path.isdir(os.path.join(root, d))]
    for c in sorted(countries):
        cdir = os.path.join(root, c)
        for p in glob.glob(os.path.join(cdir, "positive", "*")):
            rows.append((p, 0, c, "positive", None))
        for fdir, _ in FRAUD_DIRS.items():
            for p in glob.glob(os.path.join(cdir, fdir, "*")):
                key = os.path.splitext(os.path.basename(p))[0]
                rows.append((p, 1, c, fdir, key))
    return pd.DataFrame(rows, columns=["image_path", "label", "country", "kind", "key"])


def idnet_transform(img_size, mean, std, train=True):
    aug = []
    if train:
        aug = [
            A.Rotate(limit=4, border_mode=0, fill=0, p=0.3),
            A.ImageCompression(quality_range=(45, 95), p=0.5),
            A.OneOf([A.Downscale(scale_range=(0.5, 0.9), p=1.0),
                     A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.3),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.4),
            A.ISONoise(p=0.15),
        ]
    return A.Compose([A.Resize(img_size, img_size), *aug,
                      A.Normalize(mean=mean, std=std), ToTensorV2()])


class IDNetDataset(Dataset):
    def __init__(self, df, transform, mask_size=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mask_size = mask_size            # downsample mask to this (head resolution)
        self._meta = {}                       # (country,kind) -> json dict, cached

    def __len__(self):
        return len(self.df)

    def _meta_for(self, country, kind):
        cache_key = (country, kind)
        if cache_key not in self._meta:
            jf = os.path.join(IDNET_ROOT, country, "meta",
                              FRAUD_DIRS[kind].format(c=country))
            self._meta[cache_key] = json.load(open(jf)) if os.path.exists(jf) else {}
        return self._meta[cache_key]

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = np.asarray(Image.open(r["image_path"]).convert("RGB"))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        if r["label"] == 1:
            entry = self._meta_for(r["country"], r["kind"]).get(r["key"])
            if entry:
                for x0, y0, x1, y1 in _bbox_of(entry):
                    mask[max(0, y0):y1, max(0, x0):x1] = 1
        out = self.transform(image=img, mask=mask)
        m = out["mask"].float().unsqueeze(0)              # (1,H,W)
        if self.mask_size:
            m = torch.nn.functional.interpolate(
                m.unsqueeze(0), size=self.mask_size, mode="area")[0]
            m = (m > 0.5).float()
        return out["image"], int(r["label"]), m


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = build_idnet_index(countries=["EST"])
    print("index:", len(df), "| by kind:\n", df["kind"].value_counts().to_string())
    print("label balance:", df["label"].value_counts().to_dict())
    # viz: overlay mask on a few fakes to confirm alignment
    ds = IDNetDataset(df[df.label == 1].sample(4, random_state=0).reset_index(drop=True),
                      transform=idnet_transform(384, (0, 0, 0), (1, 1, 1), train=False))
    fig, ax = plt.subplots(2, 4, figsize=(16, 6))
    for i in range(4):
        img, lab, m = ds[i]
        rgb = img.permute(1, 2, 0).numpy()
        ax[0, i].imshow(rgb); ax[0, i].set_title(f"fake label={lab}", fontsize=9)
        ax[1, i].imshow(rgb); ax[1, i].imshow(m[0], alpha=0.4, cmap="Reds")
        ax[1, i].set_title(f"mask sum={int(m.sum())}", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout(); plt.savefig("out/idnet_mask_preview.png", dpi=85, bbox_inches="tight")
    print("saved out/idnet_mask_preview.png")
