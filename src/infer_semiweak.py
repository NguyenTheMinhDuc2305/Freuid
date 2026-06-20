"""Inference for the SEMI-WEAK multi-task model (MultiTaskForgery / EdgeNeXt).

Loads out/semiweak/<run>/best.pt, predicts a fraud SCORE in [0,1] (sigmoid of the
detection head) for every test image, and writes a Kaggle submission.csv.
The localization head is ignored at inference.

Usage:  see scripts/semiweak/infer.sh
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.dataset.dataset import DATA_ROOT, FraudIDDataset, build_transforms  # noqa: E402
from src.models.clip_multitask import CLIPMultiTask  # noqa: E402
from src.models.multitask import MultiTaskForgery  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Semi-weak model -> submission.csv")
    p.add_argument("--ckpt", required=True, help="out/semiweak/<run>/best.pt")
    p.add_argument("--test-dir", default="public_test",
                   help="test image folder relative to DATA_ROOT")
    p.add_argument("--align-submission", default=None,
                   help="sample_submission.csv to pad/reorder to (Kaggle needs all ids)")
    p.add_argument("--out", default=os.path.join(REPO, "out", "submission_semiweak.csv"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--default-score", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "edgenext_xx_small")
    img_size = ckpt.get("img_size", ckpt.get("args", {}).get("img_size", 384))
    mtype = ckpt.get("model_type", "multitask")
    print(f"ckpt: model={mtype} backbone={backbone} img_size={img_size} "
          f"val_metrics={ {k: round(v,4) for k,v in ckpt.get('metrics',{}).items() if isinstance(v,float)} }")

    if mtype == "clip_multitask":
        model = CLIPMultiTask(backbone, img_size=img_size, pretrained=True).to(device)
    else:
        model = MultiTaskForgery(backbone, pretrained=True).to(device)
    _, unexpected = model.load_state_dict(ckpt["model"], strict=False)  # trainable-only
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    model.eval()

    # predict every image in the test dir (id = filename stem)
    abs_dir = os.path.join(DATA_ROOT, args.test_dir)
    files = sorted(f for f in os.listdir(abs_dir)
                   if f.lower().endswith((".jpeg", ".jpg", ".png")))
    present = pd.DataFrame({"id": [os.path.splitext(f)[0] for f in files],
                            "image_path": [f"{args.test_dir}/{f}" for f in files]})
    print(f"found {len(present)} images in DATA/{args.test_dir}")

    tfm = build_transforms(img_size, train=False, mean=model.mean, std=model.std)
    ds = FraudIDDataset(present, transform=tfm, return_meta=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    ids, scores = [], []
    with torch.no_grad():
        for x, _, meta in tqdm(loader, desc="infer"):
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                cls, _ = model(x)                       # detection head only
            scores.append(torch.sigmoid(cls).float().cpu().numpy())
            ids.extend(meta["id"])
    score_map = dict(zip(ids, np.concatenate(scores)))

    if args.align_submission:                           # pad to full id list (Kaggle)
        sub = pd.read_csv(args.align_submission)
        out = sub[["id"]].copy()
        out["label"] = out["id"].map(lambda i: score_map.get(i, args.default_score))
        miss = int(out["id"].map(lambda i: i not in score_map).sum())
        print(f"aligned to {len(out)} rows ({miss} default-filled)")
    else:
        out = present[["id"]].copy()
        out["label"] = out["id"].map(score_map)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows -> {args.out}")
    print(out["label"].describe().round(4).to_string())


if __name__ == "__main__":
    main()
