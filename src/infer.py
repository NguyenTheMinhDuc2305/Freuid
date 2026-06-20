"""Inference -> Kaggle submission for the CLIP-linear forgery detector.

Loads a trained checkpoint (only LN+head were tuned; backbone restored from timm),
predicts a fraud SCORE in [0,1] for EVERY image in --test-dir, and writes
submission.csv (id = filename stem). No sample_submission cross-check by default;
pass --align-submission to pad/reorder to a full sample_submission id list.

The FREUID metric is AuDET / APCER@1%BPCER — these are threshold-swept over a
continuous score, so we output the sigmoid PROBABILITY (not a hard 0/1) by default.
Use --hard-label only if the Kaggle page explicitly asks for 0/1.

Usage:  see scripts/clip_linear/infer.sh
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
from src.models.clip_classifier import CLIPLinearForgery  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Predict test set -> submission.csv")
    p.add_argument("--ckpt", required=True, help="path to best.pt / last.pt")
    p.add_argument("--test-dir", default="public_test",
                   help="test image folder relative to DATA_ROOT (all images here are predicted)")
    p.add_argument("--align-submission", default=None,
                   help="optional sample_submission.csv: reindex output to ITS full id "
                        "list, default-filling ids without an image. Off by default — "
                        "we just predict every image in --test-dir.")
    p.add_argument("--out", default=os.path.join(REPO, "out", "submission.csv"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--hard-label", action="store_true",
                   help="write 0/1 instead of probability (only if metric needs it)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--default-score", type=float, default=0.5,
                   help="score for align-submission ids that have no image")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "vit_large_patch14_clip_224.openai")
    img_size = ckpt.get("args", {}).get("img_size", 224)
    print(f"ckpt: backbone={backbone} img_size={img_size} "
          f"(trained val AUC={ckpt.get('auc')})")

    model = CLIPLinearForgery(backbone=backbone, pretrained=True, img_size=img_size).to(device)
    _, unexpected = model.load_state_dict(ckpt["trainable"], strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    model.eval()

    # predict EVERY image in the test dir (id = filename stem) — no csv cross-check
    abs_dir = os.path.join(DATA_ROOT, args.test_dir)
    files = sorted(f for f in os.listdir(abs_dir) if f.lower().endswith((".jpeg", ".jpg", ".png")))
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logit = model(x)
            scores.append(torch.sigmoid(logit).float().cpu().numpy())
            ids.extend(meta["id"])
    score_map = dict(zip(ids, np.concatenate(scores) if scores else []))

    # default: one row per predicted image. Optionally pad to a sample_submission.
    if args.align_submission:
        sub = pd.read_csv(args.align_submission)
        out = sub[["id"]].copy()
        out["label"] = out["id"].map(lambda i: score_map.get(i, args.default_score))
        miss = int(out["id"].map(lambda i: i not in score_map).sum())
        print(f"aligned to {args.align_submission}: {len(out)} rows, {miss} default-filled")
    else:
        out = present[["id"]].copy()
        out["label"] = out["id"].map(score_map)
    if args.hard_label:
        out["label"] = (out["label"] > args.threshold).astype(int)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows -> {args.out}")
    print(out["label"].describe() if not args.hard_label
          else out["label"].value_counts())


if __name__ == "__main__":
    main()
