"""Inference TruFor fine-tuned -> submission.csv (forensic member cho ensemble).

Nạp checkpoint từ src/train_trufor.py, chấm sigmoid(det) cho ảnh public_test (RGB [0,1],
resize img_size), align ra 142,818 dòng. Mask không dùng (chỉ điểm detection).
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
sys.path.insert(0, os.path.join(REPO, "third_party", "TruFor", "test_docker", "src"))
sys.path.insert(0, REPO)

from src.dataset.dataset import DATA_ROOT, FraudIDDataset, build_transforms  # noqa: E402
from src.models.trufor_infer import load_trufor  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="TruFor fine-tuned -> submission")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test-dir", default="public_test")
    p.add_argument("--align-submission", default=os.path.join(DATA_ROOT, "sample_submission.csv"))
    p.add_argument("--out", default=os.path.join(REPO, "out", "submission_trufor.csv"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--default-score", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    img_size = ckpt.get("img_size", 512)
    print(f"ckpt: model=trufor img_size={img_size} (trained val AUC={ckpt.get('auc')})")

    model = load_trufor(device)                          # build kiến trúc
    model.load_state_dict(ckpt["state_dict"])            # nạp weight fine-tuned
    model.eval()

    abs_dir = os.path.join(DATA_ROOT, args.test_dir)
    files = sorted(f for f in os.listdir(abs_dir) if f.lower().endswith((".jpeg", ".jpg", ".png")))
    present = pd.DataFrame({"id": [os.path.splitext(f)[0] for f in files],
                            "image_path": [f"{args.test_dir}/{f}" for f in files]})
    print(f"found {len(present)} images in DATA/{args.test_dir}")
    tfm = build_transforms(img_size, train=False, normalize=False)
    loader = DataLoader(FraudIDDataset(present, transform=tfm, return_meta=True),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    ids, scores = [], []
    with torch.no_grad():
        for x, _, meta in tqdm(loader, desc="infer"):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, _, det, _ = model(x)
            scores.append(torch.sigmoid(det.reshape(det.size(0), -1)[:, 0]).float().cpu().numpy())
            ids.extend(meta["id"])
    sc = np.concatenate(scores) if scores else np.array([])
    score_map = dict(zip(ids, sc))
    if len(sc):
        q = np.quantile(sc, [.01, .5, .99])
        print(f"score: min={sc.min():.3f} mean={sc.mean():.3f} max={sc.max():.3f} q01/50/99={np.round(q,3)}")

    sub = pd.read_csv(args.align_submission)[["id"]].copy()
    sub["label"] = sub["id"].map(lambda i: score_map.get(i, args.default_score))
    miss = int(sub["id"].map(lambda i: i not in score_map).sum())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"aligned: {len(sub)} rows, {miss} default-filled -> {args.out}")


if __name__ == "__main__":
    main()
