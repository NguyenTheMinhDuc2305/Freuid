"""Inference + Test-Time Adaptation cho CLIP/DINOv2/ConvNeXt-linear forgery detector.

Mở rộng src/infer.py với 2 cơ chế TTA (xem docs/test_time_adaptation.md):
  - TTAug  : nhiều view (hflip + scale-jitter) -> TRUNG BÌNH sigmoid. Đổi ranking -> có tác dụng
             cho metric rank-based AuDET/APCER. An toàn (không đổi trọng số).
  - TENT-LN: minimize entropy dự đoán trên test, CHỈ tune LayerNorm affine, lr nhỏ, ít step.
             Mạnh hơn nhưng rủi ro collapse -> có guardrail (in entropy + phân bố điểm).

LƯU Ý metric rank-based: mọi biến đổi điểm ĐƠN ĐIỆU (z-score, min-max) KHÔNG đổi AuDET/APCER
-> không implement score-norm (chỉ in chẩn đoán).

Usage: scripts/clip_linear/infer_tta.sh
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.dataset.dataset import DATA_ROOT, FraudIDDataset, build_transforms  # noqa: E402
from src.models.clip_classifier import CLIPLinearForgery  # noqa: E402
from src.models.forensic_classifier import ForensicForgery  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Infer + Test-Time Adaptation -> submission.csv")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test-dir", default="public_test")
    p.add_argument("--align-submission", default=None)
    p.add_argument("--out", default=os.path.join(REPO, "out", "submission_tta.csv"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--default-score", type=float, default=0.5)
    # ---- TTA ----
    p.add_argument("--mode", default="ttaug",
                   choices=["none", "ttaug", "tent", "ttaug+tent"],
                   help="none=baseline | ttaug=augment | tent=entropy-min LN | ttaug+tent=cả hai")
    p.add_argument("--tta-scales", default="0.85",
                   help="danh sách scale-jitter cho TTAug, vd '0.85,1.15' (1.0 + hflip luôn có)")
    p.add_argument("--tent-steps", type=int, default=1, help="số lượt quét test khi TENT")
    p.add_argument("--tent-lr", type=float, default=1e-4)
    return p.parse_args()


def make_views(x, scales):
    """Trả về list view (B,3,H,W): gốc + hflip + scale-jitter (resize xuống rồi lên lại)."""
    H, W = x.shape[-2:]
    views = [x, torch.flip(x, dims=[3])]
    for s in scales:
        small = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        views.append(F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False))
    return views


def binary_entropy(logit):
    """H(p) trung bình, p=sigmoid(logit). Dùng cho TENT."""
    p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


def configure_tent(model):
    """Chỉ bật grad cho LayerNorm affine; đóng băng head + phần còn lại (TENT chuẩn)."""
    for p in model.parameters():
        p.requires_grad_(False)
    ln_params = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
                ln_params.append(p)
    return ln_params


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scales = [float(s) for s in args.tta_scales.split(",") if s.strip()] if args.tta_scales else []

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "vit_large_patch14_clip_224.openai")
    img_size = ckpt.get("args", {}).get("img_size", 224)
    forensic = ckpt.get("forensic", ckpt.get("args", {}).get("forensic", 0))
    ModelCls = ForensicForgery if forensic else CLIPLinearForgery
    print(f"ckpt: backbone={backbone} img_size={img_size} forensic={forensic} mode={args.mode} "
          f"(trained val AUC={ckpt.get('auc')})")

    model = ModelCls(backbone=backbone, pretrained=True, img_size=img_size).to(device)
    _, unexpected = model.load_state_dict(ckpt["trainable"], strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"

    abs_dir = os.path.join(DATA_ROOT, args.test_dir)
    files = sorted(f for f in os.listdir(abs_dir) if f.lower().endswith((".jpeg", ".jpg", ".png")))
    present = pd.DataFrame({"id": [os.path.splitext(f)[0] for f in files],
                            "image_path": [f"{args.test_dir}/{f}" for f in files]})
    print(f"found {len(present)} images in DATA/{args.test_dir}")
    tfm = build_transforms(img_size, train=False, mean=model.mean, std=model.std)
    ds = FraudIDDataset(present, transform=tfm, return_meta=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ---------- TENT: adapt LN affine bằng entropy minimization ----------
    if args.mode in ("tent", "ttaug+tent"):
        ln_params = configure_tent(model)
        opt = torch.optim.Adam(ln_params, lr=args.tent_lr)
        n_ln = sum(p.numel() for p in ln_params)
        print(f"TENT: tune {n_ln/1e3:.1f}K LN params, {args.tent_steps} step(s), lr={args.tent_lr}")
        model.train()                      # LN per-sample -> train/eval giống nhau, chỉ để có grad
        for step in range(args.tent_steps):
            ent_sum, nb = 0.0, 0
            for x, _, _ in tqdm(loader, desc=f"tent step {step}"):
                x = x.to(device, non_blocking=True)
                opt.zero_grad()
                loss = binary_entropy(model(x))   # fp32 cho ổn định
                loss.backward()
                opt.step()
                ent_sum += loss.item(); nb += 1
            print(f"  [tent step {step}] mean entropy={ent_sum/max(nb,1):.4f}  "
                  f"(giảm là tốt; ~0 mà điểm dồn 1 phía = COLLAPSE)")
    model.eval()

    # ---------- predict (kèm TTAug nếu chọn) ----------
    use_aug = args.mode in ("ttaug", "ttaug+tent")
    ids, scores = [], []
    with torch.no_grad():
        for x, _, meta in tqdm(loader, desc="infer"):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                if use_aug:
                    views = make_views(x, scales)
                    probs = torch.stack([torch.sigmoid(model(v)) for v in views], 0).mean(0)
                else:
                    probs = torch.sigmoid(model(x))
            scores.append(probs.float().cpu().numpy())
            ids.extend(meta["id"])
    sc = np.concatenate(scores) if scores else np.array([])
    score_map = dict(zip(ids, sc))

    # ---------- chẩn đoán phân bố (guardrail) ----------
    if len(sc):
        q = np.quantile(sc, [.01, .25, .5, .75, .99])
        near_half = float(np.mean(np.abs(sc - 0.5) < 1e-3))
        print(f"score: min={sc.min():.3f} mean={sc.mean():.3f} max={sc.max():.3f} | "
              f"q01/25/50/75/99={np.round(q,3)} | %~0.5={near_half*100:.1f}")
        if near_half > 0.5:
            print("  ⚠️ >50% điểm ~0.5 -> nghi model bỏ phiếu trắng / TENT collapse. Kiểm tra Val-OOD trước khi nộp!")

    # ---------- ghi submission ----------
    if args.align_submission:
        sub = pd.read_csv(args.align_submission)
        out = sub[["id"]].copy()
        out["label"] = out["id"].map(lambda i: score_map.get(i, args.default_score))
        miss = int(out["id"].map(lambda i: i not in score_map).sum())
        print(f"aligned to {args.align_submission}: {len(out)} rows, {miss} default-filled")
    else:
        out = present[["id"]].copy()
        out["label"] = out["id"].map(score_map)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
