"""Fine-tune TruFor (forensic FRAMEWORK) cho DETECTION trên FREUID — nhãn ảnh, KHÔNG cần mask.

TruFor = Noiseprint++ (DnCNN) + encoder fuse RGB+noise + heads (localization/conf/detection).
Zero-shot AUC=0.45 (feature pretrained không hợp FREUID) -> phải fine-tune. Ta CHỈ supervise
nhánh DETECTION (BCE nhãn ảnh) nên KHÔNG cần mask:
  - đông băng Noiseprint++ (dncnn) + decode_head/_conf (localization không dùng)
  - fine-tune `backbone` (encoder) + `detection` head, LR thấp + early-stop (chống overfit)
Mục tiêu: 1 forensic member MẠNH + corr THẤP với CLIP/DINO cho ensemble (đòn UAM thắng private).

Usage: scripts/trufor/train.sh
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party", "TruFor", "test_docker", "src"))
sys.path.insert(0, REPO)

from src.dataset.dataset import TRAIN_CSV, FraudIDDataset, build_transforms, make_splits  # noqa: E402
from src.metrics import all_metrics  # noqa: E402
from src.models.trufor_infer import load_trufor  # noqa: E402


def set_trainable(model, mode):
    """Đông tất cả rồi mở theo mode. dncnn (Noiseprint++) LUÔN đông."""
    for p in model.parameters():
        p.requires_grad_(False)
    tr = []
    for n, p in model.named_parameters():
        g = n.split(".")[0]
        keep = (g == "detection")                                  # head luôn train
        if mode == "encoder_head" and g == "backbone":
            keep = True
        if mode == "all_minus_np" and g != "dncnn":
            keep = True
        if keep:
            p.requires_grad_(True); tr.append(p)
    return tr


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels, types = [], [], []
    for x, y, meta in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, _, det, _ = model(x)
        scores.append(torch.sigmoid(det.reshape(det.size(0), -1)[:, 0]).float().cpu().numpy())
        labels.append(y.numpy()); types += list(meta["type"])
    s, y = np.concatenate(scores), np.concatenate(labels)
    m = all_metrics(s, y)
    # per-type AUC
    dft = pd.DataFrame({"s": s, "y": y, "t": types})
    pt = {t: all_metrics(g.s.values, g.y.values)["AUC"] for t, g in dft.groupby("t") if g.y.nunique() > 1}
    return m, pt


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune TruFor detection on FREUID")
    p.add_argument("--run-name", default=None)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--trainable", default="encoder_head",
                   choices=["head", "encoder_head", "all_minus_np"])
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--leave-out-type", default=None)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-dir", default=os.path.join(REPO, "out", "trufor"))
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = args.run_name or "trufor_" + __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.smoke:
        run = "smoke_" + run
    log_dir = os.path.join(args.out_dir, run); os.makedirs(log_dir, exist_ok=True)
    print(f"device={device} | run={run}\nlogdir={log_dir}")

    model = load_trufor(device)                       # build + nạp pretrained
    tr = set_trainable(model, args.trainable)
    ntr = sum(p.numel() for p in tr); ntot = sum(p.numel() for p in model.parameters())
    print(f"trainable {ntr/1e6:.2f}M / {ntot/1e6:.1f}M ({100*ntr/ntot:.1f}%) | mode={args.trainable}")

    train_df, val_df = make_splits(TRAIN_CSV, args.val_frac, args.seed, args.leave_out_type)
    if args.smoke:
        train_df = train_df.groupby("label", group_keys=False).sample(400, random_state=0)
        val_df = val_df.groupby("label", group_keys=False).sample(150, random_state=0)
        args.epochs = 1
    pos_w = torch.tensor([(train_df.label == 0).sum() / max((train_df.label == 1).sum(), 1)], device=device)
    print(f"train={len(train_df)} val={len(val_df)} pos_weight={pos_w.item():.2f}")

    # TruFor ăn RGB [0,1] (KHÔNG ImageNet-normalize) -> normalize=False
    tfm_tr = build_transforms(args.img_size, train=True, normalize=False)
    tfm_va = build_transforms(args.img_size, train=False, normalize=False)
    ld_tr = DataLoader(FraudIDDataset(train_df, transform=tfm_tr, return_meta=True),
                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    ld_va = DataLoader(FraudIDDataset(val_df, transform=tfm_va, return_meta=True),
                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(tr, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * len(ld_tr), 1))

    best_auc, no_imp = -1.0, 0
    for ep in range(args.epochs):
        model.train(); model.dncnn.eval()             # giữ đông BN của Noiseprint++
        for x, y, _ in tqdm(ld_tr, desc=f"epoch {ep}"):
            x, yf = x.to(device, non_blocking=True), y.float().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, _, det, _ = model(x)
                loss = crit(det.reshape(det.size(0), -1)[:, 0], yf)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        m, pt = evaluate(model, ld_va, device)
        ptxt = " ".join(f"{t.split('/')[0]}={a:.3f}" for t, a in pt.items())
        print(f"[epoch {ep}] loss={loss.item():.4f} AUC={m['AUC']:.4f} "
              f"APCER@1%BPCER={m['APCER@1%BPCER']:.4f} AuDET={m['AuDET~']:.4f} | {ptxt}")
        ckpt = {"state_dict": model.state_dict(), "model_type": "trufor",
                "img_size": args.img_size, "trainable": args.trainable,
                "epoch": ep, "auc": m["AUC"], "apcer1": m["APCER@1%BPCER"]}
        torch.save(ckpt, os.path.join(log_dir, "last.pt"))
        if not np.isnan(m["AUC"]) and m["AUC"] > best_auc:
            best_auc = m["AUC"]; no_imp = 0
            torch.save(ckpt, os.path.join(log_dir, "best.pt"))
        else:
            no_imp += 1
            print(f"  no improvement {no_imp}/{args.patience} (best AUC={best_auc:.4f})")
            if no_imp >= args.patience:
                print("early stop"); break
    print(f"DONE. best val AUC={best_auc:.4f} | {log_dir}")


if __name__ == "__main__":
    main()
