"""Stage-A: pretrain the multi-task forgery model (detection + localization) on
IDNet. Produces forensic features that transfer to Stage-B (finetune on FREUID).

Validation holds out a COUNTRY (OOD-by-country) to mirror FREUID's unseen-type goal.
Logs detection AUC + APCER@1%BPCER (src/metrics) + mask IoU. Saves the FULL model.

Quick check:  python src/pretrain_stageA.py --countries EST --smoke
Full run:     see scripts/stageA/train.sh
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.dataset.idnet import (IDNetDataset, build_idnet_index,  # noqa: E402
                               idnet_transform)
from src.metrics import all_metrics  # noqa: E402
from src.models.multitask import MultiTaskForgery, dice_bce_mask_loss  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Stage-A multi-task pretrain on IDNet")
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--countries", nargs="*", default=None, help="subset; default=all extracted")
    p.add_argument("--val-country", default=None, help="hold out this country for OOD val")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--mask-weight", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out-dir", default=os.path.join(REPO, "out", "stageA"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--wandb-project", default="fraud-id-stageA")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, mask_weight, max_batches=0):
    model.eval()
    ys, ps, ious, losses = [], [], [], []
    bce = nn.BCEWithLogitsLoss()
    for i, (x, y, m) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, yf, m = x.to(device), y.float().to(device), m.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            cls, mask = model(x)
            loss = bce(cls, yf) + mask_weight * dice_bce_mask_loss(mask, m)
        losses.append(loss.item())
        ps.append(torch.sigmoid(cls).float().cpu().numpy()); ys.append(y.numpy())
        # mask IoU on fakes
        pm = (torch.sigmoid(torch.nn.functional.interpolate(
            mask, size=m.shape[-2:], mode="bilinear")) > 0.5).float()
        inter = (pm * m).sum((1, 2, 3)); union = ((pm + m) > 0).float().sum((1, 2, 3))
        iou = (inter / union.clamp(min=1))[y.to(device) == 1]
        if iou.numel():
            ious.append(iou.mean().item())
    y = np.concatenate(ys); p = np.concatenate(ps)
    met = all_metrics(p, y)
    met["loss"] = float(np.mean(losses))
    met["mask_IoU"] = float(np.mean(ious)) if ious else float("nan")
    return met


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    from datetime import datetime
    run_name = args.run_name or datetime.now().strftime("stageA_%Y%m%d_%H%M%S")
    if args.smoke:
        run_name = "smoke_" + run_name
    log_dir = os.path.join(args.out_dir, run_name); os.makedirs(log_dir, exist_ok=True)
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args),
               mode=args.wandb_mode, dir=args.out_dir)

    model = MultiTaskForgery(args.backbone, pretrained=True).to(device)
    print(f"device={device} | run={run_name} | mean={model.mean}")

    df = build_idnet_index(countries=args.countries)
    if args.val_country and args.val_country in set(df["country"]):
        train_df = df[df["country"] != args.val_country]
        val_df = df[df["country"] == args.val_country]
    else:                                  # random split (e.g. single-country smoke)
        train_df, val_df = train_test_split(df, test_size=0.15, random_state=0,
                                            stratify=df["label"])
    if args.smoke:
        train_df = train_df.groupby("label", group_keys=False).sample(min(500, len(train_df)//2), random_state=0)
        val_df = val_df.groupby("label", group_keys=False).sample(min(150, len(val_df)//2), random_state=0)
        args.epochs = 1
    print(f"train={len(train_df)} val={len(val_df)} "
          f"(val_country={args.val_country or 'random'})")

    ms = (args.img_size // 4, args.img_size // 4)
    tr_ds = IDNetDataset(train_df, idnet_transform(args.img_size, model.mean, model.std, True), ms)
    va_ds = IDNetDataset(val_df, idnet_transform(args.img_size, model.mean, model.std, False), ms)
    tr_ld = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
                       pin_memory=True, drop_last=True)
    va_ld = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.num_workers,
                       pin_memory=True)

    pw = torch.tensor([(train_df.label == 0).sum() / max((train_df.label == 1).sum(), 1)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * len(tr_ld), 1))

    best_auc, gstep = -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(tr_ld, desc=f"epoch {epoch}")
        for i, (x, y, m) in enumerate(pbar):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            x, yf, m = x.to(device, non_blocking=True), y.float().to(device), m.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                cls, mask = model(x)
                lcls = bce(cls, yf); lmask = dice_bce_mask_loss(mask, m)
                loss = lcls + args.mask_weight * lmask
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if gstep % 20 == 0:
                wandb.log({"train/loss": loss.item(), "train/cls": lcls.item(),
                           "train/mask": lmask.item(), "train/lr": sched.get_last_lr()[0]}, step=gstep)
                pbar.set_postfix(cls=f"{lcls.item():.3f}", mask=f"{lmask.item():.3f}")
            gstep += 1

        met = evaluate(model, va_ld, device, args.mask_weight)
        wandb.log({f"val/{k}": v for k, v in met.items() if isinstance(v, float)}, step=gstep)
        print(f"[epoch {epoch}] val loss={met['loss']:.4f} AUC={met['AUC']:.4f} "
              f"APCER@1%={met['APCER@1%BPCER']:.4f} maskIoU={met['mask_IoU']:.3f}")
        ckpt = {"model": model.state_dict(), "args": vars(args),
                "backbone": args.backbone, "epoch": epoch, "metrics": met}
        torch.save(ckpt, os.path.join(log_dir, "last.pt"))
        if not np.isnan(met["AUC"]) and met["AUC"] > best_auc:
            best_auc = met["AUC"]; torch.save(ckpt, os.path.join(log_dir, "best.pt"))

    wandb.finish()
    print(f"DONE. best val AUC={best_auc:.4f} | {log_dir}")


if __name__ == "__main__":
    main()