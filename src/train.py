"""Train the classification-only CLIP-linear forgery detector.

- image-level binary labels only (no masks)
- AMP (bf16) on GPU, AdamW over LayerNorm + linear head
- Logging: Weights & Biases (wandb). Train loss/lr per step; val loss/acc/AUC +
  per-type AUC per epoch. Crashes are logged to the W&B run (alert + summary).
  View loss curves on the web under your account at the printed run URL.

Quick error check:   python src/train.py --smoke --wandb-mode offline
Full run:            see scripts/clip_linear/train.sh   (needs `wandb login` once)
"""
import argparse
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.dataset.dataset import TRAIN_CSV, build_loaders, load_env, make_splits  # noqa: E402
from src.models.clip_classifier import CLIPLinearForgery, count_params  # noqa: E402
from src.models.forensic_classifier import ForensicForgery  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="CLIP-linear forgery trainer (W&B logging)")
    p.add_argument("--backbone", default="vit_large_patch14_clip_224.openai")
    p.add_argument("--forensic", type=int, default=0,
                   help="1 = SRM noise-residual front-end (forensic baseline cho ensemble)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--leave-out-type", default=None,
                   help="e.g. 'EGYPT/DL' for leave-one-type-out generalization eval")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--aug", default="default",
                   choices=["default", "p1p3", "fourier", "p1p3_fourier"],
                   help="data-centric augmentation preset (see docs/generalization.md §1)")
    p.add_argument("--sbi-prob", type=float, default=0.0,
                   help="P(turn a real image into a self-blended synthetic fake)")
    p.add_argument("--exp-name", default=None,
                   help="experiment tag recorded in out/experiments/registry.csv")
    p.add_argument("--out-dir", default=os.path.join(REPO, "out", "clip_linear"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="subsample to ~2k train / 600 val, 1 epoch — fast error check")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    # Weights & Biases — view loss curves on the web under your account
    p.add_argument("--wandb-project", default="fraud-id")
    p.add_argument("--wandb-entity", default=None, help="your W&B username/team (optional)")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"],
                   help="online=upload to your account; offline=local only; disabled=off")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion, max_batches=0):
    """Returns (metrics_dict, preds_df). metrics_dict includes the competition
    metric APCER@1%BPCER + AuDET (src/metrics.py) so runs are LB-comparable."""
    import pandas as pd
    from src.metrics import all_metrics
    model.eval()
    losses, ys, ps, types, ids = [], [], [], [], []
    for i, (x, y, meta) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, yf = x.to(device, non_blocking=True), y.float().to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            logit = model(x)
            loss = criterion(logit, yf)
        losses.append(loss.item())
        ps.append(torch.sigmoid(logit).float().cpu().numpy())
        ys.append(y.numpy())
        types.extend(meta["type"])
        ids.extend(meta["id"])
    y = np.concatenate(ys); p = np.concatenate(ps)
    types = np.array(types)
    m = all_metrics(p, y)                       # APCER@1%BPCER, AuDET, EER, AUC
    m["loss"] = float(np.mean(losses))
    m["acc"] = float(((p > 0.5).astype(int) == y).mean())
    m["per_type_auc"] = {t: float(roc_auc_score(y[types == t], p[types == t]))
                         for t in np.unique(types)
                         if len(np.unique(y[types == t])) > 1}
    preds = pd.DataFrame({"id": ids, "type": types, "label": y, "score": p})
    return m, preds


def _record_experiment(args, log_dir, metrics):
    """Save the run config + append one row to the central experiment registry so
    augmentation/generalization methods are easy to compare side by side."""
    import csv
    import json
    cfg = {**vars(args), "log_dir": log_dir}
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    reg_dir = os.path.join(REPO, "out", "experiments")
    os.makedirs(reg_dir, exist_ok=True)
    reg = os.path.join(reg_dir, "registry.csv")
    row = {
        "exp_name": args.exp_name or os.path.basename(log_dir),
        "aug": args.aug, "sbi_prob": args.sbi_prob,
        "val_mode": "ood-type" if args.leave_out_type else "random",
        "ood_type": args.leave_out_type or "", "img_size": args.img_size,
        "epochs": args.epochs, "lr": args.lr,
        "best_AUC": round(metrics.get("AUC", float("nan")), 4),
        "best_APCER@1%BPCER": round(metrics.get("APCER@1%BPCER", float("nan")), 4),
        "AuDET": round(metrics.get("AuDET~", float("nan")), 4),
        "EER": round(metrics.get("EER", float("nan")), 4),
        "weight": os.path.join(log_dir, "best.pt"),
    }
    new = not os.path.exists(reg)
    with open(reg, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"recorded -> {reg}")


def main():
    load_env()                          # nạp .env (WANDB/KAGGLE) cho RIÊNG process này, không source ra shell
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    run_name = args.run_name or datetime.now().strftime("clip_linear_%Y%m%d_%H%M%S")
    if args.smoke:
        run_name = "smoke_" + run_name
    log_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name,
               config=vars(args), mode=args.wandb_mode, dir=args.out_dir)
    if wandb.run is not None and wandb.run.url:
        print(f"W&B run: {wandb.run.url}")
    print(f"device={device} | run={run_name}\nlogdir={log_dir}")

    # ---- model (defines the exact CLIP normalization to use) ----
    ModelCls = ForensicForgery if args.forensic else CLIPLinearForgery
    model = ModelCls(backbone=args.backbone, pretrained=True,
                     dropout=args.dropout, img_size=args.img_size).to(device)
    tr, tot = count_params(model)
    print(f"trainable {tr/1e6:.3f}M / {tot/1e6:.1f}M ({100*tr/tot:.2f}%) | "
          f"input={model.input_size} mean={model.mean}")
    wandb.config.update({"trainable_params": tr, "total_params": tot})
    wandb.watch(model, log="gradients", log_freq=200)

    # ---- data ----
    train_df, val_df = make_splits(TRAIN_CSV, args.val_frac, args.seed,
                                   args.leave_out_type)
    if args.smoke:
        # subsample balanced, then shuffle so truncated eval batches see both classes
        train_df = train_df.groupby("label", group_keys=False).sample(
            1000, random_state=args.seed).sample(frac=1, random_state=args.seed)
        val_df = val_df.groupby("label", group_keys=False).sample(
            300, random_state=args.seed).sample(frac=1, random_state=args.seed)
        args.epochs = 1
    pos_weight = torch.tensor(
        [(train_df["label"] == 0).sum() / max((train_df["label"] == 1).sum(), 1)],
        device=device)
    print(f"train={len(train_df)} val={len(val_df)} pos_weight={pos_weight.item():.3f}")

    train_loader, val_loader = build_loaders(
        img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, mean=model.mean, std=model.std,
        return_meta=True, train_df=train_df, val_df=val_df,
        aug=args.aug, sbi_prob=args.sbi_prob)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(args.epochs * len(train_loader), 1))

    best_auc, best_metrics, gstep = -1.0, {}, 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for i, (x, y, _) in enumerate(pbar):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            x, yf = x.to(device, non_blocking=True), y.float().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logit = model(x)
                loss = criterion(logit, yf)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            if gstep % args.log_every == 0:
                wandb.log({"train/loss": loss.item(),
                           "train/lr": sched.get_last_lr()[0],
                           "epoch": epoch}, step=gstep)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            gstep += 1

        m, preds = evaluate(model, val_loader, device, criterion, args.max_val_batches)
        auc, apcer = m["AUC"], m["APCER@1%BPCER"]
        log = {"val/loss": m["loss"], "val/acc": m["acc"], "val/auc": auc,
               "val/APCER_1BPCER": apcer, "val/AuDET": m["AuDET~"],
               "val/EER": m["EER"], "epoch": epoch}
        log.update({f"val_auc_type/{t.replace('/', '-')}": a
                    for t, a in m["per_type_auc"].items()})
        wandb.log(log, step=gstep)
        ptxt = " ".join(f"{t.split('/')[0]}={a:.3f}" for t, a in m["per_type_auc"].items())
        print(f"[epoch {epoch}] loss={m['loss']:.4f} acc={m['acc']:.4f} AUC={auc:.4f} "
              f"APCER@1%BPCER={apcer:.4f} | {ptxt}")

        # LN-tuning: only the trainable params are not reproducible from the
        # pretrained backbone, so save just those (~0.4MB vs 1.2GB).
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        ckpt = {"trainable": {k: v for k, v in model.state_dict().items()
                              if k in trainable_names},
                "args": vars(args), "backbone": args.backbone,
                "forensic": args.forensic,
                "epoch": epoch, "auc": auc, "apcer1": apcer}
        torch.save(ckpt, os.path.join(log_dir, "last.pt"))   # always
        if not np.isnan(auc) and auc > best_auc:              # best (guard nan)
            best_auc, best_metrics = auc, m
            torch.save(ckpt, os.path.join(log_dir, "best.pt"))
            preds.to_csv(os.path.join(log_dir, "val_predictions.csv"), index=False)

    wandb.summary["best_auc"] = best_auc
    wandb.finish()
    _record_experiment(args, log_dir, best_metrics)
    print(f"DONE. best val AUC={best_auc:.4f} "
          f"APCER@1%BPCER={best_metrics.get('APCER@1%BPCER', float('nan')):.4f} | {log_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        # surface the crash in the W&B run (alert + summary), then mark it failed
        try:
            if wandb.run is not None:
                wandb.alert(title="train.py crashed", text=tb)
                wandb.summary["error"] = tb
                wandb.finish(exit_code=1)
        except Exception:
            pass
        sys.exit(1)
