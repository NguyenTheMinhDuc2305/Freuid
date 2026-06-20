"""Semi-weakly-supervised JOINT training.

One model, one training loop over the COMBINED stream (FREUID label-only +
IDNet/FantasyID label+mask). Classification loss on every sample; mask loss only
where a mask exists (has_mask=1). A held-out FREUID validation set (the metric we
care about) is evaluated each epoch with the competition metric APCER@1%BPCER.

Quick check:  python src/train_semiweak.py --smoke --idnet-countries EST
Full:         see scripts/semiweak/train.sh
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.dataset.combined import (CombinedDataset, build_combined_index,  # noqa: E402
                                  combined_transform)
from src.dataset.dataset import (FraudIDDataset, build_transforms,  # noqa: E402
                                  make_splits)
from src.metrics import all_metrics  # noqa: E402
from src.models.clip_multitask import CLIPMultiTask  # noqa: E402
from src.models.multitask import MultiTaskForgery  # noqa: E402


def build_model(backbone, img_size, unfreeze_blocks=0):
    """CLIP ViT backbone -> CLIPMultiTask (LN + last-K blocks + 2 heads); else MultiTaskForgery."""
    if "clip" in backbone or backbone.startswith("vit_"):
        return CLIPMultiTask(backbone, img_size=img_size, pretrained=True,
                             unfreeze_blocks=unfreeze_blocks), "clip_multitask"
    return MultiTaskForgery(backbone, pretrained=True), "multitask"


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in names}


def parse_args():
    p = argparse.ArgumentParser(description="Semi-weak joint training (FREUID + IDNet + FantasyID)")
    p.add_argument("--backbone", default="vit_large_patch14_clip_224.openai",
                   help="CLIP ViT-L (LN-tuning, foundation) -> CLIPMultiTask; "
                        "or edgenext_xx_small -> MultiTaskForgery")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)        # EdgeDoc
    p.add_argument("--weight-decay", type=float, default=5e-4)  # EdgeDoc
    p.add_argument("--mask-weight", type=float, default=3.0)    # EdgeDoc λ=3
    p.add_argument("--label-smoothing", type=float, default=0.05,
                   help="soft targets fake=1-ε, real=ε (chống overconfident; 0=tắt)")
    p.add_argument("--unfreeze-blocks", type=int, default=0,
                   help="CLIP: mở băng K transformer block cuối để MASK reshape được "
                        "feature (multi-task thật sự); 0=chỉ LN. Mỗi block ~12.6M params")
    p.add_argument("--patience", type=int, default=5,
                   help="early-stop after N epochs without FREUID-val AUC improvement")
    p.add_argument("--num-workers", type=int, default=8)
    # data sources
    p.add_argument("--use-idnet", type=int, default=1)
    p.add_argument("--use-fantasy", type=int, default=1)
    p.add_argument("--use-freuid", type=int, default=1,
                   help="0 = pretrain trên data NGOÀI only (Stage A), FREUID chỉ để val")
    p.add_argument("--init-from", default=None,
                   help="checkpoint best.pt để nạp weight ban đầu (Stage B finetune)")
    p.add_argument("--idnet-countries", nargs="*", default=None)
    p.add_argument("--idnet-cap-per-country", type=int, default=2000,
                   help="cap images/country/class so IDNet doesn't drown FREUID")
    # FREUID validation holdout
    p.add_argument("--freuid-val-frac", type=float, default=0.15)
    p.add_argument("--freuid-val-type", default=None,
                   help="hold out this FREUID type as OOD val (e.g. 'MAURITIUS/ID'); "
                        "else stratified random split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=os.path.join(REPO, "out", "semiweak"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--wandb-project", default="fraud-id-semiweak")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


def masked_mask_loss(mask_logit, mask_gt, has_mask):
    """BCE+Dice per sample, averaged ONLY over samples with has_mask=1."""
    mg = F.interpolate(mask_gt, size=mask_logit.shape[-2:], mode="area")
    mg = (mg > 0.5).float()
    bce = F.binary_cross_entropy_with_logits(mask_logit, mg, reduction="none").mean((1, 2, 3))
    p = torch.sigmoid(mask_logit)
    dice = 1 - (2 * (p * mg).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + mg.sum((1, 2, 3)) + 1)
    per = bce + dice
    w = has_mask.float()
    return (per * w).sum() / w.sum().clamp(min=1)


@torch.no_grad()
def eval_freuid(model, loader, device):
    model.eval()
    ys, ps, types = [], [], []
    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            cls, _ = model(x)
        ps.append(torch.sigmoid(cls).float().cpu().numpy()); ys.append(y.numpy())
        types.extend(meta["type"])
    y = np.concatenate(ys); p = np.concatenate(ps); types = np.array(types)
    m = all_metrics(p, y)
    m["per_type_auc"] = {t: float(roc_auc_score(y[types == t], p[types == t]))
                         for t in np.unique(types) if len(np.unique(y[types == t])) > 1}
    return m


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    run_name = args.run_name or datetime.now().strftime("semiweak_%Y%m%d_%H%M%S")
    if args.smoke:
        run_name = "smoke_" + run_name
    log_dir = os.path.join(args.out_dir, run_name); os.makedirs(log_dir, exist_ok=True)
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args),
               mode=args.wandb_mode, dir=args.out_dir)

    model, model_type = build_model(args.backbone, args.img_size, args.unfreeze_blocks)
    model = model.to(device)
    if args.init_from:                                     # Stage B: warm-start from Stage A
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        _, unexp = model.load_state_dict(ck.get("model", ck), strict=False)
        print(f"init weights from {args.init_from} (unexpected {len(unexp)})")
    ntr = sum(p.numel() for p in model.trainable_parameters()) / 1e6
    print(f"device={device} run={run_name} model={model_type} "
          f"trainable={ntr:.2f}M mean={model.mean}")

    # ---- FREUID train/val split (val = ONLY FREUID samples) ----
    ftr = os.path.join(REPO, "DATA", "freuid_train.csv")
    fva = os.path.join(REPO, "DATA", "freuid_val.csv")
    if args.freuid_val_type:                       # OOD-by-type override
        freuid_train, freuid_val = make_splits(seed=args.seed, leave_out_type=args.freuid_val_type)
        val_source = f"OOD-type={args.freuid_val_type}"
    elif os.path.exists(ftr) and os.path.exists(fva):
        import pandas as pd
        freuid_train, freuid_val = pd.read_csv(ftr), pd.read_csv(fva)
        val_source = "fixed leakage-free group-split (DATA/freuid_{train,val}.csv)"
    else:
        freuid_train, freuid_val = make_splits(val_frac=args.freuid_val_frac, seed=args.seed)
        val_source = "stratified-random (leaky)"
    # ---- combined train index (FREUID label-only + IDNet/FantasyID label+mask) ----
    idx = build_combined_index(
        freuid_train if args.use_freuid else freuid_train.iloc[:0],  # Stage A: exclude FREUID
        use_idnet=bool(args.use_idnet), use_fantasy=bool(args.use_fantasy),
        idnet_countries=args.idnet_countries,
        idnet_cap_per_country=args.idnet_cap_per_country)
    if args.smoke:
        idx = idx.groupby("source", group_keys=False).sample(frac=0.05, random_state=0)
        freuid_val = freuid_val.groupby("label", group_keys=False).sample(200, random_state=0)
        args.epochs = 1
    print("combined train by source/has_mask:\n",
          idx.groupby(["source", "has_mask"]).size().to_string())
    print(f"FREUID val (ONLY FREUID): {len(freuid_val)} | source = {val_source}")

    ms = (args.img_size // 4, args.img_size // 4)
    tr_ds = CombinedDataset(idx, combined_transform(args.img_size, model.mean, model.std, True), ms)
    val_ds = FraudIDDataset(freuid_val, transform=build_transforms(
        args.img_size, train=False, mean=model.mean, std=model.std), return_meta=True)
    tr_ld = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
                       pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

    pw = torch.tensor([(idx.label == 0).sum() / max((idx.label == 1).sum(), 1)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * len(tr_ld), 1))

    best_auc, gstep, no_improve = -1.0, 0, 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(tr_ld, desc=f"epoch {epoch}")
        for i, (x, y, m, hm) in enumerate(pbar):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            x, yf, m, hm = x.to(device), y.float().to(device), m.to(device), hm.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                cls, mask = model(x)
                eps = args.label_smoothing                  # fake:1-ε, real:ε
                lcls = bce(cls, yf * (1 - 2 * eps) + eps)
                lmask = masked_mask_loss(mask, m, hm)
                loss = lcls + args.mask_weight * lmask
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if gstep % 20 == 0:
                wandb.log({"train/loss": loss.item(), "train/cls": lcls.item(),
                           "train/mask": lmask.item(), "train/lr": sched.get_last_lr()[0],
                           "train/frac_masked": hm.float().mean().item()}, step=gstep)
                pbar.set_postfix(cls=f"{lcls.item():.3f}", mask=f"{lmask.item():.3f}")
            gstep += 1

        met = eval_freuid(model, val_ld, device)
        wandb.log({f"freuid_val/{k}": v for k, v in met.items() if isinstance(v, float)}, step=gstep)
        for t, a in met["per_type_auc"].items():
            wandb.log({f"freuid_val_auc/{t.replace('/', '-')}": a}, step=gstep)
        pt = " ".join(f"{t.split('/')[0]}={a:.3f}" for t, a in met["per_type_auc"].items())
        print(f"[epoch {epoch}] FREUID-val AUC={met['AUC']:.4f} "
              f"APCER@1%={met['APCER@1%BPCER']:.4f} | {pt}")
        ckpt = {"model": trainable_state(model), "model_type": model_type,
                "args": vars(args), "backbone": args.backbone,
                "img_size": args.img_size, "epoch": epoch, "metrics": met}
        torch.save(ckpt, os.path.join(log_dir, "last.pt"))
        if not np.isnan(met["AUC"]) and met["AUC"] > best_auc:
            best_auc = met["AUC"]; no_improve = 0
            torch.save(ckpt, os.path.join(log_dir, "best.pt"))
        else:
            no_improve += 1
            print(f"  no improvement {no_improve}/{args.patience} (best AUC={best_auc:.4f})")
            if no_improve >= args.patience:                 # EARLY STOPPING
                print(f"EARLY STOP at epoch {epoch} (no FREUID-val improvement in {args.patience}).")
                break

    wandb.summary["best_freuid_auc"] = best_auc
    wandb.finish()
    print(f"DONE. best FREUID-val AUC={best_auc:.4f} | {log_dir}")


if __name__ == "__main__":
    main()
