"""TruFor inference on the fraud dataset (pretrained weights, no fine-tuning).

Wraps the official test_docker inference code (third_party/TruFor) and runs it
on a balanced sample of our train set to sanity-check:
  - does the detection score separate real vs fake? (AUC)
  - does the localization map light up the portrait region on fakes?

Outputs:
  out/trufor_probe/scores.csv                per-image detection score
  out/trufor_probe/heatmaps/*.png            localization overlays for inspection
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRUFOR_SRC = os.path.join(REPO, "third_party", "TruFor", "test_docker", "src")
sys.path.insert(0, TRUFOR_SRC)
sys.path.insert(0, REPO)

from src.dataset.dataset import DATA_ROOT, TRAIN_CSV  # noqa: E402


def load_trufor(device: str = "cuda"):
    """Build TruFor (detconfcmx) and load official pretrained weights."""
    from config import _C as config  # TruFor's yacs config
    config.defrost()
    config.merge_from_file(os.path.join(TRUFOR_SRC, "trufor.yaml"))
    weights = os.path.join(REPO, "third_party", "TruFor", "test_docker",
                           "weights", "trufor.pth.tar")
    config.TEST.MODEL_FILE = weights
    config.freeze()

    from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    model = confcmx(cfg=config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    return model


@torch.no_grad()
def infer_image(model, img_path: str, device: str = "cuda"):
    """Replicates official preprocessing: RGB float / 256, batch 1, full res."""
    from PIL import Image
    rgb = np.array(Image.open(img_path).convert("RGB"))
    x = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float, device=device)
    x = (x / 256.0).unsqueeze(0)
    pred, conf, det, _ = model(x)
    score = torch.sigmoid(det).item()                    # image-level fake score
    loc_map = F.softmax(pred.squeeze(0), dim=0)[1].cpu().numpy()
    conf_map = torch.sigmoid(conf.squeeze(0))[0].cpu().numpy()
    return score, loc_map, conf_map


def save_overlay(img_path, loc_map, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    im = Image.open(img_path).convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].imshow(im); axes[0].set_title(title, fontsize=9)
    axes[1].imshow(loc_map, cmap="RdBu_r", vmin=0, vmax=1)
    axes[1].set_title("TruFor localization map", fontsize=9)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-label", type=int, default=100)
    ap.add_argument("--n-heatmaps", type=int, default=6,
                    help="overlays saved per label for visual check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "out", "trufor_probe"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model = load_trufor(device)
    print("TruFor weights loaded.")

    df = pd.read_csv(TRAIN_CSV)
    rng = np.random.default_rng(args.seed)
    sample = pd.concat([
        df[df["label"] == lab].sample(args.n_per_label, random_state=args.seed)
        for lab in (0, 1)
    ]).reset_index(drop=True)

    os.makedirs(os.path.join(args.out, "heatmaps"), exist_ok=True)
    records, saved = [], {0: 0, 1: 0}
    for i, row in sample.iterrows():
        p = os.path.join(DATA_ROOT, row["image_path"])
        try:
            score, loc_map, conf_map = infer_image(model, p, device)
        except Exception as e:
            print(f"FAIL {row['image_path']}: {e}")
            continue
        records.append({"id": row["id"], "type": row["type"],
                        "label": row["label"], "score": score,
                        "loc_mean": float(loc_map.mean()),
                        "loc_p99": float(np.quantile(loc_map, 0.99))})
        lab = int(row["label"])
        if saved[lab] < args.n_heatmaps:
            name = f"{'real' if lab == 0 else 'fake'}_{row['type'].replace('/', '-')}_{row['id'][:8]}.png"
            save_overlay(p, loc_map, os.path.join(args.out, "heatmaps", name),
                         f"{row['type']}  label={lab}  score={score:.3f}")
            saved[lab] += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(sample)}")

    res = pd.DataFrame(records)
    res.to_csv(os.path.join(args.out, "scores.csv"), index=False)

    from sklearn.metrics import roc_auc_score
    print("\n=== TruFor pretrained, zero-shot on our data ===")
    print(res.groupby("label")[["score", "loc_mean", "loc_p99"]]
             .agg(["mean", "std"]).round(4))
    print(f"\ndetection-score AUC : {roc_auc_score(res['label'], res['score']):.4f}")
    print(f"loc_p99 AUC         : {roc_auc_score(res['label'], res['loc_p99']):.4f}")
    print("\nper-type detection AUC:")
    for t, g in res.groupby("type"):
        if g["label"].nunique() == 2:
            print(f"  {t:15s} {roc_auc_score(g['label'], g['score']):.4f}  (n={len(g)})")
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
