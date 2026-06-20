"""Render out/experiments/registry.csv into a comparison table for easy method
selection. Lower APCER@1%BPCER / AuDET is better; higher AUC is better.

The OOD-validation runs (val_mode=ood-type) are the ones that reflect the FREUID
private test (unseen document types) — sort by those.
"""
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(REPO, "out", "experiments", "registry.csv")
OUT = os.path.join(REPO, "docs", "reports", "experiments.md")

COLS = ["exp_name", "aug", "sbi_prob", "val_mode", "ood_type", "img_size",
        "epochs", "best_AUC", "best_APCER@1%BPCER", "AuDET", "EER"]


def main():
    if not os.path.exists(REG):
        print(f"no registry yet at {REG}")
        return
    df = pd.read_csv(REG)
    df = df.drop_duplicates("exp_name", keep="last")

    lines = ["# Experiments — Data-centric generalization (so sánh method)", "",
             "> Tự sinh từ `out/experiments/registry.csv` (`python src/summarize_experiments.py`).",
             "> **APCER@1%BPCER / AuDET: THẤP hơn = tốt hơn** (giống LB). AUC: cao hơn tốt hơn.",
             "> Hàng `val_mode=ood-type` (validate trên type giữ lại) mới phản ánh private test.",
             ""]

    def md_table(sub, cols):
        out = ["| " + " | ".join(cols) + " |",
               "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, r in sub.iterrows():
            out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(out)

    for mode, sub in df.groupby("val_mode"):
        lines.append(f"## val_mode = {mode}")
        sub = sub.sort_values("best_APCER@1%BPCER")    # lower = better
        show = [c for c in COLS if c in sub.columns]
        lines.append(md_table(sub, show))
        lines.append("")
        if len(sub) and mode == "ood-type":
            best = sub.iloc[0]
            lines.append(f"➡️ **Tốt nhất (OOD):** `{best['exp_name']}` — "
                         f"APCER@1%BPCER={best['best_APCER@1%BPCER']}, AUC={best['best_AUC']}")
            lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}  ({len(df)} experiments)")
    print(df[[c for c in COLS if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
