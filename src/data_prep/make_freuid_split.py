"""Create a FIXED, leakage-free FREUID train/val split.

FREUID is template-redundant (EDA: ~62k/69k images share an aHash template), so a
random split leaks templates and inflates val to ~1.0. We GROUP by aHash template
and split per-type (stratified), so a template never appears in both train and val
-> the val honestly estimates performance on NEW images.

Output: DATA/freuid_train.csv, DATA/freuid_val.csv (same columns as train_labels).
"""
import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "DATA")
VAL_FRAC = 0.15
SEED = 42


def main():
    labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
    stats = pd.read_csv(os.path.join(REPO, "out", "eda_full", "full_stats.csv"),
                        dtype={"ahash": str})[["id", "ahash"]]
    df = labels.merge(stats, on="id", how="left")
    df["ahash"] = df["ahash"].fillna(df["id"])      # fallback: own id = its own group

    # per-type GroupShuffleSplit by aHash template (stratify by type, group by template)
    val_ids = []
    for t, sub in df.groupby("type"):
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
        _, va = next(gss.split(sub, groups=sub["ahash"]))
        val_ids += sub.iloc[va]["id"].tolist()
    val_ids = set(val_ids)

    val = df[df["id"].isin(val_ids)]
    train = df[~df["id"].isin(val_ids)]

    # guard: no template (aHash) shared across train/val (move any leaked one to train)
    leaked = set(train["ahash"]) & set(val["ahash"])
    if leaked:
        move = val["ahash"].isin(leaked)
        train = pd.concat([train, val[move]]); val = val[~move]
    assert not (set(train["ahash"]) & set(val["ahash"])), "template leak remains!"

    cols = list(labels.columns)
    train[cols].to_csv(os.path.join(DATA, "freuid_train.csv"), index=False)
    val[cols].to_csv(os.path.join(DATA, "freuid_val.csv"), index=False)

    print(f"train={len(train)}  val={len(val)}  (val_frac={len(val)/len(df):.3f})")
    print(f"shared templates train∩val: {len(set(train['ahash']) & set(val['ahash']))} (must be 0)")
    print("\nval balance theo (type,label):")
    print(pd.crosstab(val["type"], val["label"]))


if __name__ == "__main__":
    main()
