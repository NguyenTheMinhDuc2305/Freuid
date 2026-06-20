"""Ensemble nhiều file submission -> 1 file nộp.

Metric FREUID là RANK-BASED (AuDET/APCER@1%BPCER) -> ensemble đúng nhất là RANK-AVERAGE:
mỗi model quy điểm về percentile-rank [0,1] rồi trung bình. Tránh việc 1 model thang điểm
khác (CLIP mượt vs DINO phân cực) lấn át khi cộng điểm thô.

Chỉ ensemble trên các id THỰC được chấm (ảnh có trong --scored-from); id còn lại giữ
--default-score (phần private chưa phát hành, không tính public).

Usage: xem scripts/clip_linear/ensemble.sh
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.dataset.dataset import DATA_ROOT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble submissions (rank-average)")
    p.add_argument("--inputs", nargs="+", required=True, help="≥2 file submission csv (id,label)")
    p.add_argument("--method", default="rank", choices=["rank", "mean"],
                   help="rank=trung bình percentile-rank (khuyến nghị) | mean=trung bình điểm thô")
    p.add_argument("--weights", default=None, help="trọng số mỗi file, vd '1,1' (mặc định đều nhau)")
    p.add_argument("--out", default=os.path.join(REPO, "out", "submission_ensemble.csv"))
    p.add_argument("--sample-submission", default=os.path.join(DATA_ROOT, "sample_submission.csv"))
    p.add_argument("--scored-from", default="public_test",
                   help="thư mục ảnh (dưới DATA) để xác định id thực được chấm")
    p.add_argument("--default-score", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    dfs = [pd.read_csv(f).set_index("id")["label"] for f in args.inputs]
    weights = ([float(w) for w in args.weights.split(",")] if args.weights
               else [1.0] * len(dfs))
    assert len(weights) == len(dfs), "số weight phải bằng số file"

    # id thực được chấm = ảnh có trong DATA/<scored-from>
    abs_dir = os.path.join(DATA_ROOT, args.scored_from)
    scored = sorted(os.path.splitext(f)[0] for f in os.listdir(abs_dir)
                    if f.lower().endswith((".jpeg", ".jpg", ".png")))
    print(f"id thực được chấm: {len(scored)} (từ DATA/{args.scored_from})")

    # ma trận điểm các model trên tập scored
    M = pd.DataFrame({f"m{i}": d.reindex(scored) for i, d in enumerate(dfs)})
    if M.isna().any().any():
        miss = M.isna().sum().to_dict()
        print(f"⚠️ thiếu id ở vài file (sẽ điền default): {miss}")
        M = M.fillna(args.default_score)

    # tương quan rank giữa các model (cao -> ensemble ít lợi; thấp -> bổ trợ tốt)
    if M.shape[1] == 2:
        rho = M["m0"].corr(M["m1"], method="spearman")
        print(f"Spearman rank-corr giữa 2 model: {rho:.3f}  "
              f"({'bổ trợ tốt' if rho < 0.9 else 'khá giống nhau'})")

    if args.method == "rank":
        R = M.rank(pct=True)                 # mỗi cột -> percentile [0,1]
    else:
        R = M
    w = np.array(weights) / np.sum(weights)
    ens = (R.values * w).sum(axis=1)         # trung bình có trọng số
    ens_map = dict(zip(scored, ens))

    # align ra full sample_submission, id không chấm -> default
    sub = pd.read_csv(args.sample_submission)[["id"]].copy()
    sub["label"] = sub["id"].map(lambda i: ens_map.get(i, args.default_score))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub.to_csv(args.out, index=False)

    r = sub.loc[sub["label"] != args.default_score, "label"]
    print(f"method={args.method} weights={weights}")
    print(f"điểm ensemble (phần thực): min={r.min():.3f} mean={r.mean():.3f} max={r.max():.3f} "
          f"| #giá_trị_khác_nhau={r.round(4).nunique()}")
    print(f"wrote {len(sub)} rows -> {args.out}")


if __name__ == "__main__":
    main()
