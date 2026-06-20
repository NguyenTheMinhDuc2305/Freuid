"""Competition metrics for FREUID — so local numbers are comparable to the LB.

Convention: score = P(fraud/attack) in [0,1]; label 1 = fraud (attack),
label 0 = genuine (bona-fide). Decision: flag as attack if score >= threshold.

  BPCER(t) = fraction of GENUINE (label 0) with score >= t   (genuine wrongly flagged)
  APCER(t) = fraction of FRAUD   (label 1) with score <  t   (fraud passes as genuine)

Both are error rates → LOWER IS BETTER (same direction as the Kaggle LB).
"""
import numpy as np


def apcer_at_bpcer(scores, labels, bpcer_target=0.01):
    """APCER at a fixed BPCER operating point (the secondary FREUID metric).

    Set the threshold so exactly `bpcer_target` of genuine docs are flagged,
    then report the fraction of frauds that slip through. Lower = better.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    genuine = scores[labels == 0]
    attack = scores[labels == 1]
    if len(genuine) == 0 or len(attack) == 0:
        return float("nan")
    # threshold = (1 - bpcer_target) quantile of genuine scores
    t = np.quantile(genuine, 1.0 - bpcer_target)
    apcer = float(np.mean(attack < t))      # frauds below threshold = passed
    return apcer


def eer(scores, labels):
    """Equal Error Rate: threshold where APCER == BPCER. Lower = better."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    thr = np.unique(scores)
    genuine, attack = scores[labels == 0], scores[labels == 1]
    if len(genuine) == 0 or len(attack) == 0:
        return float("nan")
    best, gap = 0.5, 1e9
    for t in thr:
        bpcer = np.mean(genuine >= t)
        apcer = np.mean(attack < t)
        if abs(bpcer - apcer) < gap:
            gap, best = abs(bpcer - apcer), (bpcer + apcer) / 2
    return float(best)


def audet(scores, labels, n=200):
    """Approx Area under the DET curve (APCER vs BPCER, linear). Lower = better.
    Primary FREUID metric is AuDET; exact definition unspecified, so this is an
    interpretable proxy — trust APCER@1%BPCER for LB comparison."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    genuine, attack = scores[labels == 0], scores[labels == 1]
    if len(genuine) == 0 or len(attack) == 0:
        return float("nan")
    ts = np.quantile(scores, np.linspace(0, 1, n))
    bpcer = np.array([np.mean(genuine >= t) for t in ts])
    apcer = np.array([np.mean(attack < t) for t in ts])
    order = np.argsort(bpcer)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # numpy 2.x renamed it
    return float(trapz(apcer[order], bpcer[order]))


def all_metrics(scores, labels):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels, int)
    out = {"n": len(labels), "n_fraud": int(labels.sum()),
           "APCER@1%BPCER": apcer_at_bpcer(scores, labels, 0.01),
           "APCER@10%BPCER": apcer_at_bpcer(scores, labels, 0.10),
           "EER": eer(scores, labels),
           "AuDET~": audet(scores, labels)}
    if len(np.unique(labels)) > 1:
        out["AUC"] = float(roc_auc_score(labels, scores))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
    # perfect separation -> APCER@1%BPCER ~ 0 ; random -> ~0.99
    perfect = np.r_[rng.uniform(0, .4, 1000), rng.uniform(.6, 1, 1000)]
    rand = rng.uniform(0, 1, 2000)
    print("perfect:", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in all_metrics(perfect, y).items()})
    print("random :", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in all_metrics(rand, y).items()})
