"""Metrics for a review queue: ranking quality, precision and recall at a fixed review capacity,
calibration, and patient-clustered bootstrap intervals."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def top_k_mask(score, fraction):
    """The top `fraction` of rows by score (ties already broken by the scorer)."""
    k = int(np.ceil(fraction * len(score)))
    mask = np.zeros(len(score), bool)
    mask[np.argsort(-score, kind="stable")[:k]] = True
    return mask


def at_capacity(y, score, fraction):
    m = top_k_mask(score, fraction)
    tp = int(y[m].sum())
    return {"precision": tp / max(m.sum(), 1), "recall": tp / max(y.sum(), 1), "flagged": int(m.sum())}


def ranking_metrics(y, score, fraction):
    c = at_capacity(y, score, fraction)
    return {"auroc": roc_auc_score(y, score), "ap": average_precision_score(y, score),
            "precision_at_capacity": c["precision"], "recall_at_capacity": c["recall"]}


def capacity_curve(y, score, fractions):
    return [{"fraction": float(f), **at_capacity(y, score, f)} for f in fractions]


def calibration_bins(y, prob, n_bins=10):
    """Observed vs mean predicted risk in deciles of predicted risk."""
    order = np.argsort(prob, kind="stable")
    rows = []
    for idx in np.array_split(order, n_bins):
        rows.append({"mean_predicted": float(prob[idx].mean()), "observed": float(y[idx].mean()), "n": len(idx)})
    return rows


def brier(y, prob):
    return brier_score_loss(y, prob)


def cluster_bootstrap(groups, n_boot, seed, fn):
    """Resample patients with replacement, keep all encounters of each sampled patient, and apply
    fn(row_index) -> dict of numbers. Returns {name: (2.5th, 97.5th percentile)}."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws.append(fn(np.concatenate([rows_of[p] for p in pick])))
    return {k: (float(np.percentile([d[k] for d in draws], 2.5)), float(np.percentile([d[k] for d in draws], 97.5)))
            for k in draws[0]}
