"""Batch scoring: a CSV of discharges in the raw table's format -> scores, calibrated risk and queue flag.

    python -m dischargequeue.score discharges.csv scored.csv
"""
import sys

import joblib
import pandas as pd

from .config import ARTIFACTS
from .features import build_features


def load_artifact(path=ARTIFACTS / "model.joblib"):
    return joblib.load(path)


def score_frame(df, art):
    X = build_features(df)
    out = df.copy()
    out["score"] = art["model"].predict_proba(X)[:, 1]
    out["risk"] = art["calibrator"].predict_proba(X)[:, 1]
    out["in_review_queue"] = out["score"] >= art["threshold"]
    return out


def main(src, dst):
    df = pd.read_csv(src, na_values="?", low_memory=False)
    out = score_frame(df, load_artifact())
    out.sort_values("score", ascending=False).to_csv(dst, index=False)
    print(f"scored {len(out)} discharges, {int(out['in_review_queue'].sum())} in the review queue -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
