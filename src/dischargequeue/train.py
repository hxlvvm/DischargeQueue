"""Fit the candidates on the training patients, choose one on the validation patients, calibrate it on
the calibration patients, and save it. The test patients are never read here.

    python -m dischargequeue.train
"""
import json
import subprocess

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from .config import ARTIFACTS, GBM_MIN_AP_GAIN, REVIEW_FRACTION, SEED
from .data import assign_split, build_cohort, load_raw
from .evaluate import ranking_metrics
from .features import build_features
from .models import full_gbm, full_lr, prior_inpatient_rule, utilisation_lr


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    df = assign_split(build_cohort(load_raw()))
    parts = {s: df[df["split"] == s] for s in ("train", "val", "calib")}
    X = {s: build_features(p) for s, p in parts.items()}
    y = {s: p["y"].to_numpy() for s, p in parts.items()}
    print({s: (len(p), int(p["y"].sum())) for s, p in parts.items()}, flush=True)

    candidates = {"utilisation_lr": utilisation_lr(), "full_lr": full_lr(), "full_gbm": full_gbm()}
    val = {"prior_inpatient_rule": ranking_metrics(y["val"], prior_inpatient_rule(X["val"]), REVIEW_FRACTION)}
    for name, model in candidates.items():
        model.fit(X["train"], y["train"])
        val[name] = ranking_metrics(y["val"], model.predict_proba(X["val"])[:, 1], REVIEW_FRACTION)
        print(name, {k: round(v, 4) for k, v in val[name].items()}, flush=True)

    # pre-registered rule (config.GBM_MIN_AP_GAIN): prefer the interpretable model unless GBM is clearly better
    chosen = "full_gbm" if val["full_gbm"]["ap"] >= val["full_lr"]["ap"] + GBM_MIN_AP_GAIN else "full_lr"
    model = candidates[chosen]
    # isotonic calibration on held-out patients; the queue still ranks on the uncalibrated score
    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(X["calib"], y["calib"])

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "baseline": candidates["utilisation_lr"],
                 "chosen": chosen}, ARTIFACTS / "model.joblib", compress=3)
    rec = {"commit": git_commit(), "seed": SEED, "chosen": chosen, "rule": f"full_gbm only if val AP >= full_lr AP + {GBM_MIN_AP_GAIN}",
           "n": {s: int(len(v)) for s, v in y.items()}, "positives": {s: int(v.sum()) for s, v in y.items()},
           "validation": val}
    (ARTIFACTS / "selection.json").write_text(json.dumps(rec, indent=1))
    print("chosen:", chosen, flush=True)


if __name__ == "__main__":
    main()
