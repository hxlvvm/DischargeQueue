"""Score the test patients once and write the report. Refuses to run again once a report exists, so
"the test set was used once" can be checked from the commit history.

    python -m dischargequeue.final_eval
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS, N_BOOTSTRAP, REVIEW_FRACTION, SEED
from .data import assign_split, build_cohort, load_raw
from .evaluate import (brier, calibration_bins, capacity_curve, cluster_bootstrap, ranking_metrics,
                       top_k_mask)
from .features import build_features
from .models import prior_inpatient_rule
from .train import git_commit

REPORT = ARTIFACTS / "test_report.json"


def age_band(age):
    a = age.str.extract(r"\[(\d+)-")[0].astype(int)
    return pd.cut(a, [0, 50, 70, 200], right=False, labels=["<50", "50-69", "70+"]).astype(str)


def main():
    if REPORT.exists():
        sys.exit(f"{REPORT} already exists: the test set has been scored. Delete it only to reproduce.")
    art = joblib.load(ARTIFACTS / "model.joblib")
    df = assign_split(build_cohort(load_raw()))
    t = df[df["split"] == "test"].reset_index(drop=True)
    X, y, g = build_features(t), t["y"].to_numpy(), t["patient_nbr"].to_numpy()

    s_model = art["model"].predict_proba(X)[:, 1]          # ranks the queue
    p_model = art["calibrator"].predict_proba(X)[:, 1]     # displayed risk, Brier, reliability
    s_base = art["baseline"].predict_proba(X)[:, 1]
    s_rule = prior_inpatient_rule(X)
    scores = {art["chosen"]: s_model, "utilisation_lr": s_base, "prior_inpatient_rule": s_rule}
    f = REVIEW_FRACTION

    def paired(idx):
        out = {}
        for name, s in scores.items():
            m = ranking_metrics(y[idx], s[idx], f)
            out.update({f"{name}.{k}": v for k, v in m.items()})
        for k in ("auroc", "ap", "recall_at_capacity", "precision_at_capacity"):
            out[f"gain_vs_utilisation_lr.{k}"] = out[f"{art['chosen']}.{k}"] - out[f"utilisation_lr.{k}"]
        return out

    point = paired(np.arange(len(y)))
    ci = cluster_bootstrap(g, N_BOOTSTRAP, SEED, paired)

    first = t.groupby("patient_nbr").cumcount().to_numpy() == 0  # robustness: one encounter per patient
    flagged = top_k_mask(s_model, f)
    groups = {"age": age_band(t["age"]), "gender": t["gender"].fillna("missing"),
              "race": t["race"].fillna("missing")}
    sub = {}
    for kind, lab in groups.items():
        for val in sorted(lab.unique()):
            m = (lab == val).to_numpy()
            if m.sum() < 200 or y[m].sum() < 20:
                sub[f"{kind}={val}"] = {"n": int(m.sum()), "note": "too small to report"}
                continue
            sub_ci = cluster_bootstrap(g[m], 300, SEED, lambda i, m=m: {
                "auroc": ranking_metrics(y[m][i], s_model[m][i], f)["auroc"],
                "recall_flagged": float((flagged[m][i] & (y[m][i] == 1)).sum() / max(y[m][i].sum(), 1))})
            sub[f"{kind}={val}"] = {"n": int(m.sum()), "prevalence": float(y[m].mean()),
                                    "auroc": ranking_metrics(y[m], s_model[m], f)["auroc"],
                                    "share_flagged": float(flagged[m].mean()),
                                    "recall_flagged": float((flagged[m] & (y[m] == 1)).sum() / y[m].sum()),
                                    "ci": sub_ci}

    rep = {"commit": git_commit(), "chosen": art["chosen"], "n": int(len(y)), "patients": int(len(np.unique(g))),
           "positives": int(y.sum()), "prevalence": float(y.mean()), "review_fraction": f,
           "point": point, "ci95_patient_bootstrap": ci, "n_bootstrap": N_BOOTSTRAP,
           "brier": {"calibrated": brier(y, p_model), "prevalence_only": brier(y, np.full(len(y), y.mean()))},
           "calibration_bins": calibration_bins(y, p_model),
           "capacity_curve": {n: capacity_curve(y, s, np.arange(0.01, 0.51, 0.01)) for n, s in scores.items()},
           "first_encounter_only": {n: ranking_metrics(y[first], s[first], f) for n, s in scores.items()},
           "subgroups": sub}
    REPORT.write_text(json.dumps(rep, indent=1))

    keep = ["encounter_id", "age", "gender", "race", "time_in_hospital", "number_inpatient", "number_emergency",
            "num_medications", "number_diagnoses", "discharge_disposition_id", "A1Cresult", "insulin"]
    out = t[keep].copy()
    out["diag_1_group"] = X["diag_1_group"].to_numpy()
    out["score"], out["risk"], out["baseline_score"], out["readmitted_30d"] = s_model, p_model, s_base, y
    out.to_csv(ARTIFACTS / "test_scores.csv.gz", index=False, float_format="%.5f")
    c = rep["point"]
    print(f"test: n={len(y)} prevalence={y.mean():.3f}; {art['chosen']} AUROC {c[art['chosen'] + '.auroc']:.3f} "
          f"AP {c[art['chosen'] + '.ap']:.3f}; top {f:.0%}: precision {c[art['chosen'] + '.precision_at_capacity']:.3f} "
          f"recall {c[art['chosen'] + '.recall_at_capacity']:.3f} vs utilisation LR "
          f"{c['utilisation_lr.recall_at_capacity']:.3f}", flush=True)


if __name__ == "__main__":
    main()
