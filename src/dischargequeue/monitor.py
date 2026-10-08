"""Input monitoring for incoming discharge batches. Readmission labels arrive 30 days late, so these
checks run on inputs only: schema, unseen categories, missingness and population stability (PSI)
against a reference profile of the training patients.

    python -m dischargequeue.monitor reference          # write artifacts/reference_profile.json
    python -m dischargequeue.monitor check batch.csv    # print the report for one batch
    python -m dischargequeue.monitor demo               # clean vs deliberately shifted batch
"""
import json
import sys

import numpy as np
import pandas as pd

from .config import ARTIFACTS, SEED
from .data import assign_split, build_cohort, load_raw
from .features import CATEGORICAL, NUMERIC, RAW_INPUT, build_features

PROFILE = ARTIFACTS / "reference_profile.json"
PSI_ALERT = 0.2      # common rule of thumb: < 0.1 stable, 0.1-0.2 watch, > 0.2 shifted
MISSING_ALERT = 0.05  # absolute rise in the missing share of a raw column


def build_reference(raw_rows):
    X = build_features(raw_rows)
    prof = {"numeric": {}, "categorical": {}, "missing": raw_rows[RAW_INPUT].isna().mean().to_dict()}
    for c in NUMERIC:
        edges = np.unique(np.quantile(X[c], np.linspace(0, 1, 11)))
        prof["numeric"][c] = {"edges": edges.tolist(), "share": _hist(X[c], edges).tolist()}
    for c in CATEGORICAL:
        prof["categorical"][c] = X[c].value_counts(normalize=True).to_dict()
    return prof


def _hist(x, edges):
    inner = edges[1:-1]
    counts = np.bincount(np.searchsorted(inner, x, side="right"), minlength=len(edges) - 1)
    return counts / max(counts.sum(), 1)


def psi(expected, actual, eps=1e-4):
    e, a = np.clip(np.asarray(expected), eps, None), np.clip(np.asarray(actual), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def check_batch(batch, prof):
    missing_cols = [c for c in RAW_INPUT if c not in batch.columns]
    if missing_cols:
        return {"ok": False, "schema_errors": missing_cols}
    X = build_features(batch)
    rep = {"n": int(len(batch)), "schema_errors": [], "unseen_categories": {}, "missing_rise": {}, "psi": {}}
    for c in CATEGORICAL:
        ref = prof["categorical"][c]
        new = sorted(set(X[c].unique()) - set(ref))
        if new:
            rep["unseen_categories"][c] = new
        share = X[c].value_counts(normalize=True)
        keys = sorted(set(ref) | set(share.index))
        rep["psi"][c] = psi([ref.get(k, 0) for k in keys], [share.get(k, 0) for k in keys])
    for c in NUMERIC:
        edges = np.array(prof["numeric"][c]["edges"])
        rep["psi"][c] = psi(prof["numeric"][c]["share"], _hist(X[c], edges))
    miss = batch[RAW_INPUT].isna().mean()
    for c, v in miss.items():
        if v - prof["missing"][c] > MISSING_ALERT:
            rep["missing_rise"][c] = {"reference": round(prof["missing"][c], 3), "batch": round(float(v), 3)}
    rep["shifted_features"] = sorted(k for k, v in rep["psi"].items() if v > PSI_ALERT)
    rep["ok"] = not (rep["unseen_categories"] or rep["missing_rise"] or rep["shifted_features"])
    return rep


def main(cmd, *args):
    if cmd == "reference":
        df = assign_split(build_cohort(load_raw()))
        PROFILE.write_text(json.dumps(build_reference(df[df["split"] == "train"]), indent=1))
        print("wrote", PROFILE)
    elif cmd == "check":
        print(json.dumps(check_batch(pd.read_csv(args[0], na_values="?", low_memory=False),
                                     json.loads(PROFILE.read_text())), indent=1))
    elif cmd == "demo":
        prof = json.loads(PROFILE.read_text())
        df = assign_split(build_cohort(load_raw()))
        val = df[df["split"] == "val"]
        clean = val.sample(2000, random_state=SEED)
        shifted = val[val["age"].isin(["[70-80)", "[80-90)", "[90-100)"])].sample(2000, random_state=SEED).copy()
        rng = np.random.default_rng(SEED)
        shifted.loc[rng.random(len(shifted)) < 0.3, "A1Cresult"] = np.nan        # a lab feed drops values
        shifted.loc[shifted.index[:100], "admission_source_id"] = 99               # a new source code appears
        out = {"clean": check_batch(clean, prof), "shifted": check_batch(shifted, prof),
               "shift_applied": ["only patients aged 70+", "30 % of A1Cresult blanked", "admission_source_id 99 on 100 rows"]}
        (ARTIFACTS / "monitoring_demo.json").write_text(json.dumps(out, indent=1))
        for k in ("clean", "shifted"):
            r = out[k]
            print(k, "ok" if r["ok"] else "ALERT", "| shifted:", r["shifted_features"], "| unseen:",
                  list(r["unseen_categories"]), "| missing rise:", list(r["missing_rise"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
