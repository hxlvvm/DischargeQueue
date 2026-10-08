"""DischargeQueue demo. Loads precomputed artifacts only (held-out test scores and reports); no training."""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="DischargeQueue", layout="wide")


@st.cache_data
def load():
    rep = json.loads((ART / "test_report.json").read_text())
    sel = json.loads((ART / "selection.json").read_text())
    mon = json.loads((ART / "monitoring_demo.json").read_text())
    scores = pd.read_csv(ART / "test_scores.csv.gz")
    return rep, sel, mon, scores


rep, sel, mon, df = load()
chosen = rep["chosen"]
DISPOSITION = {1: "home", 3: "skilled nursing facility", 6: "home with health service", 18: "unknown",
               22: "rehab facility", 5: "other inpatient care", 25: "not mapped", 4: "intermediate care facility",
               7: "left against medical advice"}

st.title("DischargeQueue")
st.caption("Which discharged diabetes patients should an outreach team call first? A capacity-aware 30-day "
           "readmission review queue, evaluated on held-out patients of the UCI Diabetes 130-US hospitals data "
           "(de-identified, 1999-2008). Research demo, not a clinical tool.")

frac = st.slider("Review capacity: share of discharges the team can call", 0.01, 0.30, rep["review_fraction"], 0.01,
                 format="%.2f")
order = np.argsort(-df["score"].to_numpy(), kind="stable")
k = int(np.ceil(frac * len(df)))
top = df.iloc[order[:k]]
base_top = df.iloc[np.argsort(-df["baseline_score"].to_numpy(), kind="stable")[:k]]
pos = df["readmitted_30d"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Discharges to call", f"{k:,} of {len(df):,}")
c2.metric("Precision (called who were readmitted)", f"{top['readmitted_30d'].mean():.1%}",
          f"{top['readmitted_30d'].mean() / df['readmitted_30d'].mean():.1f}x the base rate", delta_color="off")
c3.metric("Readmissions reached", f"{top['readmitted_30d'].sum() / pos:.1%}",
          f"{(top['readmitted_30d'].sum() - base_top['readmitted_30d'].sum()) / pos:+.1%} vs prior-visit baseline")
c4.metric("Base rate", f"{df['readmitted_30d'].mean():.1%}")

tab_q, tab_e, tab_f, tab_m, tab_a = st.tabs(["Review queue", "Evaluation", "Subgroups", "Monitoring", "About"])

with tab_q:
    show = top.copy()
    show["discharged to"] = show["discharge_disposition_id"].map(DISPOSITION).fillna("other")
    show = show.rename(columns={"risk": "calibrated risk", "number_inpatient": "prior inpatient visits",
                                "number_emergency": "prior ER visits", "time_in_hospital": "days in hospital",
                                "diag_1_group": "primary diagnosis", "readmitted_30d": "readmitted (held-out label)"})
    st.dataframe(show[["calibrated risk", "age", "gender", "prior inpatient visits", "prior ER visits",
                       "days in hospital", "primary diagnosis", "discharged to", "A1Cresult", "insulin",
                       "readmitted (held-out label)"]].head(200),
                 hide_index=True, use_container_width=True,
                 column_config={"calibrated risk": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    st.caption("Ordered by model score; the first 200 rows are shown. The held-out label is visible here only "
               "because this is an evaluation set.")

with tab_e:
    p, ci = rep["point"], rep["ci95_patient_bootstrap"]
    rows = []
    for name, label in [(chosen, "Gradient boosting, all discharge-time features" if chosen == "full_gbm"
                         else "Logistic regression, all discharge-time features"),
                        ("utilisation_lr", "Baseline: prior outpatient/ER/inpatient visit counts"),
                        ("prior_inpatient_rule", "Rule: prior inpatient visits")]:
        fmt = lambda m: f"{p[f'{name}.{m}']:.3f} ({ci[f'{name}.{m}'][0]:.3f}-{ci[f'{name}.{m}'][1]:.3f})"  # noqa: E731
        rows.append({"model": label, "AUROC": fmt("auroc"), "average precision": fmt("ap"),
                     f"precision @ {rep['review_fraction']:.0%}": fmt("precision_at_capacity"),
                     f"recall @ {rep['review_fraction']:.0%}": fmt("recall_at_capacity")})
    st.subheader(f"Held-out test: {rep['n']:,} discharges, {rep['patients']:,} patients, "
                 f"{rep['positives']:,} readmissions ({rep['prevalence']:.1%})")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    g = ci["gain_vs_utilisation_lr.recall_at_capacity"]
    st.caption(f"95 % intervals from {rep['n_bootstrap']} bootstrap resamples of patients. Gain in recall at "
               f"{rep['review_fraction']:.0%} over the baseline: {p['gain_vs_utilisation_lr.recall_at_capacity']:+.3f} "
               f"({g[0]:+.3f} to {g[1]:+.3f}).")
    curve = pd.concat([pd.DataFrame(v).assign(model=n) for n, v in rep["capacity_curve"].items()])
    left, right = st.columns(2)
    left.altair_chart(alt.Chart(curve).mark_line().encode(
        x=alt.X("fraction:Q", title="share of discharges called", axis=alt.Axis(format="%")),
        y=alt.Y("recall:Q", title="share of readmissions reached", axis=alt.Axis(format="%")),
        color="model:N").properties(title="Capacity curve", height=320), use_container_width=True)
    cal = pd.DataFrame(rep["calibration_bins"])
    diag = alt.Chart(pd.DataFrame({"x": [0, cal["mean_predicted"].max()]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="x")
    right.altair_chart((diag + alt.Chart(cal).mark_line(point=True).encode(
        x=alt.X("mean_predicted:Q", title="predicted risk (calibrated)"),
        y=alt.Y("observed:Q", title="observed readmission rate"))).properties(title="Calibration by decile", height=320),
        use_container_width=True)
    st.caption(f"Brier score {rep['brier']['calibrated']:.4f} vs {rep['brier']['prevalence_only']:.4f} for predicting "
               "the base rate for everyone. Robustness, first encounter per patient only: AUROC "
               f"{rep['first_encounter_only'][chosen]['auroc']:.3f} (baseline "
               f"{rep['first_encounter_only']['utilisation_lr']['auroc']:.3f}).")

with tab_f:
    sub = []
    for key, v in rep["subgroups"].items():
        if "auroc" not in v:
            sub.append({"group": key, "n": v["n"], "note": v["note"]})
            continue
        sub.append({"group": key, "n": v["n"], "base rate": f"{v['prevalence']:.1%}",
                    "AUROC (95 % CI)": f"{v['auroc']:.3f} ({v['ci']['auroc'][0]:.3f}-{v['ci']['auroc'][1]:.3f})",
                    "share called": f"{v['share_flagged']:.1%}",
                    "readmissions reached (95 % CI)": f"{v['recall_flagged']:.1%} "
                    f"({v['ci']['recall_flagged'][0]:.1%}-{v['ci']['recall_flagged'][1]:.1%})"})
    st.dataframe(pd.DataFrame(sub), hide_index=True, use_container_width=True)
    st.caption("At the global 10 % cut. Wide intervals mean the data cannot tell the groups apart; small groups "
               "are not reported. This is a check, not a fairness certification.")

with tab_m:
    st.write("Labels arrive 30 days after discharge, so day-to-day monitoring runs on inputs: schema, unseen "
             "categories, missing values and population stability (PSI > 0.2 flags a shift).")
    for name in ("clean", "shifted"):
        r = mon[name]
        st.markdown(f"**{name.capitalize()} batch** ({r['n']} discharges): "
                    + ("no alerts" if r["ok"] else "ALERT"))
        if not r["ok"]:
            st.json({"shifted features (PSI > 0.2)": {k: round(r["psi"][k], 3) for k in r["shifted_features"]},
                     "unseen categories": r["unseen_categories"], "missing-value rise": r["missing_rise"]})
    st.caption("Shift applied to the second batch: " + "; ".join(mon["shift_applied"]) + ".")

with tab_a:
    st.markdown(f"""
- **Decision supported:** after discharge, order the outreach call list. The score ranks; the calibrated risk is for display.
- **Label:** recorded readmission within 30 days. Only readmissions to the participating hospitals are recorded.
- **Cohort:** discharges excluding deaths, hospice and transfers to another acute hospital.
- **Split:** by patient (train / validation / calibration / test = 60 / 10 / 10 / 20 %). The data has no dates,
  so no time-based split is possible.
- **Model choice:** fixed in advance: keep logistic regression unless gradient boosting improves validation
  average precision by at least {sel['rule'].split('+ ')[-1]}. Chosen: `{sel['chosen']}`. The test set was scored once.
- **Not shown here:** that calling these patients prevents readmissions. A risk score finds who is likely to be
  readmitted, not who would benefit from a call.

Data: Strack B. et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission Rates*, BioMed Research
International; UCI Machine Learning Repository, doi:10.24432/C5230J, CC BY 4.0. Changes: cohort filtering,
derived features, model scores.
""")
