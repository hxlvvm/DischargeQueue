# DischargeQueue

**Which discharged diabetes patients should an outreach team call first?**

DischargeQueue is a capacity-aware review queue for 30-day hospital readmission. It turns a readmission model into an
operational decision: the team can call *k* % of discharges, so the model's job is to put the right patients in that
*k* %. It covers cohort definition, leakage-safe evaluation, calibration, a FastAPI service with the same feature code
as training, input monitoring, and a Streamlit demo.

## Headline result (held-out test set, scored once)

On **19,537 discharges from 13,759 held-out patients** (2,264 readmissions, base rate 11.6 %):

> Calling the **top 10 %** of discharges reaches **23.5 %** of 30-day readmissions (95 % CI 21.6-25.3 %), against
> **19.8 %** for a baseline built on prior-visit counts. That is a gain of **+3.7 points (2.3-5.1)**, with
> precision 27.3 %, 2.4 times the base rate.

| model | AUROC | average precision | precision @ 10 % | recall @ 10 % |
|---|---|---|---|---|
| Gradient boosting, all discharge-time features (chosen) | 0.665 (0.651-0.678) | 0.222 (0.205-0.242) | 0.273 (0.248-0.299) | 0.235 (0.216-0.253) |
| Baseline: logistic regression on prior outpatient / ER / inpatient visit counts | 0.603 (0.588-0.617) | 0.184 (0.166-0.204) | 0.229 (0.205-0.258) | 0.198 (0.179-0.218) |
| Rule: prior inpatient visits | 0.600 (0.583-0.615) | 0.192 (0.174-0.213) | 0.233 (0.209-0.260) | 0.201 (0.182-0.220) |

95 % intervals come from 1,000 bootstrap resamples of *patients*, with every encounter of a sampled patient kept
together. Gains are computed on the same resamples.

Capacity curve (share of readmissions reached):

| share called | 5 % | 10 % | 20 % | 30 % |
|---|---|---|---|---|
| model | 14.2 % | 23.5 % | 37.9 % | 50.0 % |
| prior-visit baseline | 12.5 % | 19.8 % | 33.4 % | 44.8 % |

**Calibration.** Predicted and observed risk agree within 1.3 points in every decile. The Brier score is 0.0981,
against 0.1025 for predicting the base rate for everyone.

**Where the gain comes from.** On the first encounter of each patient only, the prior-visit baseline drops to
AUROC 0.542 while the model holds 0.647. Prior-visit counts mainly identify repeat patients; the other
discharge-time features add signal for patients the system has not seen before.

## What the result does *not* show

- **That calling these patients prevents readmissions.** The model ranks who is likely to be readmitted, not who
  would benefit from a call. Showing benefit needs an intervention study.
- **Performance today.** The data covers 1999-2008 and has no dates, so a time-based split was impossible. A
  deployment would need a prospective check.
- **Every readmission.** The label is a *recorded* readmission: only readmissions within the participating
  hospitals are visible.
- **Equal performance across ages.** Patients aged 70+ get AUROC 0.632 and 20.5 % of their readmissions reached.
  For patients under 50 the figures are 0.721 and 36.1 %. Other subgroups are reported with intervals in the app;
  groups below 200 discharges are not reported.

## How it works

```
raw CSV row ──► build_features() ──► calibrated model ──► score (ranks the queue)
                     │                                     risk  (calibrated probability, display)
                     │                                     in_review_queue (score ≥ threshold fixed on calibration patients)
                     └─ same function in training, batch scoring and the API
```

- **Cohort.** Exclude discharges to death, hospice, or another acute hospital. 97,215 discharges from 68,797
  patients remain.
- **Label.** `readmitted == "<30"`. Readmissions after 30 days count as negatives.
- **Split.** By patient: train / validation / calibration / test = 60 / 10 / 10 / 20 %.
- **Model choice.** The rule was committed before training: keep the logistic regression unless gradient boosting
  improves validation average precision by ≥ 0.005. Boosting won by 0.009.
- **Calibration and threshold.** Isotonic calibration and the 10 % threshold are both fitted on the calibration
  patients. The queue ranks on the uncalibrated score, so calibration ties cannot reshuffle it.
- **Test set.** Scored once, by `final_eval.py`, which refuses to run a second time. Commit `6f912d2` holds the
  frozen selection and evaluation code.
- **Monitoring.** Labels arrive 30 days late, so the checks run on inputs: schema, unseen categories,
  missing-value rises, and PSI against a training-patient profile.
  - The demo injects an age shift, a lab feed that drops A1C values, and a new admission-source code. All three are
    caught.
  - It also flags payer mix, a real side effect of an older population.
  - It flags one rare legitimate medication value, which shows why unseen-category alerts need a human review step.

More detail on each decision is in [DESIGN.md](DESIGN.md).

## Run it

```bash
pip install -r requirements-dev.txt && pip install -e .
python -c "from dischargequeue.data import download; download()"   # UCI archive, SHA256-checked
python -m dischargequeue.train            # fit, select on validation, calibrate  (~1 min)
python -m dischargequeue.final_eval       # score the test patients once
python -m dischargequeue.monitor reference && python -m dischargequeue.monitor demo
streamlit run app.py                      # demo; reads artifacts/ only
uvicorn dischargequeue.api:app            # POST /score with raw-format discharges
pytest -q                                 # synthetic fixtures, no real rows
```

Batch scoring: `python -m dischargequeue.score discharges.csv scored.csv`.

## Repository

```
src/dischargequeue/  config, data (cohort, split), features, models, train, final_eval, evaluate, score, api, monitor
app.py               Streamlit demo
artifacts/           model, selection and test reports, monitoring profile, held-out scores
tests/               cohort, split, features, API/batch parity, input validation, monitoring, bootstrap
```

## Data and licence

The data are from Strack B., DeShazo J. P., Gennings C., et al. (2014), *Impact of HbA1c Measurement on Hospital
Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*, BioMed Research International. The
dataset is "Diabetes 130-US Hospitals for Years 1999-2008", UCI Machine Learning Repository, doi:10.24432/C5230J,
under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence. Changes made here: cohort filtering,
derived features, and model scores (`artifacts/test_scores.csv.gz`). The raw data are not redistributed; the
download script fetches them.

Code: MIT. This is a research and portfolio project, not a medical device or clinical decision tool.
