# Design decisions

Each modelling decision below was fixed before the test set was scored. The commit history shows the order:

1. `8160aef` pre-registered the cohort, label, split, candidates and selection rule.
2. `6f912d2` fixed the capacity threshold and added the one-shot test evaluation.
3. Only after that was the test set scored.

## The decision being supported

After discharge, an outreach team calls patients to check medications, follow-up appointments and warning signs.
It can reach only a fraction of discharges. The model therefore answers "who goes on today's call list", which
makes **precision and recall at a fixed review capacity** the primary metrics. AUROC is secondary, because it
averages over cut-offs the team will never use.

## Scoring moment and features

The score is produced at discharge. Every feature is known by then:
- demographics and admission type / source;
- length of stay, lab and procedure counts, medications and medication changes;
- the three recorded diagnoses, grouped as in Strack et al. 2014;
- glucose / A1C results;
- prior-year outpatient, ER and inpatient visit counts.

Dropped:
- `weight`, which is 97 % missing;
- identifiers;
- the label.

## Cohort

Excluded discharges:
- to death (codes 11, 19, 20, 21);
- to hospice (13, 14);
- to another acute hospital (2), because the care episode continues elsewhere and nobody is discharged to the
  outreach team.

Discharges against medical advice (7) are **kept**. They are a high-risk group the team should see. Unknown and
unmapped disposition codes stay as their own category.

## Label

`readmitted == "<30"` is positive. Readmissions after more than 30 days and no readmission are both negative. The
source only records readmissions inside the participating hospitals, so this is a *recorded* readmission.

## Split

The data are split **by patient**, so one patient's encounters never fall on both sides; 23 % of patients have
more than one encounter. The fractions are train 60 %, validation 10 %, calibration 10 % and test 20 %.

There are no dates, so a time-based split is impossible. Encounter-id order is not used as a stand-in for time.

## Baselines

- **Learned baseline:** logistic regression on prior-year visit counts.
- **Rule:** prior inpatient visits, with ties broken by ER visits and then a seeded random draw. The tie-break
  matters: most patients have zero prior visits, so the 10 % cut falls inside one large tie group.

Why not LACE? It needs a Charlson comorbidity score, and three diagnosis codes cannot supply one.

**Ties in the learned baseline.** It uses three visit counts, so it produces only 335 distinct scores on the test
set, and the 10 % cut falls inside a tie group of 141 discharges. The reported numbers break ties by row order.
The tie-aware expected value counts the boundary group fractionally. At 10 % it gives recall 19.85 % (reported
19.79 %) and precision 23.0 % (reported 22.9 %), so the headline gain is unchanged. At 30 % the boundary group
holds 2,573 discharges and the expected baseline recall is 44.5 % (reported 44.8 %). The model's scores have no
material ties.

## Model choice

Candidates: L2 logistic regression and histogram gradient boosting.

Gradient boosting's early stopping holds out a random 10 % of *training rows*, so one training patient's
encounters can sit on both sides of that internal split. This affects only when boosting stops. The test set is
patient-disjoint, so the reported numbers are not affected.

The rule, written in `config.py` before training: keep the logistic regression, which is easier to explain, unless
gradient boosting improves validation average precision by at least 0.005. Boosting improved it by 0.009, so it
was chosen.

## Calibration and the queue

- **Calibration.** Isotonic, fitted on the calibration patients.
- **Ranking.** The queue ranks on the *uncalibrated* score. Isotonic output is a step function, so ranking on it
  would create ties that change top-k membership.
- **Threshold.** The "top 10 %" score threshold the API uses is also fixed on the calibration patients.

## Evaluation

- **One shot.** `final_eval.py` scores the test patients once and refuses to run again.
- **Intervals.** They come from a patient-level bootstrap (1,000 resamples). Model-minus-baseline gains are
  computed on the same resamples.
- **Robustness.** One row restricts the test set to the first encounter per patient, which matches the analysis
  in Strack et al.
- **Subgroups.** Groups below 200 discharges or 20 readmissions are not reported.

## Serving

- **Parity.** The API (FastAPI + Pydantic) accepts discharges in the raw table's format and calls the same
  `build_features` as training. A test checks that API scores equal batch scores. On 50 real rows the maximum
  difference is 0.0.
- **Validation.** Invalid inputs, such as an impossible age band or a negative count, return HTTP 422.

## Monitoring

Readmission labels arrive 30 days late, so the routine checks run on inputs against a profile of the training
patients:
- schema;
- unseen categories;
- rises in missing values;
- PSI per feature, where above 0.2 counts as shifted.

Performance monitoring with labels is shown only as the held-out backtest. No live system is claimed.
