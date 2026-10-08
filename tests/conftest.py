"""Synthetic rows in the raw table's format. Invented values only; no real records are used in tests."""
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from dischargequeue.features import MEDS, build_features
from dischargequeue.models import full_gbm, utilisation_lr

AGES = ["[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]


def synthetic_raw(n=600, seed=0):
    r = np.random.default_rng(seed)
    pick = lambda vals, p=None: r.choice(np.array(vals, dtype=object), n, p=p)  # noqa: E731
    df = pd.DataFrame({
        "encounter_id": np.arange(n) + 1000, "patient_nbr": r.integers(0, n // 2, n),
        "race": pick(["Caucasian", "AfricanAmerican", "Hispanic", None]),
        "gender": pick(["Female", "Male", "Unknown/Invalid"], [0.5, 0.49, 0.01]),
        "age": pick(AGES), "weight": None,
        "admission_type_id": r.integers(1, 4, n), "discharge_disposition_id": pick([1, 1, 1, 3, 6, 11, 13, 2, 7]),
        "admission_source_id": r.integers(1, 8, n), "time_in_hospital": r.integers(1, 14, n),
        "payer_code": pick(["MC", "HM", None]), "medical_specialty": pick(["InternalMedicine", "Cardiology", None]),
        "num_lab_procedures": r.integers(1, 90, n), "num_procedures": r.integers(0, 6, n),
        "num_medications": r.integers(1, 40, n), "number_outpatient": r.poisson(0.4, n),
        "number_emergency": r.poisson(0.2, n), "number_inpatient": r.poisson(0.6, n),
        "diag_1": pick(["428", "250.83", "786", "V57", "414", None]), "diag_2": pick(["250", "401", "E888", "427"]),
        "diag_3": pick(["276", "250.01", None]), "number_diagnoses": r.integers(1, 10, n),
        "max_glu_serum": pick([None, ">200", "Norm"]), "A1Cresult": pick([None, ">8", "Norm"]),
        "change": pick(["No", "Ch"]), "diabetesMed": pick(["Yes", "No"]),
    })
    for m in MEDS:
        df[m] = pick(["No", "Steady", "Up", "Down"], [0.7, 0.2, 0.05, 0.05])
    # the label depends on prior inpatient visits, so a model has something to learn
    p = 1 / (1 + np.exp(-(-2.2 + 0.6 * df["number_inpatient"])))
    df["readmitted"] = np.where(r.random(n) < p, "<30", r.choice(["NO", ">30"], n))
    return df


@pytest.fixture(scope="session")
def raw():
    return synthetic_raw()


@pytest.fixture(scope="session")
def artifact(raw):
    X, y = build_features(raw), (raw["readmitted"] == "<30").astype(int).to_numpy()
    model = full_gbm().fit(X, y)
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(X, y)
    s = model.predict_proba(X)[:, 1]
    return {"model": model, "calibrator": cal, "baseline": utilisation_lr().fit(X, y), "chosen": "full_gbm",
            "threshold": float(np.quantile(s, 0.9))}
