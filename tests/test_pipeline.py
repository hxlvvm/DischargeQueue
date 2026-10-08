import numpy as np
import pytest
from fastapi.testclient import TestClient

import dischargequeue.api as api
from dischargequeue.data import assign_split, build_cohort
from dischargequeue.evaluate import cluster_bootstrap, top_k_mask
from dischargequeue.features import RAW_INPUT, build_features, icd9_group
from dischargequeue.monitor import build_reference, check_batch
from dischargequeue.score import score_frame


def test_cohort_excludes_deaths_hospice_transfers(raw):
    c = build_cohort(raw)
    assert not c["discharge_disposition_id"].isin([2, 11, 13, 14, 19, 20, 21]).any()
    assert set(c["y"]) <= {0, 1}
    assert (c["y"] == (c["readmitted"] == "<30")).all()
    assert set(c["gender"].dropna()) <= {"Female", "Male"}


def test_split_is_patient_disjoint_and_deterministic(raw):
    a, b = assign_split(build_cohort(raw)), assign_split(build_cohort(raw))
    assert a.groupby("patient_nbr")["split"].nunique().max() == 1
    assert (a["split"] == b["split"]).all()
    assert set(a["split"]) == {"train", "val", "calib", "test"}


@pytest.mark.parametrize("code,group", [("250.83", "diabetes"), ("428", "circulatory"), ("786", "respiratory"),
                                        ("V57", "other"), ("E888", "other"), (None, "missing"), ("820", "injury")])
def test_icd9_groups(code, group):
    assert icd9_group(code) == group


def test_features_never_see_label_or_ids(raw):
    X = build_features(raw)
    assert not {"readmitted", "y", "encounter_id", "patient_nbr", "weight"} & set(X.columns)
    assert X.select_dtypes("number").isna().sum().sum() == 0


def test_api_matches_batch_scoring(raw, artifact, monkeypatch):
    monkeypatch.setattr(api, "_ART", artifact)
    rows = raw[RAW_INPUT].head(25)
    payload = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
               for r in rows.astype(object).to_dict(orient="records")]
    for p in payload:
        p["gender"] = p["gender"] if p["gender"] in ("Female", "Male") else None
    resp = TestClient(api.app).post("/score", json=payload)
    assert resp.status_code == 200, resp.text
    batch = rows.copy()
    batch["gender"] = batch["gender"].where(batch["gender"].isin(["Female", "Male"]))
    expected = score_frame(batch, artifact)
    got = np.array([r["score"] for r in resp.json()])
    np.testing.assert_allclose(got, expected["score"].to_numpy(), rtol=1e-9)


def test_api_rejects_invalid_input(artifact, monkeypatch):
    monkeypatch.setattr(api, "_ART", artifact)
    bad = {"age": "[200-210)", "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 1,
           "time_in_hospital": 3, "num_lab_procedures": 1, "num_procedures": 0, "num_medications": 1,
           "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0, "number_diagnoses": 1}
    assert TestClient(api.app).post("/score", json=[bad]).status_code == 422


def test_monitor_flags_unseen_category_and_missingness(raw):
    prof = build_reference(raw)
    assert check_batch(raw, prof)["shifted_features"] == []
    shifted = raw.copy()
    shifted["admission_source_id"] = 99
    shifted["A1Cresult"] = None
    rep = check_batch(shifted, prof)
    assert "admission_source_id" in rep["unseen_categories"]
    assert "A1Cresult" in rep["missing_rise"]
    assert not rep["ok"]


def test_top_k_and_cluster_bootstrap_keep_patients_whole():
    s = np.arange(100, dtype=float)
    assert top_k_mask(s, 0.1).sum() == 10 and top_k_mask(s, 0.1)[-10:].all()
    groups = np.repeat(np.arange(20), 5)
    ci = cluster_bootstrap(groups, 50, 0, lambda idx: {"whole": float(np.all(np.bincount(groups[idx]) % 5 == 0))})
    assert ci["whole"] == (1.0, 1.0)
