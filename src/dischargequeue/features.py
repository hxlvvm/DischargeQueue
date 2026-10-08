"""Features available at the moment of discharge, built the same way for training, batch scoring
and the API."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
        "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
        "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone"]

# Raw columns a caller must send (CSV-shaped, as in the source table).
RAW_INPUT = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id", "admission_source_id",
             "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures", "num_procedures",
             "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "diag_1",
             "diag_2", "diag_3", "number_diagnoses", "max_glu_serum", "A1Cresult", "change",
             "diabetesMed"] + MEDS

NUMERIC = ["age_years", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
           "n_med_changes"]
CATEGORICAL = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
               "payer_code", "medical_specialty", "diag_1_group", "diag_2_group", "diag_3_group",
               "max_glu_serum", "A1Cresult", "change", "diabetesMed"] + MEDS
UTILISATION = ["number_outpatient", "number_emergency", "number_inpatient"]


def icd9_group(code):
    """Map an ICD-9 code to the broad groups used by Strack et al. (2014)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "missing"
    s = str(code)
    if s[0] in "VE":
        return "other"
    x = float(s)
    if 250 <= x < 251:
        return "diabetes"
    if 390 <= x <= 459 or int(x) == 785:
        return "circulatory"
    if 460 <= x <= 519 or int(x) == 786:
        return "respiratory"
    if 520 <= x <= 579 or int(x) == 787:
        return "digestive"
    if 800 <= x <= 999:
        return "injury"
    if 710 <= x <= 739:
        return "musculoskeletal"
    if 580 <= x <= 629 or int(x) == 788:
        return "genitourinary"
    if 140 <= x <= 239:
        return "neoplasms"
    return "other"


def build_features(df):
    """Raw CSV-shaped rows -> model input frame. Never reads the label or any identifier."""
    X = df[RAW_INPUT].copy()
    # age arrives as a 10-year band such as "[70-80)"; use the band's midpoint
    X["age_years"] = X["age"].str.extract(r"\[(\d+)-")[0].astype(float) + 5
    for k in (1, 2, 3):
        X[f"diag_{k}_group"] = X[f"diag_{k}"].map(icd9_group)
    X["n_med_changes"] = X[MEDS].isin(["Up", "Down"]).sum(axis=1)
    for c in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        X[c] = X[c].astype("Int64").astype(str)
    for c in CATEGORICAL:
        X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
    return X[NUMERIC + CATEGORICAL]


def make_preprocessor(scale=True):
    """Numeric: median impute (+ standardise for linear models). Categorical: one-hot; categories
    seen fewer than 50 times and unseen ones share an 'infrequent' column."""
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()) if scale else \
        SimpleImputer(strategy="median")
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False)
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
