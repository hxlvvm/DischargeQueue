"""Load the raw table, define the cohort and the label, and split by patient."""
import hashlib
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from .config import DATA_SHA256, DATA_URL, EXCLUDED_DISPOSITIONS, RAW_CSV, SEED, SPLIT_FRACTIONS


def download(dest=RAW_CSV):
    """Download the UCI archive, check its SHA256 and extract the two CSV files."""
    blob = urllib.request.urlopen(DATA_URL, timeout=120).read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(f"unexpected archive checksum {digest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(blob)).extractall(dest.parent)
    return dest


def load_raw(path=RAW_CSV):
    """The raw table; '?' marks a missing value in this dataset."""
    return pd.read_csv(path, na_values="?", low_memory=False)


def build_cohort(raw):
    """Discharges eligible for outreach, with the label.

    Label: recorded readmission within 30 days (readmitted == "<30"). Readmissions after 30 days
    and no readmission are both negative. The data only sees readmissions inside the
    participating hospitals, so this is a recorded readmission, not every readmission.
    """
    df = raw[~raw["discharge_disposition_id"].isin(EXCLUDED_DISPOSITIONS)].copy()
    df["gender"] = df["gender"].where(df["gender"].isin(["Female", "Male"]))
    df["y"] = (df["readmitted"] == "<30").astype(int)
    return df.reset_index(drop=True)


def assign_split(df, seed=SEED):
    """Split by patient: every encounter of one patient lands in the same part.

    The data has no dates, so a time-based split is not possible; encounter_id order is not used
    as a proxy for time.
    """
    patients = np.sort(df["patient_nbr"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    names = list(SPLIT_FRACTIONS)
    edges = np.cumsum([0] + [SPLIT_FRACTIONS[n] for n in names])
    cut = np.round(edges * len(patients)).astype(int)
    part = {}
    for name, a, b in zip(names, cut[:-1], cut[1:]):
        part.update(dict.fromkeys(patients[a:b], name))
    out = df.copy()
    out["split"] = out["patient_nbr"].map(part)
    return out
