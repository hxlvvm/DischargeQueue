"""Paths and fixed decisions. Every number here is written down before the test set is scored."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "diabetic_data.csv"
ARTIFACTS = ROOT / "artifacts"

# UCI "Diabetes 130-US Hospitals for Years 1999-2008", CC BY 4.0, doi:10.24432/C5230J
DATA_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
DATA_SHA256 = "f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a"

SEED = 20261008

# Discharges that cannot enter a post-discharge outreach queue:
# 11, 19, 20, 21 expired; 13, 14 hospice; 2 transferred to another short-term (acute) hospital.
EXCLUDED_DISPOSITIONS = (2, 11, 13, 14, 19, 20, 21)

# Patient-level split fractions (all encounters of a patient land in the same part).
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.1, "calib": 0.1, "test": 0.2}

# Operating point: the outreach team can review this fraction of discharges.
REVIEW_FRACTION = 0.10

# Model choice on the validation split: keep the interpretable logistic regression unless the
# gradient-boosted model's average precision is higher by at least this margin.
GBM_MIN_AP_GAIN = 0.005

N_BOOTSTRAP = 1000
