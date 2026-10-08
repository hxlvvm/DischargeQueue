"""HTTP scoring service. It accepts discharges in the raw table's column format and runs the same
feature code as training and batch scoring.

    uvicorn dischargequeue.api:app
"""
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

from .score import load_artifact, score_frame

MedStatus = Literal["No", "Steady", "Up", "Down"]
_MED_FIELDS = {m: (MedStatus, "No") for m in [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
    "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]}


class _Core(BaseModel):
    race: Optional[str] = None
    gender: Optional[Literal["Female", "Male"]] = None
    age: Literal["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)",
                 "[80-90)", "[90-100)"]
    admission_type_id: int = Field(ge=1)
    discharge_disposition_id: int = Field(ge=1)
    admission_source_id: int = Field(ge=1)
    time_in_hospital: int = Field(ge=1, le=30)
    payer_code: Optional[str] = None
    medical_specialty: Optional[str] = None
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=0)
    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)
    diag_1: Optional[str] = None
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None
    number_diagnoses: int = Field(ge=0)
    max_glu_serum: Optional[Literal[">200", ">300", "Norm"]] = None
    A1Cresult: Optional[Literal[">7", ">8", "Norm"]] = None
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["No", "Yes"] = "No"


Discharge = create_model("Discharge", __base__=_Core, **_MED_FIELDS)


class Scored(BaseModel):
    score: float = Field(description="ranking score; the queue is ordered by this")
    risk: float = Field(description="calibrated probability of a recorded 30-day readmission")
    in_review_queue: bool = Field(description="score is in the top review fraction (threshold fixed on calibration patients)")


app = FastAPI(title="DischargeQueue", version="0.1.0")
_ART = None


def artifact():
    global _ART
    if _ART is None:
        _ART = load_artifact()
    return _ART


@app.get("/health")
def health():
    a = artifact()
    return {"status": "ok", "model": a["chosen"], "threshold": a["threshold"]}


@app.post("/score", response_model=list[Scored])
def score(discharges: list[Discharge]):
    df = pd.DataFrame([d.model_dump() for d in discharges])
    out = score_frame(df, artifact())
    return out[["score", "risk", "in_review_queue"]].to_dict(orient="records")
