"""
api.py

FastAPI wrapper around predict.py's predict(), exposed via /predict.

Run locally:
    uvicorn api:app --reload --port 8000

Then open http://127.0.0.1:8000/docs, or POST directly:
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{...}'
"""

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import predict

app = FastAPI(
    title="Women's Hormonal Health -- Ensemble Model API",
    description="Combines a male-skewed baseline, a women-only model, and "
                 "rule-based clinical criteria into a joint PCOS x thyroid "
                 "dysfunction prediction. See predict.py for the full "
                 "routing/heuristic documentation.",
    version="0.1.0",
)


class PatientInput(BaseModel):
    # Model 2 / 2b / 3 fields (PCOS-cohort)
    age: Optional[float] = None
    bmi: Optional[float] = None
    cycle_regularity: Optional[int] = Field(None, description="2=regular, 4=irregular")
    cycle_length_days: Optional[float] = None
    weight_gain: Optional[int] = Field(None, description="0=No, 1=Yes")
    hirsutism: Optional[int] = Field(None, description="0=No, 1=Yes")
    skin_darkening: Optional[int] = Field(None, description="0=No, 1=Yes")
    hair_loss: Optional[int] = Field(None, description="0=No, 1=Yes")
    acne: Optional[int] = Field(None, description="0=No, 1=Yes")
    fast_food: Optional[int] = Field(None, description="0=No, 1=Yes")
    regular_exercise: Optional[int] = Field(None, description="0=No, 1=Yes")
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    fsh: Optional[float] = Field(None, description="mIU/mL")
    lh: Optional[float] = Field(None, description="mIU/mL")
    fsh_lh_ratio: Optional[float] = None
    amh: Optional[float] = Field(None, description="ng/mL")
    prl: Optional[float] = Field(None, description="ng/mL")
    vit_d3: Optional[float] = Field(None, description="ng/mL")
    progesterone: Optional[float] = Field(None, description="ng/mL")
    random_blood_sugar: Optional[float] = Field(None, description="mg/dl")
    follicle_count_left: Optional[int] = None
    follicle_count_right: Optional[int] = None
    avg_follicle_size_left: Optional[float] = Field(None, description="mm")
    avg_follicle_size_right: Optional[float] = Field(None, description="mm")
    endometrium_mm: Optional[float] = None
    tsh: Optional[float] = Field(None, description="mIU/L -- used by Models 2/2b/3")

    # Model 1 fields (full thyroid panel)
    sex: Optional[str] = Field(None, description="'male' or 'female'")
    pregnant: Optional[int] = Field(None, description="0=No, 1=Yes")
    sick: Optional[int] = Field(None, description="0=No, 1=Yes")
    t3: Optional[float] = None
    t4: Optional[float] = None
    t4_uptake: Optional[float] = None
    free_thyroxine_index: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def run_prediction(patient: PatientInput):
    patient_dict = patient.model_dump()
    return predict(patient_dict)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)