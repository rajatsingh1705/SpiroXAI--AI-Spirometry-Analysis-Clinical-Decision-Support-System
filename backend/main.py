"""
FastAPI main application — SpiroXAI Lung Disease Diagnostic System
Demo mode: No authentication required.
Storage: Supabase (predictions table).
"""

import os
import json
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from models import get_predictor, CLASS_NAMES, FEATURE_NAMES
from validators import validate_prediction_input

app = FastAPI(title="SpiroXAI — Lung Disease Diagnostic API", version="2.0.0")

# ── CORS (allow all for demo) ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase client (lazy singleton) ──────────────────────────────────────────
_supabase = None


def get_supabase():
    global _supabase
    if _supabase is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if url and key:
            try:
                from supabase import create_client
                _supabase = create_client(url, key)
                print("[DB] ✅ Supabase connected")
            except Exception as e:
                print(f"[DB] ⚠ Supabase connection failed: {e}")
        else:
            print("[DB] ⚠ SUPABASE_URL or SUPABASE_KEY not set in .env")
    return _supabase


# ── Pydantic request model ────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    patient_name: Optional[str] = "Unknown"  # ADDED
    Age: float
    Sex: float
    Weight: float
    Height: float
    BMI: float
    Baseline_PEF_Ls: float
    Baseline_FEF2575_Ls: float
    Baseline_Extrapolated_Volume: float
    Baseline_Forced_Expiratory_Time: float
    Baseline_Number_Acceptable_Curves: float
    Race_Black: float = 0.0
    Race_Mexican_American: float = 0.0
    Race_Other_hispanic: float = 0.0
    Race_Other_race_including_multi_racial: float = 0.0
    Race_White: float = 1.0


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — also reports model loading status."""
    predictor = get_predictor()
    if not predictor.loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "Models not loaded. Check saved_models/ directory.",
                "models_loaded": False,
            }
        )
    return {
        "status": "ok",
        "models_loaded": True,
        "classes": CLASS_NAMES,
        "n_features": len(FEATURE_NAMES),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Run ensemble prediction on patient spirometry data.
    Returns predicted class, confidence scores, and SHAP explanations.
    """
    predictor = get_predictor()

    # Build raw input dict with correct column names (spaces in race columns)
    raw = req.dict()
    patient_name = raw.pop("patient_name", "Unknown")  # ADDED: extract before prediction
    raw["Race_Mexican American"] = raw.pop("Race_Mexican_American", 0.0)
    raw["Race_Other hispanic"] = raw.pop("Race_Other_hispanic", 0.0)
    raw["Race_Other race, including multi-racial"] = raw.pop(
        "Race_Other_race_including_multi_racial", 0.0
    )

    # Validate all fields
    errors = validate_prediction_input(raw)
    if errors:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "errors": errors}
        )

    # Run ensemble prediction
    pred_result = predictor.predict(raw)
    pred_idx = pred_result["predicted_class"]
    proba_arr = pred_result["probabilities"]

    proba_dict = {
        CLASS_NAMES[i]: round(float(proba_arr[i]), 4)
        for i in range(len(CLASS_NAMES))
    }
    confidence_pct = round(float(proba_arr[pred_idx]) * 100, 1)

    # SHAP explanation (XGBoost only)
    explanation = predictor.explain(raw, pred_result)

    timestamp = datetime.datetime.utcnow().isoformat()

    response = {
        "status": "success",
        "patient_name": patient_name,  # ADDED
        "prediction": {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence_pct": confidence_pct,
            "probabilities": proba_dict,
            "is_heuristic": pred_result.get("is_heuristic", False),
            "timestamp": timestamp,
        },
        "explanation": explanation,
    }

    # Determine race label for storage
    race_map = {
        "Race_Black": "Black",
        "Race_Mexican American": "Mexican American",
        "Race_Other hispanic": "Other Hispanic",
        "Race_Other race, including multi-racial": "Other / Multi-racial",
        "Race_White": "White",
    }
    race_label = "Unknown"
    for col, label in race_map.items():
        if raw.get(col, 0) == 1:
            race_label = label
            break

    # Save to Supabase
    try:
        sb = get_supabase()
        if sb:
            record = {
                "patient_name": patient_name,  # ADDED
                "patient_age": int(raw.get("Age", 0)),
                "patient_sex": int(raw.get("Sex", 0)),
                "race": race_label,
                "weight": round(float(raw.get("Weight", 0)), 1),
                "height": round(float(raw.get("Height", 0)), 1),
                "bmi": round(float(raw.get("BMI", 0)), 1),
                "pef": round(float(raw.get("Baseline_PEF_Ls", 0)), 2),
                "fef2575": round(float(raw.get("Baseline_FEF2575_Ls", 0)), 2),
                "extrapolated_volume": round(float(raw.get("Baseline_Extrapolated_Volume", 0)), 3),
                "forced_expiratory_time": round(float(raw.get("Baseline_Forced_Expiratory_Time", 0)), 2),
                "acceptable_curves": int(raw.get("Baseline_Number_Acceptable_Curves", 0)),
                "predicted_class": CLASS_NAMES[pred_idx],
                "confidence_normal": round(float(proba_arr[0]), 4),
                "confidence_obstruction": round(float(proba_arr[1]), 4),
                "confidence_restriction": round(float(proba_arr[2]), 4),
                "top_features": explanation.get("top_features", []),
            }
            sb.table("predictions").insert(record).execute()
    except Exception as e:
        print(f"[DB] Save error: {e}")

    return response


@app.get("/records")
def get_records():
    """Fetch past prediction records from Supabase, newest first."""
    try:
        sb = get_supabase()
        if not sb:
            return {"status": "success", "records": []}

        result = (
            sb.table("predictions")
            .select("*")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        records = result.data if result.data else []

        return {"status": "success", "records": records}
    except Exception as e:
        print(f"[DB] Fetch error: {e}")
        return {"status": "success", "records": []}


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
