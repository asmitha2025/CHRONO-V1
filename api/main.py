from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import os
import json
from engine.personal_baseline import PersonalBaseline
from engine.wiv_calculator import WIVCalculator
from engine.bav_calculator import BAVCalculator
from engine.icv_calculator import ICVCalculator
from engine.mcf_scorer import MCFScorer

app = FastAPI(title="CHRONO API", version="1.0.0")

# Enable CORS for the demo dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock database of patients (in a real app, this would be a DB)
PATIENTS = {
    "priya_001": {
        "name": "Priya",
        "birth_year": 1984,
        "birth_month": 3,
        "history": [
            ("2018-01-10", {"ldh": 165, "albumin": 4.4, "glucose": 92, "rdw": 12.8, "creatinine": 0.82, "crp": 0.8, "lymphocyte_pct": 34, "mcv": 88, "alp": 62, "wbc": 5.8, "neutrophils": 3.8, "lymphocytes": 1.9, "platelets": 240}),
            ("2020-05-15", {"ldh": 172, "albumin": 4.2, "glucose": 95, "rdw": 13.0, "creatinine": 0.85, "crp": 1.2, "lymphocyte_pct": 31, "mcv": 89, "alp": 65, "wbc": 6.2, "neutrophils": 4.1, "lymphocytes": 1.8, "platelets": 260}),
            ("2022-11-20", {"ldh": 185, "albumin": 4.1, "glucose": 98, "rdw": 13.5, "creatinine": 0.88, "crp": 2.1, "lymphocyte_pct": 29, "mcv": 90, "alp": 72, "wbc": 6.8, "neutrophils": 4.6, "lymphocytes": 1.7, "platelets": 280}),
            ("2024-01-18", {"ldh": 214, "albumin": 3.95, "glucose": 101, "rdw": 14.2, "creatinine": 0.89, "crp": 3.8, "lymphocyte_pct": 26.5, "mcv": 91, "alp": 81, "wbc": 7.8, "neutrophils": 5.4, "lymphocytes": 1.73, "platelets": 290}),
        ]
    }
}

@app.get("/api/v1/patient/{patient_id}/mcf")
async def get_patient_mcf(patient_id: str):
    if patient_id not in PATIENTS:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    pdata = PATIENTS[patient_id]
    baseline = PersonalBaseline(patient_id, pdata["birth_year"], pdata["birth_month"])
    
    # Build baseline
    for dt, markers in pdata["history"]:
        baseline.add_test(dt, markers)
    
    # Compute signals
    wiv_calc = WIVCalculator(baseline)
    bav_calc = BAVCalculator(baseline)
    icv_calc = ICVCalculator(baseline)
    mcf_scorer = MCFScorer()
    
    wiv = wiv_calc.compute(pdata["history"])
    bav = bav_calc.compute(pdata["history"])
    icv = icv_calc.compute(pdata["history"])
    
    mcf = mcf_scorer.compute(wiv, bav, icv)
    
    res = mcf.to_dict()
    # Add extra fields for dashboard compatibility
    res["band"] = res["alert_level"]
    res["confidence"] = mcf.confidence
    res["components"] = {
        "wiv": {"normalized": wiv.velocity / wiv.max_velocity, "raw": wiv.velocity},
        "bav": {"normalized": bav.velocity / bav.max_velocity, "raw": bav.velocity},
        "icv": {"normalized": icv.velocity / icv.max_velocity, "raw": icv.velocity},
    }
    return res

@app.get("/api/v1/patient/{patient_id}/history")
async def get_patient_history(patient_id: str, marker: str = "ldh"):
    if patient_id not in PATIENTS:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    pdata = PATIENTS[patient_id]
    history = []
    for dt, markers in pdata["history"]:
        if marker in markers:
            history.append({"date": dt, "value": markers[marker]})
            
    return {"patient_id": patient_id, "marker": marker, "history": history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
