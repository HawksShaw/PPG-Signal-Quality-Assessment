from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from service.processor import QualityChecker

app = FastAPI(
    title="PPG Quality Assessment Service",
    description="Microservice for Quality Checking In-The-Wild PPG signals.",
    version="1.0.0"
)

processor = QualityChecker()

class SignalWindow(BaseModel):
    subject_id: str
    sampling_rate: float
    ppg_ir: List[float]
    acc_x: List[float]
    acc_y: List[float]
    acc_z: List[float]

class QualityResponse(BaseModel):
    status: str
    confidence: float
    reasons: List[str]
    metrics: dict
    metadata: dict

@app.get("/")
def health_check():
    return {"status": "active", "service": "PPG-QA-Pipeline"}

@app.post("/assess", response_model=QualityResponse)
def assess_signal_quality(window: SignalWindow):
    """
    Receives a raw 8-second window of PPG + IMU data.
    Returns the Accept/Reject decision with detailed metrics.
    """
    try:
        expected_len = int(window.sampling_rate * 8) 
        if len(window.ppg_ir) < expected_len * 0.9:
             raise HTTPException(status_code=400, detail="Window too short for analysis")

        result = processor.window_processing(
            ppg_ir=window.ppg_ir,
            acc_x=window.acc_x,
            acc_y=window.acc_y,
            acc_z=window.acc_z,
            fs=window.sampling_rate
        )
        
        return result

    except Exception as e:
        # Log the error internally here
        print(f"Processing Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Note: In production, you run this using: uvicorn sqis.service.api:app --reload