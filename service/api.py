from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from service.processor import QualityChecker
from service.storage import save_assessment
from service.configs import logger

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

class BatchResponse(BaseModel):
    results: List[QualityResponse]
    processed_count: int

@app.get("/")
def health_check():
    return {"status": "active", "service": "PPG-QA-Pipeline"}

@app.post("/assess", response_model=QualityResponse)
def assess_signal_quality(window: SignalWindow):
    logger.info(f"STREAM Request for Subject: {window.subject_id}")
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

        raw_data = {
            "ppg_ir": window.ppg_ir,
            "acc_x" : window.acc_x,
            "acc_y" : window.acc_y,
            "acc_z" : window.acc_z,
            "fs"    : window.sampling_rate
        }

        record_id = save_assessment(
            metadata = {"subject_id" : window.subject_id},
            report = result,
            raw_signals = raw_data
        )
        
        logger.info(f"ID: {record_id} | Status: {result['status']} | Confidence Score: {result['confidence']:.2f}")

        result["metadata"]["storage_id"] = record_id
        return result

    except Exception as e:
        # Log the error internally here
        logger.error(f"Failed processing Subject {window.subject_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess/batch", response_model=BatchResponse)
def assess_batch_quality(windows: List[SignalWindow]):

    logger.info(f"BATCH Request for {len(windows)} Windows.")

    result = []

    for i, window in enumerate(windows):
        try:
            processed_data = processor.window_processing(
                ppg_ir = window.ppg_ir,
                acc_x = window.acc_x,
                acc_y = window.acc_y,
                acc_z = window.acc_z,
                fs = window.sampling_rate
            )

            raw_data = {
                "ppg_ir" : window.ppg_ir,
                "acc_x"  : window.acc_x,
                "acc_y"  : window.acc_y,
                "acc_z"  : window.acc_z,
                "fs"     : window.sampling_rate
            }

            record_id = save_assessment(
                metadata = {"subject_id" : window.subject_id},
                report = processed_data,
                raw_signals = raw_data 
            )

            processed_data["metadata"]["storage_id"] = record_id
            result.append(processed_data)

        except Exception as e:
            logger.error(f"Batch {i} failed with exception: {e}")
            continue

    logger.info(f"Batch finished. Processed {len(result)}/{len(windows)} windows successfully.")

    return {
        "results" : result,
        "processed_count" : len(result)
    }
# Run this using: uvicorn sqis.service.api:app --reload