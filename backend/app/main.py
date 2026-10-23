"""
FastAPI Backend Server for Industrial Acoustic Anomaly Detection.
Provides REST endpoints for audio ingestion, inference, and model profiling.
"""

import os
import glob
import time
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import soundfile as sf
import numpy as np

load_dotenv()

from app.audio import AudioProcessor
from app.inference import AnomalyDetector
from app.profiler import ModelProfiler
from app.llm_advisor import generate_anomaly_advice
from app.db import (
    record_detection_event,
    get_recent_detections,
    get_mongo_collection,
    get_registered_machines,
    get_machine_anomaly_history,
    analyze_machine_prognostics,
    update_detection_record,
)

app = FastAPI(
    title="Vision - Industrial Acoustic Anomaly Detection API",
    description="Backend microservice serving Librosa feature extraction, Autoencoder inference, and edge profiling.",
    version="1.0.0",
)

# Enable CORS for React + Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
audio_processor = AudioProcessor(sample_rate=16000, n_mels=64, target_frames=64)
detector = AnomalyDetector()
profiler = ModelProfiler()


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok", "service": "vision-acoustic-ai"}


@app.get("/api/profile")
def get_model_profile() -> Dict[str, Any]:
    """Returns memory footprint, quantization format, and latency benchmark."""
    return profiler.get_full_profile()


@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    machine_type: Optional[str] = Form(None),
    machine_id: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Accepts an uploaded .wav audio file and machine_type (pump/valve/fan), extracts Log-Mel Spectrogram,
    computes Autoencoder reconstruction error, stores the anomaly score in MongoDB, and returns anomaly status.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Only audio files (.wav, .flac, .mp3) are supported.")

    start_time = time.perf_counter()

    try:
        audio_bytes = await file.read()
        waveform, sr = audio_processor.load_audio_from_bytes(audio_bytes)
        spectrogram = audio_processor.compute_log_mel_spectrogram(waveform)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio with Librosa: {str(e)}")

    # Run inference
    try:
        prediction = detector.predict(spectrogram, waveform=waveform, filename=file.filename, machine_type=machine_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Store detection record in MongoDB conforming to required schema
    detection_record = record_detection_event(
        anomaly_score=prediction["score"],
        machine_type=machine_type or "pump",
        is_anomaly=prediction["is_anomaly"],
        machine_id=machine_id,
        filename=file.filename,
        defect_band=prediction.get("primary_defect_band", "Mid-Frequency"),
    )

    return {
        "filename": file.filename,
        "sample_rate": sr,
        "duration_sec": round(len(waveform) / sr, 2),
        "score": prediction["score"],
        "threshold": prediction["threshold"],
        "fixed_threshold": prediction.get("fixed_threshold", 0.020),
        "is_anomaly": prediction["is_anomaly"],
        "status": prediction["status"],
        "decision_mode": prediction.get("decision_mode", "DYNAMIC_STATISTICAL"),
        "dynamic_metrics": prediction.get("dynamic_metrics", {}),
        "frequency_bands": prediction.get("frequency_bands", []),
        "primary_defect_band": prediction.get("primary_defect_band", "None"),
        "latency_ms": total_latency_ms,
        "spectrogram": spectrogram.tolist(),
        "reconstructed_spectrogram": prediction["reconstructed_spectrogram"],
        "detection_record": {
            "machineId": detection_record["machineId"],
            "machineType": detection_record["machineType"],
            "status": detection_record["status"],
            "anomalyScore": detection_record["anomalyScore"],
            "timestamp": detection_record["timestamp"].isoformat() if hasattr(detection_record["timestamp"], "isoformat") else str(detection_record["timestamp"]),
            "recordId": str(detection_record.get("_id", "")),
        },
    }


@app.get("/api/baselines")
def get_machine_baselines() -> Dict[str, Any]:
    """
    Returns pre-calibrated and adaptive statistical baselines (mu, sigma, thresholds)
    for industrial machine assets.
    """
    from backend.app.inference import MACHINE_BASELINES
    return {
        "active_machine": "PUMP",
        "baselines": MACHINE_BASELINES,
        "ema_adaptive_baselines": detector.ema_baseline,
    }


@app.get("/api/sample-analyze")
def analyze_preset_sample(
    sample_type: str = Query("normal", pattern="^(normal|anomaly)$"),
    filename: Optional[str] = None,
    machine_type: Optional[str] = Query(None),
    machine_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Directly tests inference on real recorded dataset samples based on machine type,
    stores the anomaly score in MongoDB, and returns detection results.
    """
    start_time = time.perf_counter()
    target_machine = (machine_type or "pump").lower()

    sample_path = None
    if filename:
        candidate_paths = [
            os.path.join(f"data/{target_machine}/train", filename),
            os.path.join(f"data/{target_machine}/test", filename),
            os.path.join("data/pump/train", filename),
            os.path.join("data/pump/test", filename),
            os.path.join("data/valve/train", filename),
            os.path.join("data/valve/test", filename),
            os.path.join("data/fan/train", filename),
            os.path.join("data/fan/test", filename),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                sample_path = p
                break

    if not sample_path:
        subfolder = "train" if sample_type == "normal" else "test"
        prefix = "normal_" if sample_type == "normal" else "anomaly_"
        candidates = glob.glob(f"data/{target_machine}/{subfolder}/{prefix}*.wav")
        if not candidates:
            if sample_type == "normal":
                candidates = (
                    glob.glob("data/pump/train/normal_*.wav")
                    + glob.glob("data/valve/train/normal_*.wav")
                    + glob.glob("data/fan/train/normal_*.wav")
                )
            else:
                candidates = (
                    glob.glob("data/pump/test/anomaly_*.wav")
                    + glob.glob("data/valve/test/anomaly_*.wav")
                    + glob.glob("data/fan/test/anomaly_*.wav")
                )

        if not candidates:
            raise HTTPException(status_code=404, detail=f"No {sample_type} audio clips found in data directories")
        sample_path = candidates[0]

    waveform, sr = sf.read(sample_path)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    spectrogram = audio_processor.compute_log_mel_spectrogram(waveform)
    prediction = detector.predict(spectrogram, waveform=waveform, filename=sample_path, machine_type=target_machine)
    total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Store detection record in MongoDB conforming to required schema
    detection_record = record_detection_event(
        anomaly_score=prediction["score"],
        machine_type=target_machine,
        is_anomaly=prediction["is_anomaly"],
        machine_id=machine_id,
        filename=os.path.basename(sample_path),
        defect_band=prediction.get("primary_defect_band", "Mid-Frequency"),
    )

    return {
        "filename": os.path.basename(sample_path),
        "sample_rate": sr,
        "duration_sec": round(len(waveform) / sr, 2),
        "score": prediction["score"],
        "threshold": prediction["threshold"],
        "fixed_threshold": prediction.get("fixed_threshold", 0.020),
        "is_anomaly": prediction["is_anomaly"],
        "status": prediction["status"],
        "decision_mode": prediction.get("decision_mode", "DYNAMIC_STATISTICAL"),
        "dynamic_metrics": prediction.get("dynamic_metrics", {}),
        "frequency_bands": prediction.get("frequency_bands", []),
        "primary_defect_band": prediction.get("primary_defect_band", "None"),
        "latency_ms": total_latency_ms,
        "spectrogram": spectrogram.tolist(),
        "reconstructed_spectrogram": prediction["reconstructed_spectrogram"],
        "detection_record": {
            "machineId": detection_record["machineId"],
            "machineType": detection_record["machineType"],
            "status": detection_record["status"],
            "anomalyScore": detection_record["anomalyScore"],
            "timestamp": detection_record["timestamp"].isoformat() if hasattr(detection_record["timestamp"], "isoformat") else str(detection_record["timestamp"]),
            "recordId": str(detection_record.get("_id", "")),
        },
    }


@app.get("/api/detections")
def list_detections(
    limit: int = Query(50, ge=1, le=500),
    machine_type: Optional[str] = Query(None),
    machine_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None, pattern="^(normal|anomaly)$"),
) -> Dict[str, Any]:
    """
    Retrieves stored anomaly detection records from MongoDB.
    """
    records = get_recent_detections(
        limit=limit,
        machine_type=machine_type,
        machine_id=machine_id,
        status=status,
    )
    return {
        "count": len(records),
        "records": records,
    }


@app.get("/api/detections/stats")
def get_detections_stats() -> Dict[str, Any]:
    """
    Returns high-level statistics of stored detections from MongoDB.
    """
    col = get_mongo_collection()
    if col is None:
        return {"status": "mongodb_unavailable", "total": 0, "normal": 0, "anomaly": 0}

    try:
        total = col.count_documents({})
        normal_count = col.count_documents({"status": "normal"})
        anomaly_count = col.count_documents({"status": "anomaly"})
        return {
            "status": "connected",
            "total_detections": total,
            "normal_detections": normal_count,
            "anomaly_detections": anomaly_count,
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "total": 0}


class ReassignAnomalyRequest(BaseModel):
    record_id: str
    machine_id: str
    machine_type: Optional[str] = "pump"
    notes: Optional[str] = ""


@app.get("/api/machines")
def list_machines(machine_type: Optional[str] = Query(None)) -> Dict[str, Any]:
    """
    Returns all registered machines with inspection & anomaly counts and current health status.
    """
    machines = get_registered_machines(machine_type=machine_type)
    return {"count": len(machines), "machines": machines}


@app.get("/api/machines/{machine_id}/history")
def get_machine_history(machine_id: str) -> Dict[str, Any]:
    """
    Retrieves chronological inspection & anomaly test history for a specific machine ID.
    """
    history = get_machine_anomaly_history(machine_id=machine_id)
    return {"machineId": machine_id, "count": len(history), "history": history}


@app.get("/api/machines/{machine_id}/prognostics")
def get_machine_prognostics(machine_id: str) -> Dict[str, Any]:
    """
    Analyses the pattern, frequency of occurrence, and cause of faults
    for a particular individual machine, and predicts what will probably occur.
    """
    return analyze_machine_prognostics(machine_id=machine_id)


@app.post("/api/machines/reassign-anomaly")
def reassign_anomaly(req: ReassignAnomalyRequest) -> Dict[str, Any]:
    """
    Updates or re-assigns an anomaly detection record to an existing or new machine ID,
    and returns updated prognostic analysis.
    """
    updated = update_detection_record(
        record_id=req.record_id,
        machine_id=req.machine_id,
        machine_type=req.machine_type,
        notes=req.notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Detection record not found or update failed")

    prognostics = analyze_machine_prognostics(machine_id=req.machine_id)
    return {
        "status": "success",
        "record": updated,
        "prognostics": prognostics,
    }


class LLMAdviceRequest(BaseModel):
    machine_type: str = "pump"
    score: float = 0.02
    threshold: float = 0.0201
    z_score: float = 0.0
    is_anomaly: Optional[bool] = None
    primary_defect: str = "Uniformly Nominal"
    frequency_bands: Optional[List[Dict[str, Any]]] = None
    api_key: Optional[str] = None


@app.post("/api/llm-advice")
def get_llm_anomaly_advice(req: LLMAdviceRequest) -> Dict[str, Any]:
    """
    Invokes Gemini LLM to generate root causes, solutions, and preventive measures
    for detected machine acoustic anomalies.
    """
    return generate_anomaly_advice(
        machine_type=req.machine_type,
        score=req.score,
        threshold=req.threshold,
        z_score=req.z_score,
        primary_defect=req.primary_defect,
        frequency_bands=req.frequency_bands or [],
        api_key_override=req.api_key,
        is_anomaly=req.is_anomaly,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
