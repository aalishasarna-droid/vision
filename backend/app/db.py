"""
MongoDB Database Service for Storing Acoustic Anomaly Detection Records
and Predictive Maintenance Fault Prognostics.

Schema:
- _id: ObjectId (standard MongoDB internal unique identifier)
- machineId: str (e.g. 'p001', 'PUMP-LINE1-01', 'v001', 'f001')
- machineType: str ('pump', 'valve', or 'fan')
- status: str ('normal' or 'anomaly')
- anomalyScore: float (numeric reconstruction error produced by the model)
- defectBand: str (primary frequency defect band)
- notes: str (engineer observation / maintenance notes)
- timestamp: datetime (UTC date and time of detection)
"""

import os
import re
import threading
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

logger = logging.getLogger("vision.db")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "vision_db")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "detections")

_client: Optional[MongoClient] = None
_db = None
_collection = None
_id_lock = threading.Lock()

PREFIX_MAP = {
    "pump": "p",
    "valve": "v",
    "fan": "f",
    "motor": "m",
}


def get_mongo_collection():
    """
    Returns the MongoDB collection singleton with lazy initialization and index creation.
    """
    global _client, _db, _collection
    if _collection is not None:
        return _collection

    try:
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
            socketTimeoutMS=3000,
        )
        _db = _client[MONGODB_DB_NAME]
        _collection = _db[MONGODB_COLLECTION_NAME]

        # Ensure indexes for performant query filters
        _collection.create_index([("machineId", ASCENDING), ("timestamp", DESCENDING)])
        _collection.create_index([("timestamp", DESCENDING)])
        _collection.create_index([("machineType", ASCENDING)])
        _collection.create_index([("status", ASCENDING)])

        logger.info(f"Connected to MongoDB at {MONGODB_URI}, database: {MONGODB_DB_NAME}, collection: {MONGODB_COLLECTION_NAME}")
        return _collection
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB ({MONGODB_URI}): {e}")
        return None


def normalize_machine_type(machine_type: Optional[str]) -> str:
    """
    Normalizes machineType to one of: pump, valve, fan, motor.
    """
    if not machine_type:
        return "pump"
    m = machine_type.lower().strip()
    if m in ("pump", "valve", "fan", "motor"):
        return m
    return "pump"


def resolve_machine_id(machine_type: str, filename: Optional[str] = None, explicit_id: Optional[str] = None) -> str:
    """
    Resolves standard machineId string (e.g. p001, v001, f001, m001).
    Uses explicit_id if provided; otherwise derives from machine type and filename.
    """
    if explicit_id and explicit_id.strip():
        return explicit_id.strip()

    m_type = normalize_machine_type(machine_type)
    prefix = PREFIX_MAP.get(m_type, "p")

    # If filename contains an ID number (e.g., id_00, id_02, id_04, id_06)
    if filename:
        match = re.search(r"id_(\d+)", filename.lower())
        if match:
            id_num = int(match.group(1)) + 1
            return f"{prefix}{id_num:03d}"

    return f"{prefix}001"


def generate_unique_machine_id(machine_type: str = "pump", explicit_id: Optional[str] = None) -> str:
    """
    Generates a new sequential machine asset ID if none is provided.
    """
    if explicit_id and explicit_id.strip():
        return explicit_id.strip()

    m_type = normalize_machine_type(machine_type)
    prefix = PREFIX_MAP.get(m_type, "p")

    with _id_lock:
        col = get_mongo_collection()
        max_num = 0
        if col is not None:
            try:
                distinct_ids = col.distinct("machineId")
                for mid in distinct_ids:
                    match = re.match(f"^{prefix}(\\d+)$", str(mid))
                    if match:
                        num = int(match.group(1))
                        if num > max_num:
                            max_num = num
            except Exception as e:
                logger.warning(f"Error querying distinct machineId: {e}")

        return f"{prefix}{max_num + 1:03d}"


def record_detection_event(
    anomaly_score: float,
    machine_type: str = "pump",
    is_anomaly: bool = False,
    machine_id: Optional[str] = None,
    filename: Optional[str] = None,
    defect_band: Optional[str] = None,
    notes: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Stores a detection event in MongoDB conforming to the required schema:
      - machineId: str (e.g. p001, PUMP-LINE1-01)
      - machineType: pump, valve, fan, or motor
      - status: normal or anomaly
      - anomalyScore: numeric anomaly score produced by the model
      - defectBand: primary acoustic frequency band defect
      - notes: optional operator/engineer notes
      - timestamp: UTC datetime
    Note: Omits custom _id so MongoDB uses standard native ObjectId, avoiding duplicate _id String columns.
    """
    norm_machine_type = normalize_machine_type(machine_type)
    resolved_machine_id = resolve_machine_id(norm_machine_type, filename=filename, explicit_id=machine_id)
    status_str = "anomaly" if is_anomaly else "normal"
    utc_timestamp = timestamp or datetime.now(timezone.utc)

    detection_doc = {
        "machineId": str(resolved_machine_id),
        "machineType": str(norm_machine_type),
        "status": str(status_str),
        "anomalyScore": float(round(float(anomaly_score), 6)),
        "defectBand": str(defect_band or ("Mid-Frequency (2-5 kHz)" if is_anomaly else "Nominal Spectrum")),
        "notes": str(notes or ""),
        "timestamp": utc_timestamp,
    }

    try:
        col = get_mongo_collection()
        if col is not None:
            res = col.insert_one(detection_doc)
            detection_doc["_id"] = str(res.inserted_id)
            logger.info(f"Stored detection record in MongoDB: id={res.inserted_id}, machineId={resolved_machine_id}, status={status_str}, score={anomaly_score}")
        else:
            logger.warning("MongoDB collection unavailable; detection event processed without persistence.")
            detection_doc["_id"] = "offline_local"
    except (PyMongoError, ServerSelectionTimeoutError, Exception) as e:
        logger.error(f"Failed to persist detection record to MongoDB: {e}")
        detection_doc["_id"] = "offline_error"

    return detection_doc


def update_detection_record(
    record_id: str,
    machine_id: str,
    machine_type: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Updates the machineId and notes on an existing detection record
    (used when an engineer reassigns an anomaly to an existing or new machine ID).
    """
    col = get_mongo_collection()
    if col is None or not record_id:
        return None

    update_fields: Dict[str, Any] = {"machineId": machine_id.strip()}
    if machine_type:
        update_fields["machineType"] = normalize_machine_type(machine_type)
    if notes is not None:
        update_fields["notes"] = notes.strip()

    try:
        query = {"_id": ObjectId(record_id)} if ObjectId.is_valid(record_id) else {"_id": record_id}
        col.update_one(query, {"$set": update_fields})
        updated = col.find_one(query)
        if updated:
            updated["_id"] = str(updated["_id"])
            if isinstance(updated.get("timestamp"), datetime):
                updated["timestamp"] = updated["timestamp"].isoformat()
            return updated
    except Exception as e:
        logger.error(f"Failed to update detection record {record_id}: {e}")

    return None


def get_registered_machines(machine_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns all distinct machines registered in MongoDB with historical stats.
    """
    col = get_mongo_collection()
    if col is None:
        return []

    try:
        match_stage = {}
        if machine_type:
            match_stage = {"machineType": normalize_machine_type(machine_type)}

        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {"$sort": {"timestamp": -1}},
            {
                "$group": {
                    "_id": "$machineId",
                    "machineId": {"$first": "$machineId"},
                    "machineType": {"$first": "$machineType"},
                    "total_inspections": {"$sum": 1},
                    "anomaly_count": {
                        "$sum": {"$cond": [{"$eq": ["$status", "anomaly"]}, 1, 0]}
                    },
                    "latest_status": {"$first": "$status"},
                    "latest_score": {"$first": "$anomalyScore"},
                    "latest_defect_band": {"$first": "$defectBand"},
                    "latest_timestamp": {"$first": "$timestamp"},
                }
            },
            {"$sort": {"latest_timestamp": -1}},
        ]

        machines = list(col.aggregate(pipeline))
        results = []
        for m in machines:
            t = m.get("latest_timestamp")
            t_str = t.isoformat() if isinstance(t, datetime) else str(t or "")
            results.append({
                "machineId": m["machineId"],
                "machineType": m.get("machineType", "pump"),
                "total_inspections": m.get("total_inspections", 0),
                "anomaly_count": m.get("anomaly_count", 0),
                "latest_status": m.get("latest_status", "normal"),
                "latest_score": float(m.get("latest_score", 0.0)),
                "latest_defect_band": m.get("latest_defect_band", "Nominal"),
                "latest_timestamp": t_str,
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching registered machines: {e}")
        return []


def get_machine_anomaly_history(machine_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves full history of test records for a specific machine ID.
    """
    col = get_mongo_collection()
    if col is None:
        return []

    try:
        cursor = col.find({"machineId": machine_id.strip()}).sort("timestamp", DESCENDING)
        records = []
        for doc in cursor:
            t = doc.get("timestamp")
            records.append({
                "recordId": str(doc["_id"]),
                "machineId": doc.get("machineId", machine_id),
                "machineType": doc.get("machineType", "pump"),
                "status": doc.get("status", "normal"),
                "anomalyScore": float(doc.get("anomalyScore", 0.0)),
                "defectBand": doc.get("defectBand", "Nominal"),
                "notes": doc.get("notes", ""),
                "timestamp": t.isoformat() if isinstance(t, datetime) else str(t or ""),
            })
        return records
    except Exception as e:
        logger.error(f"Error fetching history for machine {machine_id}: {e}")
        return []


def analyze_machine_prognostics(machine_id: str) -> Dict[str, Any]:
    """
    Analyses the pattern, frequency of occurrence, and cause of faults
    for a particular individual machine, and predicts what will probably occur.
    """
    col = get_mongo_collection()
    if col is None:
        return {
            "machineId": machine_id,
            "status": "OFFLINE",
            "message": "MongoDB offline",
        }

    records = list(col.find({"machineId": machine_id.strip()}).sort("timestamp", ASCENDING))
    if not records:
        return {
            "machineId": machine_id,
            "machineType": "pump",
            "total_inspections": 0,
            "anomaly_count": 0,
            "anomaly_rate_pct": 0.0,
            "risk_level": "HEALTHY",
            "score_trend": "NO_DATA",
            "frequency_assessment": "No prior inspections recorded for this asset ID.",
            "dominant_defect_band": "None",
            "likely_failure_mode": "Asset operating without recorded history.",
            "estimated_time_to_failure": "N/A",
            "consequences": "Routine baseline monitoring required.",
            "preventive_actions": ["Conduct initial baseline acoustic inspection."],
            "history": [],
        }

    machine_type = records[-1].get("machineType", "pump").lower()
    anomalies = [r for r in records if r.get("status") == "anomaly"]
    total_inspections = len(records)
    anomaly_count = len(anomalies)
    anomaly_rate_pct = round((anomaly_count / total_inspections) * 100, 1)

    scores = [float(r.get("anomalyScore", 0.0)) for r in records]
    latest_score = scores[-1]
    peak_score = max(scores) if scores else 0.0

    # 1. Pattern & Score Progression Trend
    if len(scores) >= 2:
        recent_delta = scores[-1] - scores[-2]
        if scores[-1] >= 0.035 and recent_delta > 0.003:
            score_trend = "ESCALATING (High Urgency)"
        elif recent_delta > 0.002:
            score_trend = "INCREASING (Degradation Active)"
        elif abs(recent_delta) <= 0.002 and anomaly_count > 0:
            score_trend = "STABLE ELEVATED (Persistent Fault)"
        elif recent_delta < -0.003:
            score_trend = "IMPROVING (Post-Maintenance Recovery)"
        else:
            score_trend = "NOMINAL (Stable Baseline)"
    else:
        score_trend = "SINGLE EVENT (Initial Observation)" if anomaly_count > 0 else "NOMINAL (Single Baseline)"

    # 2. Frequency of Occurrence Analysis
    if anomaly_count >= 2:
        anomaly_times = [r["timestamp"] for r in anomalies if isinstance(r.get("timestamp"), datetime)]
        if len(anomaly_times) >= 2:
            intervals_hours = [
                (anomaly_times[i] - anomaly_times[i - 1]).total_seconds() / 3600.0
                for i in range(1, len(anomaly_times))
            ]
            avg_interval = sum(intervals_hours) / len(intervals_hours)
            latest_interval = intervals_hours[-1]
            if latest_interval < avg_interval * 0.7:
                freq_desc = f"ACCELERATING FREQUENCY: Recurrence interval shrank to {latest_interval:.1f} hrs (historical avg: {avg_interval:.1f} hrs) across {anomaly_count} recorded anomalies."
            else:
                freq_desc = f"RECURRENT ANOMALIES: {anomaly_count} anomaly events recorded with an average recurrence interval of {avg_interval:.1f} hrs."
        else:
            freq_desc = f"RECURRENT ANOMALIES: {anomaly_count} anomaly events recorded in asset history."
    elif anomaly_count == 1:
        freq_desc = "ISOLATED ANOMALY: First recorded anomaly threshold breach. Monitor closely for recurrence."
    else:
        freq_desc = "NOMINAL OPERATION: Zero anomalies detected across all historical tests."

    # 3. Cause of Fault & Dominant Frequency Defect Band
    defect_bands = [r.get("defectBand") for r in anomalies if r.get("defectBand") and r.get("defectBand") != "Nominal Spectrum"]
    if defect_bands:
        dominant_defect = max(set(defect_bands), key=defect_bands.count)
    else:
        dominant_defect = "Mid-Frequency Cavitation / Resonance" if anomaly_count > 0 else "Nominal Spectrum"

    # 4. Predictive Prognosis ("What can probably occur in this machine")
    if anomaly_count == 0:
        risk_level = "HEALTHY"
        likely_failure = "Nominal Operation — No Imminent Failure Predicted"
        time_to_failure = "Normal Lifecycle (> 2,500 Operating Hours)"
        consequences = "Asset is vibrating and sounding within healthy acoustic limits."
        actions = ["Continue regular scheduled vibration & acoustic logging."]
    else:
        if machine_type == "pump":
            if "Mid" in dominant_defect or "Cavitation" in dominant_defect:
                likely_failure = "Impeller Cavitation Pitting & Impending Mechanical Seal Rupture"
                time_to_failure = "24–48 Operating Hours" if score_trend.startswith("ESCALATING") else "72–120 Operating Hours"
                consequences = "Erosive pitting on impeller vanes leading to severe hydraulic unbalance, mechanical seal face burning, toxic fluid leakage, and complete drive shaft lockup."
                actions = [
                    "Inspect suction strainers and intake piping for NPSH starvation.",
                    "Verify pump operating point against Best Efficiency Point (BEP) curve.",
                    "Check mechanical seal flush and quench lines for flow obstruction.",
                    "Prepare backup pump unit for immediate switchover.",
                ]
            elif "Low" in dominant_defect or "Bearing" in dominant_defect or "Looseness" in dominant_defect:
                likely_failure = "Drive-End Bearing Race Spalling & Shaft Dynamic Misalignment"
                time_to_failure = "48–96 Operating Hours"
                consequences = "Bearing cage breakdown leading to rotor-stator contact, thermal expansion, motor stator winding short, and catastrophic motor burnout."
                actions = [
                    "Perform laser shaft alignment and check coupling condition.",
                    "Extract lubricant sample to test for metallic wear debris / moisture.",
                    "Check hold-down bolts for soft-foot and foundation resonance.",
                ]
            else:
                likely_failure = "Internal High-Pressure Fluid Recirculation & Casing Fatigue"
                time_to_failure = "36–72 Operating Hours"
                consequences = "High acoustic energy causing casing micro-cracking and internal check-valve chatter."
                actions = [
                    "Inspect discharge check valve for backflow leakage.",
                    "Ultrasonic thickness check on discharge volute.",
                ]

        elif machine_type == "valve":
            if "High" in dominant_defect or "Leak" in dominant_defect:
                likely_failure = "Internal Valve Seat Wire-Drawing & High-Pressure Blow-by"
                time_to_failure = "12–36 Operating Hours"
                consequences = "Seat erosion preventing 100% positive isolation, leading to process contamination, dangerous pressure buildup, and emergency relief valve lift."
                actions = [
                    "Perform acoustic leak pin-point inspection on valve body.",
                    "Cycle valve actuator through full 0-100% stroke to dislodge debris.",
                    "Schedule trim and seat replacement during next scheduled outage.",
                ]
            elif "Mid" in dominant_defect or "Flutter" in dominant_defect:
                likely_failure = "Valve Plug Hydrodynamic Flutter & Stem Galling"
                time_to_failure = "48–72 Operating Hours"
                consequences = "Vibration fatigues stem guide bushings, causing the valve to jam in mid-travel."
                actions = [
                    "Check positioner tuning parameters and supply air pressure.",
                    "Inspect packing gland for over-tightening or uneven torque.",
                ]
            else:
                likely_failure = "Pneumatic Actuator Diaphragm Hunting & Flange Looseness"
                time_to_failure = "3–5 Operating Days"
                consequences = "Control instability and process variable cycling."
                actions = [
                    "Inspect I/P converter calibration.",
                    "Check pneumatic supply tubing for pinhole leaks.",
                ]

        else:  # fan
            if "Low" in dominant_defect or "Unbalance" in dominant_defect:
                likely_failure = "Impeller Blade Mass Unbalance & Catastrophic Structural Fatigue"
                time_to_failure = "24–48 Operating Hours" if score_trend.startswith("ESCALATING") else "3–5 Operating Days"
                consequences = "Excessive 1X rotational vibration causing blade root fatigue cracking, catastrophic blade throw, destruction of housing, and extreme safety hazard."
                actions = [
                    "Lock-out tag-out and inspect fan wheel for particulate buildup or missing weights.",
                    "Perform dynamic field trim balancing.",
                    "Check pillow block bearing pedestal bolts and anti-vibration dampers.",
                ]
            elif "Mid" in dominant_defect:
                likely_failure = "Blade Passing Aerodynamic Resonance & Stator Vane Rubbing"
                time_to_failure = "48–96 Operating Hours"
                consequences = "Thermal friction at casing contact points and aerodynamic ducting separation."
                actions = [
                    "Check radial tip clearances around entire perimeter.",
                    "Inspect expansion joints and flexible duct transitions.",
                ]
            else:
                likely_failure = "Bearings Lubricant Starvation & Raceway Thermal Micro-Welding"
                time_to_failure = "24–48 Operating Hours"
                consequences = "Seized bearing resulting in belt slippage, smoke, and drive motor overload trip."
                actions = [
                    "Regrease bearings using OEM specified high-speed polyurea grease.",
                    "Check bearing housing operating temperature with IR pyrometer.",
                ]

        # Overall Risk Level
        if latest_score >= 0.045 or (anomaly_count >= 3 and score_trend.startswith("ESCALATING")):
            risk_level = "CRITICAL"
        elif latest_score >= 0.035 or anomaly_count >= 2:
            risk_level = "ELEVATED"
        else:
            risk_level = "MODERATE"

    history_output = []
    for r in reversed(records):
        t = r.get("timestamp")
        t_str = t.isoformat() if isinstance(t, datetime) else str(t or "")
        history_output.append({
            "recordId": str(r.get("_id", "")),
            "machineId": str(r.get("machineId", "")),
            "machineType": str(r.get("machineType", "")),
            "status": str(r.get("status", "")),
            "anomalyScore": float(r.get("anomalyScore", 0.0)),
            "defectBand": str(r.get("defectBand", "Broadband")),
            "notes": str(r.get("notes", "")),
            "timestamp": t_str,
        })

    return {
        "machineId": machine_id,
        "machineType": machine_type,
        "total_inspections": total_inspections,
        "anomaly_count": anomaly_count,
        "anomaly_rate_pct": anomaly_rate_pct,
        "risk_level": risk_level,
        "score_trend": score_trend,
        "latest_score": latest_score,
        "peak_score": peak_score,
        "frequency_assessment": freq_desc,
        "dominant_defect_band": dominant_defect,
        "likely_failure_mode": likely_failure,
        "estimated_time_to_failure": time_to_failure,
        "consequences": consequences,
        "preventive_actions": actions,
        "history": history_output,
    }


def get_recent_detections(
    limit: int = 50,
    machine_type: Optional[str] = None,
    machine_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves recent detection records sorted by timestamp descending.
    """
    col = get_mongo_collection()
    if col is None:
        return []

    query: Dict[str, Any] = {}
    if machine_type:
        query["machineType"] = normalize_machine_type(machine_type)
    if machine_id:
        query["machineId"] = machine_id.strip()
    if status:
        query["status"] = status.lower().strip()

    try:
        cursor = col.find(query).sort("timestamp", DESCENDING).limit(limit)
        records = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            if isinstance(doc.get("timestamp"), datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
            records.append(doc)
        return records
    except Exception as e:
        logger.error(f"Error querying detections from MongoDB: {e}")
        return []
