"""
Patient history storage: saves/loads prediction records per doctor.
"""

import json, os, uuid, datetime
from typing import List, Optional

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "data", "history.json")
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

def _load() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE) as f:
        return json.load(f)

def _save(data: dict):
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def add_record(doctor_email: str, patient_id: str, patient_name: str, patient_input: dict, prediction: dict) -> str:
    """Save one prediction record; return its ID."""
    data = _load()
    if doctor_email not in data:
        data[doctor_email] = []
    record_id = str(uuid.uuid4())
    record = {
        "id": record_id,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "patient_input": patient_input,
        "prediction": prediction,
    }
    data[doctor_email].append(record)
    _save(data)
    return record_id

def get_records(doctor_email: str) -> List[dict]:
    """Return all prediction records for a doctor (newest first)."""
    data = _load()
    records = data.get(doctor_email, [])
    return list(reversed(records))

def get_record(doctor_email: str, record_id: str) -> Optional[dict]:
    """Return a single record by ID."""
    for r in get_records(doctor_email):
        if r["id"] == record_id:
            return r
    return None

def delete_record(doctor_email: str, record_id: str) -> bool:
    """Delete a single record; return True if successfully found and removed."""
    data = _load()
    if doctor_email not in data:
        return False
    
    initial_len = len(data[doctor_email])
    data[doctor_email] = [r for r in data[doctor_email] if r["id"] != record_id]
    
    if len(data[doctor_email]) < initial_len:
        _save(data)
        return True
    return False
