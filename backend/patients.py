import sqlite3, os, uuid, datetime, json
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "data", "spiro_patients.db")
OLD_JSON_FILE = os.path.join(BASE_DIR, "data", "patients.json")
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize table
with get_db_connection() as conn:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            doctor_email TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            sex INTEGER NOT NULL,
            age INTEGER NOT NULL,
            height REAL NOT NULL,
            weight REAL NOT NULL,
            bmi REAL NOT NULL,
            race TEXT NOT NULL
        )
    ''')
    conn.commit()

# Run Migration from old JSON data if it exists
if os.path.exists(OLD_JSON_FILE) and not os.path.exists(OLD_JSON_FILE + ".migrated"):
    try:
        with open(OLD_JSON_FILE) as f:
            data = json.load(f)
        with get_db_connection() as conn:
            for doctor_email, patients in data.items():
                for p in patients:
                    conn.execute('''
                        INSERT OR IGNORE INTO patients (id, doctor_email, name, created_at, sex, age, height, weight, bmi, race)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (p["id"], doctor_email, p["name"], p.get("created_at", ""), p.get("sex", 1), p.get("age", 0), float(p.get("height", 0)), float(p.get("weight", 0)), float(p.get("bmi", 0)), p.get("race", "")))
            conn.commit()
        os.rename(OLD_JSON_FILE, OLD_JSON_FILE + ".migrated")
        print("Migrated old patients.json to SQLite db.")
    except Exception as e:
        print("Could not migrate legacy patients JSON:", e)

# --- Core API ---
def add_patient(doctor_email: str, name: str, sex: int, age: int, height: float, weight: float, race: str) -> dict:
    patient_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()
    bmi = round(weight / ((height / 100) ** 2), 1)
    
    with get_db_connection() as conn:
        conn.execute('''
            INSERT INTO patients (id, doctor_email, name, created_at, sex, age, height, weight, bmi, race)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (patient_id, doctor_email, name, created_at, sex, age, height, weight, bmi, race))
        conn.commit()
        
    return {
        "id": patient_id,
        "name": name,
        "created_at": created_at,
        "sex": sex,
        "age": age,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "race": race
    }

def get_patients(doctor_email: str) -> List[dict]:
    with get_db_connection() as conn:
        rows = conn.execute('SELECT * FROM patients WHERE doctor_email = ? ORDER BY created_at DESC', (doctor_email,)).fetchall()
        return [dict(r) for r in rows]

def get_patient(doctor_email: str, patient_id: str) -> Optional[dict]:
    with get_db_connection() as conn:
        row = conn.execute('SELECT * FROM patients WHERE doctor_email = ? AND id = ?', (doctor_email, patient_id)).fetchone()
        return dict(row) if row else None

def delete_patient(doctor_email: str, patient_id: str) -> bool:
    with get_db_connection() as conn:
        cur = conn.execute('DELETE FROM patients WHERE doctor_email = ? AND id = ?', (doctor_email, patient_id))
        conn.commit()
        return cur.rowcount > 0
