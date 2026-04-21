"""
Authentication: doctor signup / login with JWT.
Stores accounts in users.json (flat-file, sufficient for a prototype).
"""

import json, os, hashlib, secrets, datetime
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "data", "users.json")
os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

# ── JWT ────────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET", "lungdiag-secret-change-in-production")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 24

def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE) as f:
        return json.load(f)

def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def signup(email: str, password: str, name: str) -> dict:
    users = _load_users()
    if email in users:
        raise ValueError("Email already registered")
    users[email] = {
        "name": name,
        "password_hash": _hash_password(password),
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    _save_users(users)
    return {"message": "Account created", "email": email, "name": name}

def login(email: str, password: str) -> dict:
    users = _load_users()
    user = users.get(email)
    if not user or user["password_hash"] != _hash_password(password):
        raise ValueError("Invalid email or password")
    token = _create_token(email, user["name"])
    return {"token": token, "email": email, "name": user["name"]}

def _create_token(email: str, name: str) -> str:
    """Create a simple signed JWT."""
    try:
        from jose import jwt
        payload = {
            "sub": email,
            "name": name,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRE_HOURS),
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    except ImportError:
        # Fallback: simple base64 token (not secure, for demo only)
        import base64
        data = json.dumps({"sub": email, "name": name}).encode()
        return base64.b64encode(data).decode()

def verify_token(token: str) -> Optional[dict]:
    """Verify token and return payload, or None if invalid."""
    try:
        from jose import jwt, JWTError
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception:
        return None
