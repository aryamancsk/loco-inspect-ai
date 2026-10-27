import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_LOCK = threading.Lock()
DB_PATH = Path(__file__).resolve().parent / "inspector.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    )
    return dk.hex()


def init_db() -> None:
    with DB_LOCK:
        conn = _get_conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    full_name TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT 'Quality Inspector',
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    category TEXT NOT NULL,
                    operating_mode TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    z_score REAL NOT NULL DEFAULT 0.0,
                    hotspot_pct REAL NOT NULL DEFAULT 0.0,
                    peak_region TEXT NOT NULL DEFAULT '',
                    explanation TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                );
                """
            )
            # Ensure demo user exists
            cur = conn.execute(
                "SELECT id FROM users WHERE username = ? OR email = ?",
                ("demo", "demo@nus.edu.sg"),
            )
            if cur.fetchone() is None:
                salt = secrets.token_hex(16)
                pw_hash = _hash_password("demo1234", salt)
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    INSERT INTO users (username, email, full_name, role, password_hash, salt, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "demo",
                        "demo@nus.edu.sg",
                        "NUS Quality Inspector (Demo)",
                        "Lead Quality Engineer",
                        pw_hash,
                        salt,
                        now,
                    ),
                )
            conn.commit()
        finally:
            conn.close()


def create_session_for_user(conn: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=30)
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, now.isoformat(), expires.isoformat()),
    )
    return token


def register_user(
    username: str = "",
    email: str = "",
    password: str = "",
    full_name: str = "",
    role: str = "Quality Inspector",
) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[str]]:
    username = (username or "").strip()
    email = (email or "").strip().lower()
    password = (password or "").strip()
    full_name = (full_name or "").strip()
    role = (role or "Quality Inspector").strip() or "Quality Inspector"

    if not username and email:
        base_user = email.split("@")[0].strip() or "user"
        username = base_user
    if not email and username:
        if "@" in username:
            email = username.lower()
            username = username.split("@")[0].strip() or "user"
        else:
            email = f"{username.lower()}@local.inspector"

    if not username:
        return False, "Please provide a username or email address.", None, None
    if len(username) < 2:
        return False, "Username must be at least 2 characters long.", None, None
    if not password:
        return False, "Please provide a password.", None, None
    if len(password) < 4:
        return False, "Password must be at least 4 characters long.", None, None

    if not full_name:
        full_name = username

    with DB_LOCK:
        conn = _get_conn()
        try:
            existing_user = conn.execute(
                "SELECT username, email FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)",
                (username, email),
            ).fetchone()
            if existing_user is not None:
                if existing_user["username"].lower() == username.lower():
                    return False, f"Username '{username}' is already taken. Please log in or choose another username.", None, None
                return False, f"Email '{email}' is already registered. Please log in instead.", None, None

            salt = secrets.token_hex(16)
            pw_hash = _hash_password(password, salt)
            now = datetime.now(timezone.utc).isoformat()

            cur = conn.execute(
                """
                INSERT INTO users (username, email, full_name, role, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (username, email, full_name, role, pw_hash, salt, now),
            )
            user_id = cur.lastrowid
            token = create_session_for_user(conn, user_id)
            conn.commit()

            user_dict = {
                "id": user_id,
                "username": username,
                "email": email,
                "full_name": full_name,
                "role": role,
                "created_at": now,
            }
            return True, "Account created successfully!", user_dict, token
        except sqlite3.IntegrityError:
            return False, "An account with that username or email already exists.", None, None
        except Exception as exc:
            return False, f"Registration error: {exc}", None, None
        finally:
            conn.close()


def authenticate_user(
    identifier: str, password: str
) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[str]]:
    identifier = (identifier or "").strip()
    password = (password or "").strip()

    if not identifier:
        return False, "Please enter your username or email address.", None, None
    if not password:
        return False, "Please enter your password.", None, None

    with DB_LOCK:
        conn = _get_conn()
        try:
            row = conn.execute(
                """
                SELECT id, username, email, full_name, role, password_hash, salt, created_at
                FROM users
                WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)
                """,
                (identifier, identifier),
            ).fetchone()

            if row is None:
                return (
                    False,
                    f"No account found for '{identifier}'. Please check your spelling or Sign Up first.",
                    None,
                    None,
                )

            expected_hash = _hash_password(password, row["salt"])
            if not hmac.compare_digest(expected_hash, row["password_hash"]):
                return False, "Incorrect password. Please try again.", None, None

            token = create_session_for_user(conn, row["id"])
            conn.commit()

            user_dict = {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"] or row["username"],
                "role": row["role"],
                "created_at": row["created_at"],
            }
            return True, "Logged in successfully!", user_dict, token
        except Exception as exc:
            return False, f"Login error: {exc}", None, None
        finally:
            conn.close()


def get_user_by_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    token = token.strip()
    if not token:
        return None

    with DB_LOCK:
        conn = _get_conn()
        try:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.email, u.full_name, u.role, u.created_at, s.expires_at
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token = ?
                """,
                (token,),
            ).fetchone()
            if row is None:
                return None

            try:
                expires_at = datetime.fromisoformat(row["expires_at"])
                if expires_at < datetime.now(timezone.utc):
                    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                    conn.commit()
                    return None
            except Exception:
                pass

            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"] or row["username"],
                "role": row["role"],
                "created_at": row["created_at"],
            }
        finally:
            conn.close()


def delete_session(token: Optional[str]) -> None:
    if not token:
        return
    with DB_LOCK:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token.strip(),))
            conn.commit()
        finally:
            conn.close()


def record_inspection(
    user_id: Optional[int],
    category: str,
    operating_mode: str,
    filename: str,
    decision: str,
    severity: str,
    score: float,
    threshold: float,
    z_score: float,
    hotspot_pct: float,
    peak_region: str,
    explanation: str,
) -> int:
    with DB_LOCK:
        conn = _get_conn()
        try:
            now = datetime.now(timezone.utc).isoformat()
            cur = conn.execute(
                """
                INSERT INTO inspections (
                    user_id, category, operating_mode, filename, decision, severity,
                    score, threshold, z_score, hotspot_pct, peak_region, explanation, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    category,
                    operating_mode,
                    filename,
                    decision,
                    severity,
                    float(score),
                    float(threshold),
                    float(z_score),
                    float(hotspot_pct),
                    peak_region,
                    explanation,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def get_inspection_history(user_id: Optional[int] = None, limit: int = 30) -> List[Dict[str, Any]]:
    with DB_LOCK:
        conn = _get_conn()
        try:
            if user_id is not None:
                rows = conn.execute(
                    """
                    SELECT id, category, operating_mode, filename, decision, severity,
                           score, threshold, z_score, hotspot_pct, peak_region, explanation, created_at
                    FROM inspections
                    WHERE user_id = ? OR user_id IS NULL
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, category, operating_mode, filename, decision, severity,
                           score, threshold, z_score, hotspot_pct, peak_region, explanation, created_at
                    FROM inspections
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# Ensure database tables and demo user are always initialized on module load
init_db()

