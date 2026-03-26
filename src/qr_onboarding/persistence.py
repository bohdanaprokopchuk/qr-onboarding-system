from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class PersistentDeviceRecord:
    registration_id: str
    registration_token: str
    wifi: dict[str, Any]
    metadata: dict[str, Any]


class SQLiteBootstrapStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS devices (
                    registration_id TEXT PRIMARY KEY,
                    registration_token TEXT NOT NULL,
                    wifi_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_token TEXT PRIMARY KEY,
                    registration_id TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS acknowledgements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_token TEXT NOT NULL,
                    registration_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    ts REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS consents (
                    consent_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    issued_at REAL NOT NULL,
                    verified_at REAL
                );
            """)

    def add_device(self, record: PersistentDeviceRecord) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO devices(registration_id, registration_token, wifi_json, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)", (record.registration_id, record.registration_token, json.dumps(record.wifi, ensure_ascii=False), json.dumps(record.metadata, ensure_ascii=False), now))

    def get_device(self, registration_id: str) -> Optional[PersistentDeviceRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM devices WHERE registration_id = ?", (registration_id,)).fetchone()
        if row is None:
            return None
        return PersistentDeviceRecord(row['registration_id'], row['registration_token'], json.loads(row['wifi_json']), json.loads(row['metadata_json']))

    def save_session(self, session_token: str, registration_id: str, expires_at: float) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO sessions(session_token, registration_id, expires_at, created_at, consumed) VALUES (?, ?, ?, ?, COALESCE((SELECT consumed FROM sessions WHERE session_token = ?), 0))", (session_token, registration_id, expires_at, now, session_token))

    def get_session(self, session_token: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_token = ?", (session_token,)).fetchone()
        return None if row is None else dict(row)

    def add_ack(self, session_token: str, registration_id: str, device_id: str, ts: float) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO acknowledgements(session_token, registration_id, device_id, ts) VALUES (?, ?, ?, ?)", (session_token, registration_id, device_id, ts))

    def list_acks(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT session_token, registration_id, device_id, ts FROM acknowledgements ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]

    def save_consent(self, consent_id: str, payload: dict[str, Any], verified: bool = False) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO consents(consent_id, payload_json, issued_at, verified_at) VALUES (?, ?, COALESCE((SELECT issued_at FROM consents WHERE consent_id = ?), ?), ?)", (consent_id, json.dumps(payload, ensure_ascii=False), consent_id, time.time(), time.time() if verified else None))

    def mark_consent_verified(self, consent_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE consents SET verified_at = ? WHERE consent_id = ?", (time.time(), consent_id))

    def get_consent(self, consent_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM consents WHERE consent_id = ?", (consent_id,)).fetchone()
        if row is None:
            return None
        return {'consent_id': row['consent_id'], 'payload': json.loads(row['payload_json']), 'issued_at': row['issued_at'], 'verified_at': row['verified_at']}
