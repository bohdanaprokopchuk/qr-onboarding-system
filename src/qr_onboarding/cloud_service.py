from __future__ import annotations

import secrets, time
from dataclasses import dataclass, field
from typing import Dict, Optional
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from .persistence import PersistentDeviceRecord, SQLiteBootstrapStore

@dataclass
class DeviceRecord:
    registration_id: str
    registration_token: str
    wifi: dict
    metadata: dict = field(default_factory=dict)

class BootstrapSessionRequest(BaseModel): registration_id: str; registration_token: str
class RegisterDeviceRequest(BaseModel): registration_id: str; registration_token: str; wifi: dict; metadata: dict = {}
class AckRequest(BaseModel): device_id: str

class InMemoryCloudBootstrapStore:
    def __init__(self): self.devices: Dict[str, DeviceRecord] = {}; self.acks=[]
    def add_device(self, record: DeviceRecord) -> None: self.devices[record.registration_id] = record
    def validate_device(self, registration_id: str, registration_token: str) -> DeviceRecord:
        record = self.devices.get(registration_id)
        if not record or record.registration_token != registration_token: raise HTTPException(status_code=401, detail='Invalid registration credentials')
        return record
    def save_session(self, session_token: str, registration_id: str, expires_at: float) -> None: return None
    def session_known(self, session_token: str) -> bool: return True
    def add_ack(self, session_token: str, registration_id: str, device_id: str, ts: float) -> None: self.acks.append({'session_token': session_token, 'registration_id': registration_id, 'device_id': device_id, 'ts': ts})
    def list_acks(self) -> list[dict]: return list(self.acks)

class PersistentCloudBootstrapStore:
    def __init__(self, db_path: str): self.sqlite = SQLiteBootstrapStore(db_path)
    def add_device(self, record: DeviceRecord) -> None: self.sqlite.add_device(PersistentDeviceRecord(record.registration_id, record.registration_token, record.wifi, record.metadata))
    def validate_device(self, registration_id: str, registration_token: str) -> DeviceRecord:
        record = self.sqlite.get_device(registration_id)
        if not record or record.registration_token != registration_token: raise HTTPException(status_code=401, detail='Invalid registration credentials')
        return DeviceRecord(record.registration_id, record.registration_token, record.wifi, record.metadata)
    def save_session(self, session_token: str, registration_id: str, expires_at: float) -> None: self.sqlite.save_session(session_token, registration_id, expires_at)
    def session_known(self, session_token: str) -> bool: return self.sqlite.get_session(session_token) is not None
    def add_ack(self, session_token: str, registration_id: str, device_id: str, ts: float) -> None: self.sqlite.add_ack(session_token, registration_id, device_id, ts)
    def list_acks(self) -> list[dict]: return self.sqlite.list_acks()

class CloudBootstrapService:
    def __init__(self, secret: Optional[str] = None, ttl_seconds: int = 300, store: Optional[object] = None): self.secret = secret or secrets.token_hex(32); self.ttl_seconds = ttl_seconds; self.store = store or InMemoryCloudBootstrapStore()
    def register_device(self, registration_id: str, registration_token: str, wifi: dict, metadata: Optional[dict] = None) -> dict:
        self.store.add_device(DeviceRecord(registration_id, registration_token, wifi, metadata or {})); return {'ok': True, 'registration_id': registration_id}
    def issue_token(self, record: DeviceRecord):
        exp = time.time() + self.ttl_seconds
        token = jwt.encode({'jti': secrets.token_hex(8), 'registration_id': record.registration_id, 'wifi': record.wifi, 'metadata': record.metadata, 'exp': exp}, self.secret, algorithm='HS256')
        self.store.save_session(token, record.registration_id, exp)
        return token, exp
    def decode_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret, algorithms=['HS256'])
            if hasattr(self.store, 'session_known') and not self.store.session_known(token): raise HTTPException(status_code=401, detail='Unknown session token')
            return payload
        except jwt.ExpiredSignatureError as exc:
            raise HTTPException(status_code=401, detail='Session token expired') from exc
        except HTTPException: raise
        except Exception as exc:
            raise HTTPException(status_code=401, detail='Invalid session token') from exc

def build_cloud_bootstrap_app(service: Optional[CloudBootstrapService] = None) -> FastAPI:
    service = service or CloudBootstrapService(); app = FastAPI(title='QR Bootstrap Service', version='3.0'); app.state.service = service
    def _bearer(authorization: str = Header(default='')) -> str:
        if not authorization.startswith('Bearer '): raise HTTPException(status_code=401, detail='Missing bearer token')
        return authorization.split(' ', 1)[1]
    @app.post('/bootstrap/register')
    def register_device(req: RegisterDeviceRequest): return service.register_device(req.registration_id, req.registration_token, req.wifi, req.metadata)
    @app.post('/bootstrap/session')
    def create_session(req: BootstrapSessionRequest):
        record = service.store.validate_device(req.registration_id, req.registration_token); token, exp = service.issue_token(record); return {'session_token': token, 'expires_at': exp, 'registration_id': req.registration_id}
    @app.get('/bootstrap/context')
    def fetch_context(token: str = Depends(_bearer)):
        payload = service.decode_token(token); return {'registration_id': payload['registration_id'], 'wifi': payload['wifi'], 'metadata': payload['metadata']}
    @app.post('/bootstrap/ack')
    def acknowledge(req: AckRequest, token: str = Depends(_bearer)):
        payload = service.decode_token(token); ack = {'device_id': req.device_id, 'registration_id': payload['registration_id'], 'ts': time.time()}; service.store.add_ack(token, payload['registration_id'], req.device_id, ack['ts']); return {'ok': True, 'ack': ack}
    @app.get('/bootstrap/acks')
    def list_acks(): return {'acks': service.store.list_acks()}
    return app
