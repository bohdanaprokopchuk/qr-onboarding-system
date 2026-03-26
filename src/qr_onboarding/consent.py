from __future__ import annotations

import base64, hashlib, hmac, json, secrets, time
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ConsentRecord:
    consent_id: str
    subject: str
    purpose: str
    issued_at: float
    expires_at: float
    metadata: dict[str, Any]
    signature: str
    def to_payload(self) -> dict[str, Any]:
        return {'consent_id': self.consent_id, 'subject': self.subject, 'purpose': self.purpose, 'issued_at': self.issued_at, 'expires_at': self.expires_at, 'metadata': self.metadata, 'signature': self.signature}

class ConsentManager:
    def __init__(self, secret: str, store: Optional[Any] = None):
        self.secret = secret.encode('utf-8')
        self.store = store
    def _canonical(self, payload: dict[str, Any]) -> bytes:
        return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    def _sign(self, payload: dict[str, Any]) -> str:
        mac = hmac.new(self.secret, self._canonical(payload), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(mac).decode('ascii').rstrip('=')
    def issue(self, subject: str, purpose: str, ttl_seconds: int = 300, metadata: Optional[dict[str, Any]] = None) -> ConsentRecord:
        now = time.time()
        payload = {'consent_id': secrets.token_hex(8), 'subject': subject, 'purpose': purpose, 'issued_at': now, 'expires_at': now + ttl_seconds, 'metadata': metadata or {}}
        signature = self._sign(payload)
        record = ConsentRecord(signature=signature, **payload)
        if self.store is not None:
            self.store.save_consent(record.consent_id, record.to_payload(), verified=False)
        return record
    def verify(self, payload: dict[str, Any]) -> dict[str, Any]:
        signature = payload.get('signature', '')
        unsigned = {k: payload[k] for k in payload if k != 'signature'}
        expected = self._sign(unsigned)
        ok = hmac.compare_digest(signature, expected) and float(payload['expires_at']) >= time.time()
        result = {'ok': ok, 'reason': None if ok else 'expired-or-invalid-signature', 'fingerprint': hashlib.sha256(self._canonical(unsigned)).hexdigest()[:16], 'payload': payload}
        if ok and self.store is not None:
            self.store.mark_consent_verified(payload['consent_id'])
        return result
