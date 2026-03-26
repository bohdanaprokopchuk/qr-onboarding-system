from __future__ import annotations
import base64, os
from dataclasses import dataclass
from pathlib import Path
try:
    from nacl.public import PrivateKey as NaClPrivateKey, PublicKey as NaClPublicKey, SealedBox
    _HAS_NACL=True
except Exception:
    NaClPrivateKey = NaClPublicKey = SealedBox = None
    _HAS_NACL=False
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
HEXCHARS=set('0123456789abcdefABCDEF')
@dataclass
class X25519KeyPair:
    private_key: object
    @property
    def public_key(self):
        return self.private_key.public_key if _HAS_NACL and isinstance(self.private_key, NaClPrivateKey) else self.private_key.public_key()
    @property
    def public_key_hex(self)->str: return _public_key_bytes(self.public_key).hex()
    @property
    def private_key_hex(self)->str: return _private_key_bytes(self.private_key).hex()
def generate_demo_keypair()->X25519KeyPair:
    return X25519KeyPair(NaClPrivateKey.generate() if _HAS_NACL else x25519.X25519PrivateKey.generate())
def _normalize_raw_private_key_bytes(raw: bytes) -> bytes:
    if len(raw)==32: return raw
    if len(raw)>32: return raw[-32:]
    raise ValueError(f'Private key is too short: {len(raw)} bytes')
def _decode_possible_text_blob(raw: bytes) -> bytes:
    stripped=raw.strip()
    try: text=stripped.decode('ascii')
    except UnicodeDecodeError: return stripped
    if text and len(text)%2==0 and set(text)<=HEXCHARS: return bytes.fromhex(text)
    try:
        dec=base64.b64decode(text, validate=False)
        if dec: return dec
    except Exception: pass
    return stripped
def _public_key_bytes(key)->bytes:
    if _HAS_NACL and isinstance(key, NaClPublicKey): return bytes(key)
    return key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
def _private_key_bytes(key)->bytes:
    if _HAS_NACL and isinstance(key, NaClPrivateKey): return bytes(key)
    return key.private_bytes(encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw, encryption_algorithm=serialization.NoEncryption())
def load_private_key(value):
    if isinstance(value, Path): raw=_decode_possible_text_blob(value.read_bytes())
    elif isinstance(value, str):
        p=Path(value)
        if p.exists(): raw=_decode_possible_text_blob(p.read_bytes())
        else:
            s=value.strip()
            try: raw=bytes.fromhex(s)
            except ValueError: raw=base64.b64decode(s, validate=False)
    else: raw=_decode_possible_text_blob(value)
    raw=_normalize_raw_private_key_bytes(raw)
    return NaClPrivateKey(raw) if _HAS_NACL else x25519.X25519PrivateKey.from_private_bytes(raw)
def load_public_key_hex(value):
    if isinstance(value, Path): raw=_decode_possible_text_blob(value.read_bytes())
    elif isinstance(value, bytes): raw=_decode_possible_text_blob(value)
    else:
        s=value.strip(); p=Path(s)
        raw=_decode_possible_text_blob(p.read_bytes()) if p.exists() else bytes.fromhex(s)
    if len(raw)!=32: raise ValueError('X25519 public key must be 32 bytes')
    return NaClPublicKey(raw) if _HAS_NACL else x25519.X25519PublicKey.from_public_bytes(raw)
def _hkdf(shared_secret: bytes, eph_pub: bytes, recipient_pub: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'qr-onboarding-hybrid-box:'+eph_pub+recipient_pub).derive(shared_secret)
def sealed_box_encrypt(plaintext: bytes, recipient_public_key) -> bytes:
    if _HAS_NACL and isinstance(recipient_public_key, NaClPublicKey): return SealedBox(recipient_public_key).encrypt(plaintext)
    eph=x25519.X25519PrivateKey.generate(); eph_pub=_public_key_bytes(eph.public_key()); recipient_pub=_public_key_bytes(recipient_public_key)
    key=_hkdf(eph.exchange(recipient_public_key), eph_pub, recipient_pub); nonce=os.urandom(12)
    return eph_pub+nonce+AESGCM(key).encrypt(nonce, plaintext, None)
def sealed_box_decrypt(ciphertext: bytes, recipient_private_key) -> bytes:
    if _HAS_NACL and isinstance(recipient_private_key, NaClPrivateKey): return SealedBox(recipient_private_key).decrypt(ciphertext)
    eph_pub_raw, nonce, payload = ciphertext[:32], ciphertext[32:44], ciphertext[44:]
    eph_pub=x25519.X25519PublicKey.from_public_bytes(eph_pub_raw)
    recipient_pub=_public_key_bytes(recipient_private_key.public_key())
    key=_hkdf(recipient_private_key.exchange(eph_pub), eph_pub_raw, recipient_pub)
    return AESGCM(key).decrypt(nonce, payload, None)
