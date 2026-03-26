from __future__ import annotations

import base64
import hashlib
import json
import zlib
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

try:
    import cbor2 as _cbor
    _dumps = lambda o: _cbor.dumps(o)
    _loads = lambda b: _cbor.loads(b)
    CBOR_BACKEND = 'cbor2'
except Exception:
    import msgpack
    _dumps = lambda o: msgpack.packb(o, use_bin_type=True)
    _loads = lambda b: msgpack.unpackb(b, raw=False)
    CBOR_BACKEND = 'msgpack-fallback'

from .crypto_utils import load_private_key, load_public_key_hex, sealed_box_decrypt, sealed_box_encrypt
from .models import ParsedPayload

VERSION_X25519_RAW = 0x00
VERSION_CBOR_V1 = 0x01
VERSION_X25519_CBOR_V1 = 0x02
VERSION_JSON_V1 = 0x7B
SPLIT_CHUNK_PREFIX = 'QRC1'
ARMORED_TEXT_PREFIX = 'QRB1'


class PayloadError(ValueError):
    pass


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    n = dict(payload)
    if 'rid' in n and 'registration-id' not in n:
        n['registration-id'] = n.pop('rid')
    if 'rtk' in n and 'registration-token' not in n:
        n['registration-token'] = n.pop('rtk')
    if 'cc' in n and 'CC' not in n:
        n['CC'] = n.pop('cc')
    return n


def _cbor_short_map(payload: dict[str, Any]) -> dict[str, Any]:
    p = _normalize_payload(payload)
    out = {'s': p['ssid'], 'i': p.get('registration-id', p.get('rid'))}
    if p.get('psk'):
        out['p'] = p['psk']
    if p.get('CC'):
        out['c'] = p['CC']
    if p.get('registration-token'):
        out['t'] = p['registration-token']
    return out


def _from_cbor_short_map(obj: dict[str, Any]) -> dict[str, Any]:
    out = {'ssid': obj['s'], 'registration-id': obj['i']}
    if 'p' in obj:
        out['psk'] = obj['p']
    if 'c' in obj:
        out['CC'] = obj['c']
    if 't' in obj:
        out['registration-token'] = obj['t']
    return out


def encode_json_v1(payload: dict[str, Any], compact: bool = True) -> bytes:
    p = _normalize_payload(payload)
    return json.dumps(p, ensure_ascii=False, separators=(',', ':') if compact else None, indent=None if compact else 2).encode('utf-8')


def encode_cbor_v1(payload: dict[str, Any]) -> bytes:
    return bytes([VERSION_CBOR_V1]) + _dumps(_cbor_short_map(payload))


def encode_x25519_raw_json_v1(payload: dict[str, Any], public_key) -> bytes:
    pub = load_public_key_hex(public_key) if isinstance(public_key, str) else public_key
    return bytes([VERSION_X25519_RAW]) + sealed_box_encrypt(encode_json_v1(payload), pub)


def encode_x25519_cbor_v1(payload: dict[str, Any], public_key) -> bytes:
    pub = load_public_key_hex(public_key) if isinstance(public_key, str) else public_key
    return bytes([VERSION_X25519_CBOR_V1]) + sealed_box_encrypt(_dumps(_cbor_short_map(payload)), pub)


def armor_binary_payload(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f'{ARMORED_TEXT_PREFIX}|{digest}|{base64.urlsafe_b64encode(payload).decode("ascii")}'


def dearmor_binary_payload(text: str) -> bytes:
    parts = text.strip().split('|', 2)
    if len(parts) != 3 or parts[0] != ARMORED_TEXT_PREFIX:
        raise PayloadError('Not an armored binary payload')
    _, digest, b64 = parts
    payload = base64.urlsafe_b64decode(b64.encode('ascii'))
    got = hashlib.sha256(payload).hexdigest()[:16]
    if got != digest:
        raise PayloadError('Armored payload digest mismatch')
    return payload


def payload_is_text_friendly(data: bytes) -> bool:
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return False
    return text.isprintable() or ('\n' in text) or ('\t' in text)


def encode_chunk_text(payload: bytes, session_id: str, index: int, total: int, payload_digest: str | None = None) -> str:
    digest = payload_digest or hashlib.sha256(payload).hexdigest()[:16]
    crc = f'{zlib.crc32(payload) & 0xFFFFFFFF:08x}'
    b64 = base64.urlsafe_b64encode(payload).decode('ascii')
    return f'{SPLIT_CHUNK_PREFIX}|{session_id}|{index}|{total}|{digest}|{crc}|{b64}'


def parse_chunk_text(text: str) -> dict[str, Any]:
    parts = text.strip().split('|', 6)
    if len(parts) != 7 or parts[0] != SPLIT_CHUNK_PREFIX:
        raise PayloadError('Not a split chunk payload')
    _, session_id, index, total, digest, crc, b64 = parts
    chunk = base64.urlsafe_b64decode(b64.encode('ascii'))
    got_crc = f'{zlib.crc32(chunk) & 0xFFFFFFFF:08x}'
    if got_crc != crc:
        raise PayloadError('Split chunk CRC mismatch')
    return {'session_id': session_id, 'index': int(index), 'total': int(total), 'digest': digest, 'crc32': crc, 'chunk_bytes': chunk}


def _parse_json_payload(data: bytes, version: Optional[int]) -> ParsedPayload:
    try:
        obj = json.loads(data.decode('utf-8'))
    except Exception as exc:
        raise PayloadError('Invalid JSON payload') from exc
    n = _normalize_payload(obj)
    notes = []
    if 'psk' in n:
        notes.append('QR contains direct network credentials; consider secure bootstrap instead')
    return ParsedPayload('json-v1', n, 'psk' in n, version, 'json', notes)


def _parse_cbor_payload(data: bytes, version: Optional[int]) -> ParsedPayload:
    try:
        obj = _loads(data)
    except Exception as exc:
        raise PayloadError(f'Invalid CBOR payload via {CBOR_BACKEND}') from exc
    if not isinstance(obj, dict) or 's' not in obj or 'i' not in obj:
        raise PayloadError('CBOR payload does not match expected schema')
    n = _from_cbor_short_map(obj)
    notes = [f'CBOR codec backend: {CBOR_BACKEND}']
    if 'psk' in n:
        notes.append('QR contains direct network credentials; consider secure bootstrap instead')
    return ParsedPayload('cbor-v1', n, 'psk' in n, version, 'cbor', notes)


def parse_wifi_payload(text: str) -> dict[str, Any]:
    if not text.startswith('WIFI:'):
        raise PayloadError('Not a Wi-Fi payload')
    body = text[5:]
    fields: dict[str, str] = {}
    key = ''
    value = ''
    in_key = True
    escape = False
    for ch in body:
        if escape:
            if in_key:
                key += ch
            else:
                value += ch
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if in_key and ch == ':':
            in_key = False
            continue
        if (not in_key) and ch == ';':
            if key:
                fields[key] = value
            key = ''
            value = ''
            in_key = True
            continue
        if in_key:
            key += ch
        else:
            value += ch
    if key:
        fields[key] = value
    out: dict[str, Any] = {}
    if 'S' in fields:
        out['ssid'] = fields['S']
    if 'P' in fields:
        out['psk'] = fields['P']
    if 'T' in fields:
        out['security'] = fields['T']
    if 'H' in fields:
        out['hidden'] = fields['H']
    return out


def classify_text_payload(text: str, version_byte: Optional[int] = None) -> ParsedPayload:
    text = text.strip()
    if text.startswith(ARMORED_TEXT_PREFIX + '|'):
        payload = dearmor_binary_payload(text)
        parsed = decode_versioned_payload(payload)
        parsed.notes.insert(0, 'Decoded from text-armored binary payload')
        return parsed
    if text.startswith(SPLIT_CHUNK_PREFIX + '|'):
        meta = parse_chunk_text(text)
        preview = base64.urlsafe_b64encode(meta['chunk_bytes'][:16]).decode('ascii')
        return ParsedPayload('split-chunk', {'session_id': meta['session_id'], 'index': meta['index'], 'total': meta['total'], 'digest': meta['digest'], 'crc32': meta['crc32'], 'chunk_preview_b64': preview}, False, version_byte, 'chunked-text', ['One chunk of a multi-QR payload'])
    if text.startswith('WIFI:'):
        return ParsedPayload('wifi-direct', parse_wifi_payload(text), True, version_byte, 'wifi-string', ['Direct Wi-Fi onboarding payload'])
    if text.startswith('DPP:'):
        return ParsedPayload('wifi-easy-connect', {'uri': text}, False, version_byte, 'dpp-uri', ['Bootstrap URI for Wi-Fi Easy Connect / DPP'])
    if text.startswith('http://') or text.startswith('https://'):
        p = urlparse(text)
        q = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(p.query).items()}
        return ParsedPayload('secure-bootstrap-uri', {'uri': text, 'scheme': p.scheme, 'host': p.netloc, 'path': p.path, 'query': q}, False, version_byte, 'uri', ['Secure bootstrap URI'])
    try:
        return _parse_json_payload(text.encode('utf-8'), version_byte)
    except PayloadError:
        return ParsedPayload('plain-text', {'text': text}, False, version_byte, 'text', ['Unstructured payload'])


def decode_versioned_payload(data: bytes, private_key=None) -> ParsedPayload:
    if not data:
        raise PayloadError('Empty QR payload')
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        text = None
    if text and text.startswith(ARMORED_TEXT_PREFIX + '|'):
        return decode_versioned_payload(dearmor_binary_payload(text), private_key=private_key)
    version = data[0]
    if version == VERSION_JSON_V1:
        return _parse_json_payload(data, version)
    if version == VERSION_CBOR_V1:
        return _parse_cbor_payload(data[1:], version)
    if version in {VERSION_X25519_RAW, VERSION_X25519_CBOR_V1}:
        if private_key is None:
            raise PayloadError('Encrypted payload requires recipient private key')
        private = load_private_key(private_key) if not hasattr(private_key, 'exchange') and not hasattr(private_key, 'public_key') else private_key
        try:
            dec = sealed_box_decrypt(data[1:], private)
        except Exception as exc:
            raise PayloadError('Unable to decrypt sealed-box payload') from exc
        if version == VERSION_X25519_RAW:
            p = _parse_json_payload(dec, version)
            p.source_format = 'encrypted-json'
            p.notes.insert(0, 'Payload decrypted using X25519 sealed box')
            return p
        p = _parse_cbor_payload(dec, version)
        p.source_format = 'encrypted-cbor'
        p.notes.insert(0, 'Payload decrypted using X25519 sealed box')
        return p
    if text:
        return classify_text_payload(text, version_byte=version)
    raise PayloadError(f'Unknown payload version byte: 0x{version:02x}')
