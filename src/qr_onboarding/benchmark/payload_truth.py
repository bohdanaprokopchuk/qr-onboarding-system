from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..enhanced_pipeline import EnhancedScanResult
from ..models import ParsedPayload, ScanResult
from ..payload_codecs import PayloadError, classify_text_payload, decode_versioned_payload


@dataclass(slots=True)
class PayloadTruth:
    payload_id: str
    payload_kind: str
    expected_text: str
    expected_normalized_hash: str | None = None
    expected_normalized_json: str | None = None
    notes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PayloadComparison:
    decode_success: bool
    exact_text_match: bool
    normalized_match: bool
    expected_hash: str | None
    actual_hash: str | None
    expected_text: str
    actual_text: str | None
    expected_kind: str
    actual_kind: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def normalized_json_text(value: Mapping[str, Any] | list[Any] | str) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(canonicalize(value), ensure_ascii=False, separators=(',', ':'))


def normalized_hash(value: Mapping[str, Any] | list[Any]) -> str:
    return hashlib.sha256(normalized_json_text(value).encode('utf-8')).hexdigest()


def parse_expected_payload(expected_text: str) -> ParsedPayload:
    try:
        return decode_versioned_payload(expected_text.encode('utf-8'))
    except PayloadError:
        return classify_text_payload(expected_text)


def make_payload_truth(payload_id: str, expected_text: str, payload_kind: str | None = None, notes: list[str] | None = None) -> PayloadTruth:
    parsed = parse_expected_payload(expected_text)
    return PayloadTruth(
        payload_id=payload_id,
        payload_kind=payload_kind or parsed.payload_kind,
        expected_text=expected_text,
        expected_normalized_hash=normalized_hash(parsed.normalized),
        expected_normalized_json=normalized_json_text(parsed.normalized),
        notes=list(notes or parsed.notes),
    )


def save_payload_catalog(payloads: Mapping[str, PayloadTruth], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {payload_id: payload.to_dict() for payload_id, payload in payloads.items()}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def load_payload_catalog(path: str | Path) -> dict[str, PayloadTruth]:
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    return {payload_id: PayloadTruth(**value) for payload_id, value in raw.items()}


def _extract_scan_result(result: ScanResult | EnhancedScanResult) -> ScanResult | None:
    if isinstance(result, EnhancedScanResult):
        return result.base_result
    return result


def _extract_raw_text_from_scan(scan: ScanResult | None) -> str | None:
    if scan is None or not scan.raw_hex:
        return None
    try:
        raw_bytes = bytes.fromhex(scan.raw_hex)
    except Exception:
        return None
    try:
        return raw_bytes.decode('utf-8')
    except Exception:
        return None


def _infer_kind_from_text(text: str | None) -> str | None:
    if not text:
        return None
    try:
        return parse_expected_payload(text).payload_kind
    except Exception:
        return None


def compare_to_truth(result: ScanResult | EnhancedScanResult, truth: PayloadTruth) -> PayloadComparison:
    scan = _extract_scan_result(result)
    decode_success = bool(scan and scan.success and (scan.decoded_text is not None or scan.parsed_payload is not None))

    raw_text = _extract_raw_text_from_scan(scan)
    raw_kind = _infer_kind_from_text(raw_text)

    parsed_payload = None if scan is None else scan.parsed_payload
    if parsed_payload is None and raw_text is not None:
        try:
            parsed_payload = parse_expected_payload(raw_text)
        except Exception:
            parsed_payload = None

    actual_hash = None
    actual_kind = None
    if parsed_payload is not None:
        actual_hash = normalized_hash(parsed_payload.normalized)
        actual_kind = parsed_payload.payload_kind
    elif raw_kind is not None:
        actual_kind = raw_kind

    if raw_text is not None and raw_kind == truth.payload_kind:
        actual_text = raw_text
    else:
        actual_text = None if scan is None else scan.decoded_text

    exact_text_match = bool(actual_text == truth.expected_text)
    normalized_match = bool(
        actual_hash is not None
        and truth.expected_normalized_hash is not None
        and actual_hash == truth.expected_normalized_hash
    )

    return PayloadComparison(
        decode_success=decode_success,
        exact_text_match=exact_text_match,
        normalized_match=normalized_match,
        expected_hash=truth.expected_normalized_hash,
        actual_hash=actual_hash,
        expected_text=truth.expected_text,
        actual_text=actual_text,
        expected_kind=truth.payload_kind,
        actual_kind=actual_kind,
    )