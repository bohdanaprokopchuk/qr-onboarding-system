from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass, asdict
from typing import Any

from .payload_codecs import encode_cbor_v1, encode_json_v1


@dataclass
class PayloadVariant:
    name: str
    data: bytes
    estimated_size: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if isinstance(self.data, (bytes, bytearray)):
            data['data_preview_hex'] = bytes(self.data[:24]).hex()
            data.pop('data', None)
        return data


class PayloadComplexityController:
    """Builds payload variants ordered from denser to lighter encodings."""

    def _normalize(self, payload: dict[str, Any] | bytes | str) -> dict[str, Any] | bytes | str:
        return payload

    def _compact_json_text(self, payload: dict[str, Any]) -> str:
        aliases = {
            "ssid": "s",
            "registration-id": "i",
            "registration-token": "t",
            "psk": "p",
            "CC": "c",
        }
        compact = {aliases.get(k, k): v for k, v in payload.items()}
        return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

    def variants(self, payload: dict[str, Any] | bytes | str) -> list[PayloadVariant]:
        payload = self._normalize(payload)
        variants: list[PayloadVariant] = []

        if isinstance(payload, dict):
            json_v1 = encode_json_v1(payload)
            compact_text = self._compact_json_text(payload).encode("utf-8")
            cbor_v1 = encode_cbor_v1(payload)
            compressed = zlib.compress(json_v1, level=9)
            compressed_b64 = base64.urlsafe_b64encode(compressed)
            variants.extend(
                [
                    PayloadVariant("json-v1", json_v1, len(json_v1), ["baseline JSON payload"]),
                    PayloadVariant("compact-json-aliases", compact_text, len(compact_text), ["shortened keys reduce symbol count"]),
                    PayloadVariant("cbor-v1", cbor_v1, len(cbor_v1), ["binary short-map encoding"]),
                    PayloadVariant("zlib-base64-json", compressed_b64, len(compressed_b64), ["useful for benchmarking payload density only"]),
                ]
            )
        elif isinstance(payload, str):
            raw = payload.encode("utf-8")
            variants.append(PayloadVariant("plain-text", raw, len(raw), ["direct UTF-8 payload"]))
            variants.append(PayloadVariant("zlib-base64-text", base64.urlsafe_b64encode(zlib.compress(raw, 9)), len(base64.urlsafe_b64encode(zlib.compress(raw, 9))), ["compressed text payload"]))
        else:
            variants.append(PayloadVariant("raw-bytes", payload, len(payload), ["binary payload as provided"]))
            variants.append(PayloadVariant("zlib-base64-bytes", base64.urlsafe_b64encode(zlib.compress(payload, 9)), len(base64.urlsafe_b64encode(zlib.compress(payload, 9))), ["compressed byte payload"]))

        variants.sort(key=lambda item: item.estimated_size)
        return variants

    def best_variant(self, payload: dict[str, Any] | bytes | str) -> PayloadVariant:
        return self.variants(payload)[0]
