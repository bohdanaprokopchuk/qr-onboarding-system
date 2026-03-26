from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

@dataclass
class FrameQualityMetrics:
    mean_brightness: float
    contrast_stddev: float
    laplacian_variance: float
    projected_qr_size_px: Optional[float] = None
    qr_area_ratio: Optional[float] = None
    operator_hint: str = ''
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class DecodeAttempt:
    stage: str
    decoder: str
    success: bool
    raw_hex: Optional[str] = None
    preview: Optional[str] = None
    error: Optional[str] = None
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ParsedPayload:
    payload_kind: str
    normalized: dict[str, Any]
    sensitive: bool = False
    version_byte: Optional[int] = None
    source_format: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ScanResult:
    success: bool
    decoder: Optional[str] = None
    stage: Optional[str] = None
    raw_hex: Optional[str] = None
    decoded_text: Optional[str] = None
    parsed_payload: Optional[ParsedPayload] = None
    quality: Optional[FrameQualityMetrics] = None
    attempts: list[DecodeAttempt] = field(default_factory=list)
    polygon: Optional[list[tuple[int, int]]] = None
    error: Optional[str] = None
    def to_dict(self) -> dict[str, Any]:
        return {
            'success': self.success, 'decoder': self.decoder, 'stage': self.stage, 'raw_hex': self.raw_hex,
            'decoded_text': self.decoded_text, 'parsed_payload': None if self.parsed_payload is None else self.parsed_payload.to_dict(),
            'quality': None if self.quality is None else self.quality.to_dict(), 'attempts': [a.to_dict() for a in self.attempts],
            'polygon': self.polygon, 'error': self.error }
