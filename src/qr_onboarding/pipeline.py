from __future__ import annotations

import binascii
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .models import DecodeAttempt, ScanResult
from .payload_codecs import PayloadError, classify_text_payload, decode_versioned_payload
from .preprocessing import build_candidates, evaluate_quality
from .qr_decoder import OpenCVQRLocator, RawDecode, ZBarRecognizer


def _looks_like_human_text(text: str) -> bool:
    if not text:
        return False
    printable = sum(1 for ch in text if ch.isprintable() and ch not in '\x00\x01\x02\x03\x04\x05\x06\x07\x08')
    ratio = printable / max(len(text), 1)
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in '\n\r\t')
    return ratio > 0.85 and control == 0


class QRReader:
    def __init__(self, private_key: Optional[str] = None, try_opencv_text_fallback: bool = True) -> None:
        self.private_key = private_key
        self.try_opencv_text_fallback = try_opencv_text_fallback
        self.zbar = ZBarRecognizer()
        self.locator = OpenCVQRLocator()

    def scan_image_direct(self, image: np.ndarray) -> ScanResult:
        attempts: list[DecodeAttempt] = []
        points = self.locator.detect(image)
        quality = evaluate_quality(image, points)
        for raw in self._candidate_decoders(image):
            result = self._try_build_success_result(raw, attempts, quality)
            if result is not None:
                return result
        return ScanResult(success=False, attempts=attempts, quality=quality, error='No QR payload could be decoded in direct mode')

    def scan_image(self, image: np.ndarray) -> ScanResult:
        direct = self.scan_image_direct(image)
        if direct.success:
            return direct
        attempts = list(direct.attempts)
        points = self.locator.detect(image)
        quality = evaluate_quality(image, points)
        for stage, candidate in build_candidates(image, points):
            for raw in self._candidate_decoders(candidate, stage_prefix=stage):
                result = self._try_build_success_result(raw, attempts, quality)
                if result is not None:
                    return result
        return ScanResult(success=False, attempts=attempts, quality=quality, error='No QR payload could be decoded')

    def scan_path(self, path: str | Path) -> ScanResult:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return ScanResult(success=False, error=f'Failed to load image: {path}') if image is None else self.scan_image(image)

    def _candidate_decoders(self, image: np.ndarray, stage_prefix: str = 'direct') -> list[RawDecode]:
        raws = []
        zbar = self.zbar.recognize(image, stage_prefix)
        if zbar:
            raws.append(zbar)
        if self.try_opencv_text_fallback:
            opencv_result = self.locator.detect_and_decode_text(image, stage=f'{stage_prefix}_opencv')
            if opencv_result:
                raws.append(opencv_result)
        return raws

    def _try_build_success_result(self, raw: RawDecode, attempts: list[DecodeAttempt], quality) -> Optional[ScanResult]:
        raw_hex = binascii.hexlify(raw.data).decode('ascii')
        parsed_payload = None
        decoded_text = None
        interpret_error = None
        try:
            parsed_payload = decode_versioned_payload(raw.data, private_key=self.private_key)
            try:
                decoded_text = json.dumps(parsed_payload.normalized, ensure_ascii=False)
            except Exception:
                decoded_text = None
        except PayloadError as exc:
            interpret_error = str(exc)
            try:
                decoded_text = raw.data.decode('utf-8')
                if _looks_like_human_text(decoded_text):
                    parsed_payload = classify_text_payload(decoded_text)
                else:
                    decoded_text = None
            except Exception:
                parsed_payload = None
                decoded_text = None
        if parsed_payload is None and decoded_text is None:
            attempts.append(DecodeAttempt(raw.stage, raw.decoder, False, raw_hex, f'binary:{len(raw.data)} bytes', interpret_error or 'decoded bytes are not interpretable in the expected formats'))
            return None
        attempts.append(DecodeAttempt(raw.stage, raw.decoder, True, raw_hex, decoded_text[:120] if decoded_text else f'binary:{len(raw.data)} bytes', interpret_error))
        return ScanResult(True, raw.decoder, raw.stage, raw_hex, decoded_text, parsed_payload, quality, attempts, raw.polygon, None)
