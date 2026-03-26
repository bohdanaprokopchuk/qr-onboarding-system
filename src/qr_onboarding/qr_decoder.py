from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from pyzbar.pyzbar import ZBarSymbol, decode as zbar_decode

from .preprocessing import to_gray, unsharp_masking


@dataclass
class RawDecode:
    data: bytes
    polygon: Optional[list[tuple[int, int]]]
    decoder: str
    stage: str


class ZBarRecognizer:
    def recognize(self, image: np.ndarray, stage: str = 'direct') -> Optional[RawDecode]:
        gray = to_gray(image)
        sharp = unsharp_masking(gray, 1.0, 3.0)
        symbols = zbar_decode(sharp, symbols=[ZBarSymbol.QRCODE])
        if not symbols:
            return None
        sym = symbols[0]
        polygon = [(int(p.x), int(p.y)) for p in sym.polygon] if getattr(sym, 'polygon', None) else None
        return RawDecode(bytes(sym.data), polygon, 'pyzbar/zbar', stage)


class OpenCVQRLocator:
    def __init__(self) -> None:
        self.detector = cv2.QRCodeDetector()

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        ok, points = self.detector.detect(image)
        return None if (not ok or points is None) else points.reshape(-1, 2)

    def _raw(self, text: str) -> bytes:
        try:
            return text.encode('latin-1')
        except UnicodeEncodeError:
            return text.encode('utf-8')

    @staticmethod
    def _polygon(points) -> Optional[list[tuple[int, int]]]:
        if points is None:
            return None
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        return [(int(round(x)), int(round(y))) for x, y in arr]

    def detect_and_decode_text(self, image: np.ndarray, stage: str = 'opencv_text') -> Optional[RawDecode]:
        text, points, _ = self.detector.detectAndDecode(image)
        if text:
            return RawDecode(self._raw(text), self._polygon(points), 'opencv-qrcode', stage)

        try:
            ok, decoded_info, decoded_points, _ = self.detector.detectAndDecodeMulti(image)
            if ok and decoded_info:
                for text, pts in zip(decoded_info, decoded_points if decoded_points is not None else []):
                    if text:
                        return RawDecode(self._raw(text), self._polygon(pts), 'opencv-qrcode-multi', f'{stage}_multi')
        except Exception:
            pass

        try:
            text, points, _ = self.detector.detectAndDecodeCurved(image)
            if text:
                return RawDecode(self._raw(text), self._polygon(points), 'opencv-qrcode-curved', stage + '_curved')
        except Exception:
            pass
        return None
