from __future__ import annotations

import cv2
import numpy as np

from .models import ScanResult


def _put_shadowed_text(canvas: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, result: ScanResult) -> np.ndarray:
    canvas = frame.copy()
    if result.polygon and len(result.polygon) >= 4:
        pts = np.array(result.polygon, dtype=np.int32).reshape((-1, 1, 2))
        color = (0, 200, 0) if result.success else (0, 0, 255)
        cv2.polylines(canvas, [pts], True, color, 2)

    lines: list[str] = []
    if result.success:
        lines.append(f'decoder: {result.decoder} | stage: {result.stage}')
        if result.parsed_payload:
            lines.append(f'payload: {result.parsed_payload.payload_kind}')
            if result.parsed_payload.sensitive:
                lines.append('warning: payload contains sensitive data')
    else:
        lines.append('QR not decoded')
        if result.error:
            lines.append(result.error)

    if result.quality:
        quality = result.quality
        lines.append(
            f'brightness={quality.mean_brightness:.1f} contrast={quality.contrast_stddev:.1f} sharpness={quality.laplacian_variance:.1f}'
        )
        if quality.projected_qr_size_px is not None:
            lines.append(f'projected_qr_size={quality.projected_qr_size_px:.1f}px')
        lines.append(f'hint: {quality.operator_hint}')

    x, y = 15, 25
    for line in lines:
        _put_shadowed_text(canvas, line, (x, y))
        y += 24
    return canvas
