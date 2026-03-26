from __future__ import annotations

import io
import json
from typing import Any, Iterable

import qrcode
from PIL import Image


def make_qr_image(data: str | bytes, error_correction: str = 'M', box_size: int = 10, border: int = 4) -> Image.Image:
    level = {'L': qrcode.constants.ERROR_CORRECT_L, 'M': qrcode.constants.ERROR_CORRECT_M, 'Q': qrcode.constants.ERROR_CORRECT_Q, 'H': qrcode.constants.ERROR_CORRECT_H}[error_correction]
    qr = qrcode.QRCode(error_correction=level, box_size=box_size, border=border)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color='black', back_color='white').convert('RGB')


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def build_payload_qr(normalized_payload: dict[str, Any], **kwargs: Any) -> bytes:
    text = json.dumps(normalized_payload, ensure_ascii=False, separators=(',', ':'))
    return image_to_png_bytes(make_qr_image(text, **kwargs))


def build_text_qr(text: str, **kwargs: Any) -> bytes:
    return image_to_png_bytes(make_qr_image(text, **kwargs))


def build_binary_payload_qr(data: bytes, **kwargs: Any) -> bytes:
    return image_to_png_bytes(make_qr_image(data, **kwargs))


def build_split_qr_pngs(text_chunks: Iterable[str], **kwargs: Any) -> list[bytes]:
    return [image_to_png_bytes(make_qr_image(chunk, **kwargs)) for chunk in text_chunks]
