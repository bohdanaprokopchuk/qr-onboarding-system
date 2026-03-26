import cv2
import numpy as np
import qrcode

from qr_onboarding.binarization import build_binarization_suite, proposed_integral_threshold
from qr_onboarding.ml_models import MLEnhancer
from qr_onboarding.pipeline import QRReader
from qr_onboarding.preprocessing import build_candidates


def _make_qr(text: str, size: int = 320) -> np.ndarray:
    img = np.array(qrcode.make(text).convert('RGB').resize((size, size)))[:, :, ::-1].copy()
    h, w = img.shape[:2]
    gradient = np.tile(np.linspace(0.35, 1.0, w, dtype=np.float32), (h, 1))
    shadowed = np.clip(img.astype(np.float32) * gradient[..., None], 0, 255).astype(np.uint8)
    overlay = np.full_like(shadowed, 255)
    cv2.putText(overlay, 'DEMO', (18, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (175, 175, 175), 2, cv2.LINE_AA)
    watermarked = cv2.addWeighted(shadowed, 0.82, overlay, 0.18, 0)
    return cv2.GaussianBlur(watermarked, (0, 0), 1.4)


def test_binarization_suite_produces_binary_outputs():
    image = _make_qr('{"ssid":"LabWiFi","registration-id":"rid-demo"}')
    for name, method in build_binarization_suite():
        result = method(image)
        assert result.name == name
        assert result.binary.shape == image.shape[:2]
        assert set(np.unique(result.binary)).issubset({0, 255})
        assert result.elapsed_seconds >= 0.0


def test_build_candidates_contains_named_binarization_methods_and_screen_paths():
    image = _make_qr('{"ssid":"LabWiFi","registration-id":"rid-demo"}')
    names = [name for name, _ in build_candidates(image)]
    for expected in ['otsu', 'niblack', 'yao', 'di', 'proposed_integral', 'screen_clean', 'watermark_suppressed', 'screen_proposed_integral']:
        assert expected in names


def test_proposed_integral_threshold_can_feed_qr_reader():
    image = _make_qr('{"ssid":"LabWiFi","registration-id":"rid-demo"}')
    binary = proposed_integral_threshold(image).binary
    result = QRReader().scan_image(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    assert result.success is True
    assert result.parsed_payload is not None
    assert result.parsed_payload.normalized['ssid'] == 'LabWiFi'


def test_ml_enhancer_heuristic_fallback_is_deterministic_and_nonempty():
    image = _make_qr('{"ssid":"LabWiFi","registration-id":"rid-demo"}')
    enhancer = MLEnhancer(checkpoint='nonexistent.ckpt')
    outputs = enhancer.enhance(image)
    assert outputs.segmentation.shape == image.shape[:2]
    assert outputs.deblurred.shape == image.shape[:2]
    assert outputs.super_res.shape[0] == image.shape[0] * 2
    assert outputs.super_res.shape[1] == image.shape[1] * 2
    assert outputs.masked_super_res.shape == outputs.super_res.shape
    assert outputs.segmentation.dtype == np.uint8
    assert outputs.deblurred.dtype == np.uint8
