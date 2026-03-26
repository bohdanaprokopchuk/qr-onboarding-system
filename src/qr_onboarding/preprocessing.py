from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .binarization import di_threshold, niblack_threshold, otsu_threshold, proposed_integral_threshold, yao_threshold
from .models import FrameQualityMetrics


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def unsharp_masking(gray: np.ndarray, amount: float = 1.0, threshold: float = 3.0) -> np.ndarray:
    gf = gray.astype(np.float32)
    blur = cv2.GaussianBlur(gf, (0, 0), 1)
    sharp = cv2.addWeighted(gf, 1 + amount, blur, -amount, 0)
    mask = np.abs(gf - blur) < threshold
    sharp[mask] = gf[mask]
    return np.clip(sharp, 0, 255).astype(np.uint8)


def dynamic_illumination_equalization(gray: np.ndarray) -> np.ndarray:
    gray32 = gray.astype(np.float32)
    kernel = max(31, (min(gray.shape[:2]) // 8) | 1)
    background = cv2.GaussianBlur(gray32, (kernel, kernel), 0)
    background = np.maximum(background, 1.0)
    corrected = gray32 * (np.mean(background) / background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def suppress_screen_artifacts(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    if min(h, w) >= 96:
        down = cv2.resize(gray, (max(32, w // 2), max(32, h // 2)), interpolation=cv2.INTER_AREA)
        resampled = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        resampled = gray
    bilateral = cv2.bilateralFilter(resampled, 7, 35, 35)
    median = cv2.medianBlur(bilateral, 3)
    return median


def watermark_suppression(gray: np.ndarray) -> np.ndarray:
    equalized = dynamic_illumination_equalization(gray)
    denoised = suppress_screen_artifacts(equalized)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(closed)


def screen_artifact_score(image: np.ndarray) -> float:
    gray = to_gray(image)
    suppressed = suppress_screen_artifacts(gray)
    residual = cv2.absdiff(gray, suppressed)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    axis_bias = abs(float(np.mean(np.abs(gx))) - float(np.mean(np.abs(gy))))
    return float(np.mean(residual) + 0.15 * axis_bias)


def operator_hint(mean_brightness: float, lap_var: float, projected_qr_size_px: Optional[float]) -> str:
    hints = []
    if mean_brightness < 55:
        hints.append('increase illumination')
    elif mean_brightness > 220:
        hints.append('reduce glare / overexposure')
    if lap_var < 60:
        hints.append('hold device steady and wait for focus')
    if projected_qr_size_px is not None and projected_qr_size_px < 120:
        hints.append('move camera closer or enlarge QR')
    if projected_qr_size_px is not None and projected_qr_size_px > 900:
        hints.append('move camera slightly farther away')
    return '; '.join(hints) if hints else 'frame quality is acceptable'


def evaluate_quality(image: np.ndarray, points: Optional[np.ndarray] = None) -> FrameQualityMetrics:
    gray = to_gray(image)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    size = None
    ratio = None
    if points is not None and len(points) >= 4:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        size = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % len(pts)]) for i in range(len(pts))]))
        area = float(cv2.contourArea(pts.astype(np.float32)))
        ratio = area / float(image.shape[0] * image.shape[1]) if image.size else None
    return FrameQualityMetrics(mean, std, lap, size, ratio, operator_hint(mean, lap, size))


def rectify_candidate(gray: np.ndarray, points: np.ndarray, size: int = 512) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(gray, m, (size, size))


def build_candidates(image: np.ndarray, points: Optional[np.ndarray] = None) -> list[tuple[str, np.ndarray]]:
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharp = unsharp_masking(gray)
    clahe_sharp = unsharp_masking(clahe)

    dynamic_eq = dynamic_illumination_equalization(gray)
    screen_clean = suppress_screen_artifacts(gray)
    watermark_clean = watermark_suppression(gray)
    screen_sharp = unsharp_masking(screen_clean, amount=0.8, threshold=2.0)

    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
    adaptive_sharp = unsharp_masking(adaptive, amount=0.6, threshold=2.0)
    upscaled = cv2.resize(clahe_sharp, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    upscaled_adaptive = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 3)
    median = cv2.medianBlur(clahe_sharp, 3)

    otsu = otsu_threshold(gray).binary
    niblack = niblack_threshold(dynamic_eq).binary
    yao = yao_threshold(dynamic_eq).binary
    di = di_threshold(dynamic_eq).binary
    proposed = proposed_integral_threshold(gray).binary
    screen_proposed = proposed_integral_threshold(screen_clean).binary
    watermark_proposed = proposed_integral_threshold(watermark_clean).binary

    out = [
        ('gray', gray),
        ('sharp', sharp),
        ('clahe', clahe),
        ('clahe_sharp', clahe_sharp),
        ('dynamic_equalized', dynamic_eq),
        ('screen_clean', screen_clean),
        ('screen_sharp', screen_sharp),
        ('watermark_suppressed', watermark_clean),
        ('adaptive', adaptive),
        ('adaptive_sharp', adaptive_sharp),
        ('median', median),
        ('upscaled', upscaled),
        ('upscaled_adaptive', upscaled_adaptive),
        ('otsu', otsu),
        ('niblack', niblack),
        ('yao', yao),
        ('di', di),
        ('proposed_integral', proposed),
        ('screen_proposed_integral', screen_proposed),
        ('watermark_proposed_integral', watermark_proposed),
    ]
    if points is not None and len(np.asarray(points).reshape(-1, 2)) >= 4:
        rect = rectify_candidate(gray, points)
        rclahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(rect)
        rwatermark = watermark_suppression(rect)
        rad = cv2.adaptiveThreshold(rclahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        rprop = proposed_integral_threshold(rect).binary
        rscreen = proposed_integral_threshold(suppress_screen_artifacts(rect)).binary
        out += [
            ('rectified', rect),
            ('rectified_clahe', rclahe),
            ('rectified_watermark_suppressed', rwatermark),
            ('rectified_adaptive', rad),
            ('rectified_proposed_integral', rprop),
            ('rectified_screen_proposed_integral', rscreen),
        ]
    return out
