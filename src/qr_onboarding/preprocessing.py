from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .binarization import di_threshold, niblack_threshold, otsu_threshold, proposed_integral_threshold, yao_threshold
from .models import FrameQualityMetrics


@dataclass
class PreprocessCandidate:
    name: str
    image: np.ndarray
    scale_x: float = 1.0
    scale_y: float = 1.0
    inverse_perspective: Optional[np.ndarray] = None

    def remap_polygon(self, polygon: Any) -> list[tuple[int, int]] | None:
        if polygon is None:
            return None
        arr = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] < 4:
            return None
        if self.inverse_perspective is not None:
            arr = cv2.perspectiveTransform(arr.reshape(1, -1, 2), self.inverse_perspective).reshape(-1, 2)
        if self.scale_x != 1.0:
            arr[:, 0] /= self.scale_x
        if self.scale_y != 1.0:
            arr[:, 1] /= self.scale_y
        return [(int(round(x)), int(round(y))) for x, y in arr]



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


def gamma_boost(gray: np.ndarray, target_mean: float = 138.0) -> np.ndarray:
    mean_value = float(np.mean(gray))
    if not np.isfinite(mean_value):
        return gray.copy()

    current_ratio = float(np.clip(mean_value / 255.0, 1e-4, 1.0 - 1e-4))
    target_ratio = float(np.clip(target_mean / 255.0, 1e-4, 1.0 - 1e-4))

    denominator = float(np.log(current_ratio))
    if abs(denominator) < 1e-6:
        return gray.copy()

    gamma = float(np.clip(np.log(target_ratio) / denominator, 0.45, 1.85))
    lut = np.array([((idx / 255.0) ** gamma) * 255.0 for idx in range(256)], dtype=np.float32)
    return cv2.LUT(gray, np.clip(lut, 0, 255).astype(np.uint8))


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


def detail_preserving_boost(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(denoised)
    return unsharp_masking(clahe, amount=0.9, threshold=2.0)


def glare_compensation(gray: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    compensated = cv2.normalize(closed.astype(np.float32) - gray.astype(np.float32) + 128.0, None, 0, 255, cv2.NORM_MINMAX)
    compensated_u8 = compensated.astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(compensated_u8)


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


def rectification_matrices(points: np.ndarray, size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    inverse = cv2.getPerspectiveTransform(dst, pts)
    return matrix, inverse


def rectify_candidate(gray: np.ndarray, points: np.ndarray, size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    matrix, inverse = rectification_matrices(points, size=size)
    return cv2.warpPerspective(gray, matrix, (size, size)), inverse


def build_candidates(image: np.ndarray, points: Optional[np.ndarray] = None) -> list[PreprocessCandidate]:
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharp = unsharp_masking(gray)
    clahe_sharp = unsharp_masking(clahe)
    gamma = gamma_boost(gray)
    gamma_sharp = unsharp_masking(gamma, amount=0.85, threshold=2.0)
    dynamic_eq = dynamic_illumination_equalization(gray)
    screen_clean = suppress_screen_artifacts(gray)
    watermark_clean = watermark_suppression(gray)
    glare_clean = glare_compensation(gray)
    detail_preserved = detail_preserving_boost(gray)
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
    glare_proposed = proposed_integral_threshold(glare_clean).binary

    out = [
        PreprocessCandidate('gray', gray),
        PreprocessCandidate('sharp', sharp),
        PreprocessCandidate('clahe', clahe),
        PreprocessCandidate('clahe_sharp', clahe_sharp),
        PreprocessCandidate('gamma_boost', gamma),
        PreprocessCandidate('gamma_sharp', gamma_sharp),
        PreprocessCandidate('dynamic_equalized', dynamic_eq),
        PreprocessCandidate('detail_preserved', detail_preserved),
        PreprocessCandidate('glare_compensated', glare_clean),
        PreprocessCandidate('screen_clean', screen_clean),
        PreprocessCandidate('screen_sharp', screen_sharp),
        PreprocessCandidate('watermark_suppressed', watermark_clean),
        PreprocessCandidate('adaptive', adaptive),
        PreprocessCandidate('adaptive_sharp', adaptive_sharp),
        PreprocessCandidate('median', median),
        PreprocessCandidate('upscaled', upscaled, scale_x=2.0, scale_y=2.0),
        PreprocessCandidate('upscaled_adaptive', upscaled_adaptive, scale_x=2.0, scale_y=2.0),
        PreprocessCandidate('otsu', otsu),
        PreprocessCandidate('niblack', niblack),
        PreprocessCandidate('yao', yao),
        PreprocessCandidate('di', di),
        PreprocessCandidate('proposed_integral', proposed),
        PreprocessCandidate('screen_proposed_integral', screen_proposed),
        PreprocessCandidate('watermark_proposed_integral', watermark_proposed),
        PreprocessCandidate('glare_proposed_integral', glare_proposed),
    ]
    if points is not None and len(np.asarray(points).reshape(-1, 2)) >= 4:
        rect, inverse = rectify_candidate(gray, points)
        rclahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(rect)
        rgamma = gamma_boost(rect)
        rdetail = detail_preserving_boost(rect)
        rwatermark = watermark_suppression(rect)
        rglare = glare_compensation(rect)
        rad = cv2.adaptiveThreshold(rclahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        rprop = proposed_integral_threshold(rect).binary
        rscreen = proposed_integral_threshold(suppress_screen_artifacts(rect)).binary
        rglare_prop = proposed_integral_threshold(rglare).binary
        out += [
            PreprocessCandidate('rectified', rect, inverse_perspective=inverse),
            PreprocessCandidate('rectified_clahe', rclahe, inverse_perspective=inverse),
            PreprocessCandidate('rectified_gamma', rgamma, inverse_perspective=inverse),
            PreprocessCandidate('rectified_detail_preserved', rdetail, inverse_perspective=inverse),
            PreprocessCandidate('rectified_watermark_suppressed', rwatermark, inverse_perspective=inverse),
            PreprocessCandidate('rectified_glare_compensated', rglare, inverse_perspective=inverse),
            PreprocessCandidate('rectified_adaptive', rad, inverse_perspective=inverse),
            PreprocessCandidate('rectified_proposed_integral', rprop, inverse_perspective=inverse),
            PreprocessCandidate('rectified_screen_proposed_integral', rscreen, inverse_perspective=inverse),
            PreprocessCandidate('rectified_glare_proposed_integral', rglare_prop, inverse_perspective=inverse),
        ]
    return out
