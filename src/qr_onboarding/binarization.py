from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass
class BinarizationResult:
    name: str
    binary: np.ndarray
    threshold_map: Optional[np.ndarray]
    elapsed_seconds: float
    notes: list[str]
    window_size: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            'name': self.name,
            'elapsed_seconds': self.elapsed_seconds,
            'notes': self.notes,
            'window_size': self.window_size,
            'foreground_ratio': float(np.mean(self.binary == 0)),
        }


def _to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _integral_mean_std(gray: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    gray32 = gray.astype(np.float32)
    radius = window_size // 2
    integral = cv2.integral(gray32)
    integral_sq = cv2.integral(gray32 * gray32)

    padded = cv2.copyMakeBorder(gray32, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
    padded_sq = cv2.copyMakeBorder(gray32 * gray32, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
    integral = cv2.integral(padded)
    integral_sq = cv2.integral(padded_sq)
    h, w = gray.shape

    y0 = np.arange(0, h)[:, None]
    x0 = np.arange(0, w)[None, :]
    y1 = y0 + window_size
    x1 = x0 + window_size

    area = float(window_size * window_size)
    sums = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    sums_sq = integral_sq[y1, x1] - integral_sq[y0, x1] - integral_sq[y1, x0] + integral_sq[y0, x0]
    mean = sums / area
    variance = np.maximum(sums_sq / area - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(variance, dtype=np.float32)


def _binary_from_threshold(gray: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    out = np.where(gray.astype(np.float32) > threshold, 255, 0).astype(np.uint8)
    return out


def _estimate_window(gray: np.ndarray) -> int:
    h, w = gray.shape[:2]
    base = max(15, min(h, w) // 14)
    return min(61, _ensure_odd(base, 15))


def otsu_threshold(image: np.ndarray) -> BinarizationResult:
    gray = _to_gray(image)
    start = perf_counter()
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return BinarizationResult('otsu', binary, np.full_like(gray, threshold, dtype=np.float32), perf_counter() - start, ['global Otsu threshold'])


def niblack_threshold(image: np.ndarray, window_size: int | None = None, k: float = -0.2) -> BinarizationResult:
    gray = _to_gray(image)
    window_size = _ensure_odd(window_size or _estimate_window(gray), 15)
    start = perf_counter()
    mean, std = _integral_mean_std(gray, window_size)
    threshold_map = mean + k * std
    binary = _binary_from_threshold(gray, threshold_map)
    return BinarizationResult('niblack', binary, threshold_map, perf_counter() - start, ['local Niblack threshold via integral image'], window_size)


def yao_threshold(image: np.ndarray, window_size: int | None = None, k: float = -0.18) -> BinarizationResult:
    """Hybrid thresholding baseline inspired by Yao et al.

    Combines a local Niblack-style threshold with a global Otsu threshold
    using variance and edge-aware gating.
    """
    gray = _to_gray(image)
    window_size = _ensure_odd(window_size or _estimate_window(gray), 15)
    start = perf_counter()
    otsu_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mean, std = _integral_mean_std(gray, window_size)
    local_threshold = mean + k * std
    gradient = cv2.Laplacian(gray, cv2.CV_32F)
    boundary_strength = cv2.GaussianBlur(np.abs(gradient), (0, 0), 1.1)
    variance_gate = np.clip(std / (std.max() + 1e-6), 0.0, 1.0)
    boundary_gate = np.clip(boundary_strength / (boundary_strength.max() + 1e-6), 0.0, 1.0)
    alpha = np.clip(0.65 * variance_gate + 0.35 * (1.0 - boundary_gate), 0.15, 0.92)
    threshold_map = alpha * local_threshold + (1.0 - alpha) * float(otsu_value)
    binary = _binary_from_threshold(gray, threshold_map)
    notes = ['hybrid local/global threshold', 'global Otsu fused with local Niblack response']
    return BinarizationResult('yao', binary, threshold_map, perf_counter() - start, notes, window_size)


def di_threshold(image: np.ndarray, window_size: int | None = None, k: float = 0.15) -> BinarizationResult:
    """Multi-scale local thresholding baseline inspired by Di et al.

    Blends local mean estimates from two window scales and adjusts the
    threshold using local contrast.
    """
    gray = _to_gray(image)
    window_size = _ensure_odd(window_size or _estimate_window(gray), 15)
    start = perf_counter()
    mean_small, std_small = _integral_mean_std(gray, window_size)
    mean_large, _ = _integral_mean_std(gray, min(81, _ensure_odd(window_size * 2 + 1, 21)))
    contrast = std_small / (std_small.max() + 1e-6)
    blended_mean = 0.7 * mean_small + 0.3 * mean_large
    threshold_map = blended_mean * (1.0 - k) + contrast * 8.0
    binary = _binary_from_threshold(gray, threshold_map)
    notes = ['multi-level local mean', 'Wellner-style adaptive baseline with two window scales']
    return BinarizationResult('di', binary, threshold_map, perf_counter() - start, notes, window_size)


def proposed_integral_threshold(image: np.ndarray, window_size: int | None = None, k: float = 0.34, r: float = 128.0) -> BinarizationResult:
    """Proposed adaptive thresholding pipeline for challenging QR images.

    Applies illumination normalization, local contrast enhancement, and
    integral-image statistics before Sauvola-style thresholding.
    """

    gray = _to_gray(image)
    window_size = _ensure_odd(window_size or _estimate_window(gray), 15)
    start = perf_counter()

    h, w = gray.shape[:2]
    if min(h, w) >= 96:
        down = cv2.resize(gray, (max(32, w // 2), max(32, h // 2)), interpolation=cv2.INTER_AREA)
        resampled = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        resampled = gray
    denoised = cv2.bilateralFilter(resampled, 7, 35, 35)
    background_kernel = max(31, (min(gray.shape[:2]) // 8) | 1)
    background = cv2.GaussianBlur(denoised.astype(np.float32), (background_kernel, background_kernel), 0)
    background = np.maximum(background, 1.0)
    equalized = np.clip(denoised.astype(np.float32) * (np.mean(background) / background), 0, 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    top_hat = cv2.morphologyEx(equalized, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.normalize(cv2.addWeighted(equalized, 0.7, top_hat, 0.8, 0), None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(enhanced)

    mean, std = _integral_mean_std(enhanced, window_size)
    threshold_map = mean * (1.0 + k * ((std / max(r, 1.0)) - 1.0))
    threshold_map = cv2.GaussianBlur(threshold_map.astype(np.float32), (0, 0), 0.8)
    binary = _binary_from_threshold(enhanced, threshold_map)
    binary = cv2.medianBlur(binary, 3)
    notes = [
        'adaptive window + integral-image statistics',
        'dynamic illumination equalization and edge enhancement before local Sauvola-style threshold',
        'mild de-moire resampling for screen-captured QR codes',
    ]
    return BinarizationResult('proposed_integral', binary, threshold_map, perf_counter() - start, notes, window_size)


def build_binarization_suite() -> list[tuple[str, Callable[[np.ndarray], BinarizationResult]]]:
    return [
        ('otsu', otsu_threshold),
        ('niblack', niblack_threshold),
        ('yao', yao_threshold),
        ('di', di_threshold),
        ('proposed_integral', proposed_integral_threshold),
    ]
