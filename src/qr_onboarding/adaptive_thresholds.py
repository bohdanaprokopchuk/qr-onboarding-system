from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


WARMUP_FRAMES: int = 40


@dataclass
class CalibratedThresholds:
    low_light_brightness: float = 78.0
    motion_defocus_sharpness: float = 85.0
    small_qr_size_px: float = 140.0
    glare_brightness: float = 208.0
    glare_low_contrast: float = 64.0
    is_calibrated: bool = False

    def describe(self) -> str:
        status = 'calibrated' if self.is_calibrated else 'default'
        return (
            f'Thresholds ({status}): '
            f'low_light={self.low_light_brightness:.1f}, '
            f'defocus={self.motion_defocus_sharpness:.1f}, '
            f'small_qr={self.small_qr_size_px:.1f}px, '
            f'glare_brightness={self.glare_brightness:.1f}, '
            f'glare_contrast={self.glare_low_contrast:.1f}'
        )


class AdaptiveThresholdCalibrator:
    def __init__(self, warmup_frames: int = WARMUP_FRAMES) -> None:
        self.warmup_frames = max(5, int(warmup_frames))
        self._brightness: Deque[float] = deque(maxlen=self.warmup_frames)
        self._sharpness: Deque[float] = deque(maxlen=self.warmup_frames)
        self._contrast: Deque[float] = deque(maxlen=self.warmup_frames)
        self._calibrated: CalibratedThresholds | None = None

    def update(self, brightness: float, sharpness: float, contrast: float = 0.0) -> None:
        self._brightness.append(float(brightness))
        self._sharpness.append(float(sharpness))
        if contrast > 0:
            self._contrast.append(float(contrast))
        if len(self._brightness) >= self.warmup_frames and self._calibrated is None:
            self._calibrated = self._compute()

    @property
    def is_ready(self) -> bool:
        return self._calibrated is not None

    @property
    def frames_collected(self) -> int:
        return len(self._brightness)

    def thresholds(self) -> CalibratedThresholds:
        return self._calibrated or CalibratedThresholds()

    def _compute(self) -> CalibratedThresholds:
        b = np.asarray(self._brightness, dtype=np.float32)
        s = np.asarray(self._sharpness, dtype=np.float32)
        c = np.asarray(self._contrast, dtype=np.float32) if self._contrast else np.asarray([64.0], dtype=np.float32)

        low_light = float(np.percentile(b, 25)) * 0.75
        defocus = float(np.percentile(s, 20)) * 0.80
        glare = float(np.percentile(b, 85)) * 1.10
        glare_contrast = float(np.percentile(c, 20)) * 1.20

        return CalibratedThresholds(
            low_light_brightness=max(30.0, min(low_light, 100.0)),
            motion_defocus_sharpness=max(20.0, min(defocus, 150.0)),
            small_qr_size_px=140.0,
            glare_brightness=max(160.0, min(glare, 240.0)),
            glare_low_contrast=max(30.0, min(glare_contrast, 100.0)),
            is_calibrated=True,
        )

    def progress_line(self) -> str:
        if self._calibrated is not None:
            return f'Calibration complete ({self.warmup_frames} frames)'
        return f'Calibration warmup: {self.frames_collected}/{self.warmup_frames} frames'
