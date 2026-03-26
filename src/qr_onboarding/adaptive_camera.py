from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2


@dataclass
class CameraAdaptationDecision:
    brightness: float
    sharpness: float
    low_light: bool
    low_sharpness: bool
    actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveCameraController:
    """Best-effort camera tuning for Linux/OpenCV/V4L2 backends.

    The controller applies conservative camera adjustments. If a camera or
    backend does not support a property, the write is ignored and the pipeline
    continues without raising an exception.
    """

    def __init__(self, capture: cv2.VideoCapture | None = None) -> None:
        self.capture = capture
        self.last_decision: CameraAdaptationDecision | None = None

    def bind(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture

    def _safe_set(self, prop: int, value: float, label: str, actions: list[str]) -> None:
        if self.capture is None:
            return
        try:
            ok = bool(self.capture.set(prop, value))
            if ok:
                actions.append(f"{label}={value}")
        except Exception:
            return

    def adapt(self, brightness: float, sharpness: float) -> CameraAdaptationDecision:
        actions: list[str] = []
        low_light = brightness < 78.0
        low_sharpness = sharpness < 85.0

        if low_light:
            self._safe_set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75, "auto_exposure", actions)
            self._safe_set(cv2.CAP_PROP_EXPOSURE, -4.0, "exposure", actions)
            self._safe_set(cv2.CAP_PROP_GAIN, 12.0, "gain", actions)
            self._safe_set(cv2.CAP_PROP_BRIGHTNESS, 150.0, "brightness", actions)
        elif brightness > 175.0:
            self._safe_set(cv2.CAP_PROP_EXPOSURE, -7.0, "exposure", actions)
            self._safe_set(cv2.CAP_PROP_GAIN, 4.0, "gain", actions)

        if low_sharpness:
            self._safe_set(cv2.CAP_PROP_AUTOFOCUS, 1.0, "autofocus", actions)
            self._safe_set(cv2.CAP_PROP_FOCUS, 30.0, "focus", actions)

        decision = CameraAdaptationDecision(
            brightness=brightness,
            sharpness=sharpness,
            low_light=low_light,
            low_sharpness=low_sharpness,
            actions=actions,
        )
        self.last_decision = decision
        return decision
