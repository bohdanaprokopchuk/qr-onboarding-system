from __future__ import annotations

import time
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

    def __init__(self, capture: cv2.VideoCapture | None = None) -> None:
        self.capture = capture
        self.last_decision: CameraAdaptationDecision | None = None
        self._last_profile = 'initial'
        self._last_apply_ts = 0.0

    def bind(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture
        self._last_profile = 'initial'
        self._last_apply_ts = 0.0

    def _safe_set(self, prop: int, value: float, label: str, actions: list[str]) -> None:
        if self.capture is None:
            return
        try:
            ok = bool(self.capture.set(prop, value))
            if ok:
                actions.append(f"{label}={value}")
        except Exception:
            return

    def _apply_profile(self, profile: str, actions: list[str]) -> None:
        now = time.monotonic()
        if profile == self._last_profile and (now - self._last_apply_ts) < 1.8:
            return

        if profile == 'low_light':
            self._safe_set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75, 'auto_exposure', actions)
            self._safe_set(cv2.CAP_PROP_GAIN, 4.0, 'gain', actions)
        elif profile == 'overbright':
            self._safe_set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75, 'auto_exposure', actions)
            self._safe_set(cv2.CAP_PROP_GAIN, 1.0, 'gain', actions)
            self._safe_set(cv2.CAP_PROP_EXPOSURE, -6.0, 'exposure', actions)
        else:
            self._safe_set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75, 'auto_exposure', actions)

        self._last_profile = profile
        self._last_apply_ts = now

    def adapt(self, brightness: float, sharpness: float) -> CameraAdaptationDecision:
        actions: list[str] = []
        low_light = brightness < 78.0
        low_sharpness = sharpness < 85.0

        if low_light:
            self._apply_profile('low_light', actions)
        elif brightness > 185.0:
            self._apply_profile('overbright', actions)
        else:
            self._apply_profile('balanced', actions)

        if low_sharpness:
            self._safe_set(cv2.CAP_PROP_AUTOFOCUS, 1.0, 'autofocus', actions)

        decision = CameraAdaptationDecision(
            brightness=brightness,
            sharpness=sharpness,
            low_light=low_light,
            low_sharpness=low_sharpness,
            actions=actions,
        )
        self.last_decision = decision
        return decision
