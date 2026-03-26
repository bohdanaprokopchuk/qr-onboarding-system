from __future__ import annotations

import sys
from typing import Iterable

import cv2
import numpy as np

from .adaptive_camera import AdaptiveCameraController, CameraAdaptationDecision
from .preprocessing import evaluate_quality


class LinuxCameraSource:
    """OpenCV camera source with platform-specific backend fallback.

    Tries multiple capture backends depending on the operating system to
    improve camera initialization reliability across Linux, Windows, and macOS.
    """

    def __init__(
        self,
        device: str | int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        adaptive_controller: AdaptiveCameraController | None = None,
    ) -> None:
        self.device = self._normalize_device(device)
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None
        self.backend_name: str | None = None
        self.adaptive_controller = adaptive_controller or AdaptiveCameraController()

    @staticmethod
    def _normalize_device(device: str | int) -> str | int:
        if isinstance(device, str):
            stripped = device.strip()
            if stripped.isdigit():
                return int(stripped)
            return stripped
        return int(device)

    @staticmethod
    def _backend_candidates(device: str | int) -> Iterable[tuple[int | None, str]]:
        if isinstance(device, str) and not device.isdigit():
            yield None, 'CAP_ANY'
            return
        if sys.platform.startswith('win'):
            yield cv2.CAP_DSHOW, 'CAP_DSHOW'
            yield cv2.CAP_MSMF, 'CAP_MSMF'
            yield cv2.CAP_ANY, 'CAP_ANY'
            return
        if sys.platform == 'darwin':
            yield cv2.CAP_AVFOUNDATION, 'CAP_AVFOUNDATION'
            yield cv2.CAP_ANY, 'CAP_ANY'
            return
        yield cv2.CAP_V4L2, 'CAP_V4L2'
        gst = getattr(cv2, 'CAP_GSTREAMER', None)
        if gst is not None:
            yield gst, 'CAP_GSTREAMER'
        yield cv2.CAP_ANY, 'CAP_ANY'

    def _configure_capture(self, capture) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        try:
            capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except Exception:
            pass

    def open(self) -> None:
        errors: list[str] = []
        for backend, label in self._backend_candidates(self.device):
            try:
                capture = cv2.VideoCapture(self.device) if backend is None else cv2.VideoCapture(self.device, backend)
            except Exception as exc:
                errors.append(f'{label}: constructor failed: {exc}')
                continue
            if capture is None or not capture.isOpened():
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        pass
                errors.append(f'{label}: open failed')
                continue
            self._configure_capture(capture)
            ok, _ = capture.read()
            if not ok:
                errors.append(f'{label}: opened but frame read failed')
                try:
                    capture.release()
                except Exception:
                    pass
                continue
            self.capture = capture
            self.backend_name = label
            self.adaptive_controller.bind(self.capture)
            return
        raise RuntimeError(f'Unable to open camera source: {self.device}. Tried backends: {'; '.join(errors) if errors else 'none'}')

    def read(self):
        if self.capture is None:
            self.open()
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError('Failed to read frame from camera')
        return frame

    def read_adaptive(self) -> tuple[np.ndarray, CameraAdaptationDecision | None]:
        frame = self.read()
        try:
            quality = evaluate_quality(frame)
            decision = self.adaptive_controller.adapt(quality.mean_brightness, quality.laplacian_variance)
        except Exception:
            decision = None
        return frame, decision

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self.backend_name = None


class _Picamera2ControlAdapter:
    def __init__(self, picam2):
        self.picam2 = picam2
        self.controls: dict[str, float | int] = {}

    def set(self, prop, value):
        mapping = {
            cv2.CAP_PROP_AUTO_EXPOSURE: ('AeEnable', bool(value)),
            cv2.CAP_PROP_EXPOSURE: ('ExposureTime', max(int(abs(value) * 1000), 100)),
            cv2.CAP_PROP_GAIN: ('AnalogueGain', max(float(value), 1.0)),
            cv2.CAP_PROP_BRIGHTNESS: ('Brightness', float(value)),
            cv2.CAP_PROP_AUTOFOCUS: ('AfMode', 2 if value else 0),
            cv2.CAP_PROP_FOCUS: ('LensPosition', float(value)),
        }
        item = mapping.get(prop)
        if item is None:
            return False
        key, converted = item
        self.controls[key] = converted
        try:
            self.picam2.set_controls({key: converted})
            return True
        except Exception:
            return False


class RaspberryPiCameraSource:
    """Picamera2-backed camera source for Raspberry Pi devices."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        adaptive_controller: AdaptiveCameraController | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.picam2 = None
        self.adaptive_controller = adaptive_controller or AdaptiveCameraController()

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
        except Exception as exc:
            raise RuntimeError('Picamera2 is not installed. Install picamera2 on Raspberry Pi OS to use the Pi camera stack.') from exc
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={'size': (self.width, self.height), 'format': 'RGB888'}, controls={'FrameRate': self.fps})
        self.picam2.configure(config)
        self.picam2.start()
        self.adaptive_controller.bind(_Picamera2ControlAdapter(self.picam2))

    def read(self):
        if self.picam2 is None:
            self.open()
        frame = self.picam2.capture_array()
        if frame is None:
            raise RuntimeError('Failed to read frame from Picamera2')
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def read_adaptive(self) -> tuple[np.ndarray, CameraAdaptationDecision | None]:
        frame = self.read()
        try:
            quality = evaluate_quality(frame)
            decision = self.adaptive_controller.adapt(quality.mean_brightness, quality.laplacian_variance)
        except Exception:
            decision = None
        return frame, decision

    def release(self) -> None:
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            finally:
                self.picam2 = None
