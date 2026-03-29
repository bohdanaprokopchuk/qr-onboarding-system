from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, Iterable

import cv2
import numpy as np

from .adaptive_camera import AdaptiveCameraController, CameraAdaptationDecision
from .preprocessing import evaluate_quality


logger = logging.getLogger(__name__)


class _BufferedFrameMixin:

    def _init_buffering(self) -> None:
        self._stop_event = threading.Event()
        self._frame_ready = threading.Event()
        self._frame_lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_ts = 0.0
        self._latest_frame_id = 0

    def _store_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = frame.copy()
            self._latest_frame_ts = time.monotonic()
            self._latest_frame_id += 1
            self._frame_ready.set()

    def _clear_buffer(self) -> None:
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_ts = 0.0
            self._latest_frame_id = 0
            self._frame_ready.clear()

    def _copy_latest_frame(self) -> tuple[np.ndarray | None, float, int]:
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            return frame, self._latest_frame_ts, self._latest_frame_id

    def _read_buffered_frame(self, timeout: float = 1.0, max_staleness: float = 1.0) -> np.ndarray:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame, ts, _ = self._copy_latest_frame()
            if frame is not None and (time.monotonic() - ts) <= max_staleness:
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._frame_ready.wait(timeout=min(0.05, remaining))
        raise RuntimeError('Failed to read a fresh frame from camera')

    @staticmethod
    def _resize_for_quality(frame: np.ndarray, max_dim: int = 640) -> np.ndarray:
        sample = frame
        current_max_dim = max(sample.shape[:2])
        if current_max_dim <= max_dim:
            return sample
        scale = max_dim / float(current_max_dim)
        new_w = max(1, int(round(sample.shape[1] * scale)))
        new_h = max(1, int(round(sample.shape[0] * scale)))
        return cv2.resize(sample, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _adaptive_decision(self, frame: np.ndarray, adaptive_controller: AdaptiveCameraController, read_count: int) -> CameraAdaptationDecision | None:
        try:
            if read_count % 4 != 0 and adaptive_controller.last_decision is not None:
                return adaptive_controller.last_decision
            sample = self._resize_for_quality(frame, max_dim=640)
            quality = evaluate_quality(sample)
            return adaptive_controller.adapt(quality.mean_brightness, quality.laplacian_variance)
        except Exception as exc:
            logger.debug('Adaptive camera update failed: %s', exc, exc_info=True)
            return adaptive_controller.last_decision


class LinuxCameraSource(_BufferedFrameMixin):

    def __init__(
        self,
        device: str | int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        adaptive_controller: AdaptiveCameraController | None = None,
    ) -> None:
        self.device = self._normalize_device(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.capture: cv2.VideoCapture | None = None
        self.backend_name: str | None = None
        self.adaptive_controller = adaptive_controller or AdaptiveCameraController()
        self._adaptive_read_count = 0
        self._init_buffering()

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
            yield cv2.CAP_ANY, 'CAP_ANY'
            yield cv2.CAP_MSMF, 'CAP_MSMF'
            yield cv2.CAP_DSHOW, 'CAP_DSHOW'
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

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        requested = [
            (cv2.CAP_PROP_FRAME_WIDTH, float(self.width)),
            (cv2.CAP_PROP_FRAME_HEIGHT, float(self.height)),
            (cv2.CAP_PROP_FPS, float(self.fps)),
        ]
        for prop, value in requested:
            try:
                capture.set(prop, value)
            except Exception:
                logger.debug('Failed to set capture property %s=%s', prop, value, exc_info=True)

        optional = [
            (getattr(cv2, 'CAP_PROP_BUFFERSIZE', None), 1),
            (getattr(cv2, 'CAP_PROP_AUTOFOCUS', None), 1),
            (getattr(cv2, 'CAP_PROP_AUTO_WB', None), 1),
        ]
        for prop, value in optional:
            if prop is None:
                continue
            try:
                capture.set(prop, value)
            except Exception:
                logger.debug('Optional capture property %s=%s not supported', prop, value, exc_info=True)

        try:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except Exception:
            logger.debug('Auto exposure is not supported by current backend', exc_info=True)

        try:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            logger.debug('MJPG FOURCC not supported by current backend', exc_info=True)

    def _warmup_capture(self, capture: cv2.VideoCapture, attempts: int = 8, timeout: float = 1.5) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        last_good: np.ndarray | None = None

        for _ in range(max(1, attempts)):
            if time.monotonic() >= deadline:
                break
            try:
                ok, frame = capture.read()
            except Exception as exc:
                logger.debug('Capture warmup read failed: %s', exc, exc_info=True)
                ok, frame = False, None
            if ok and frame is not None and frame.size:
                last_good = frame
                time.sleep(0.02)

        return last_good

    def open(self) -> None:
        if self.capture is not None:
            return

        errors: list[str] = []
        self.release()

        for backend, label in self._backend_candidates(self.device):
            capture: cv2.VideoCapture | None = None
            try:
                capture = cv2.VideoCapture(self.device) if backend is None else cv2.VideoCapture(self.device, backend)
            except Exception as exc:
                errors.append(f'{label}: constructor failed: {exc}')
                logger.debug('Camera constructor failed for %s', label, exc_info=True)
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
            first_frame = self._warmup_capture(capture)
            if first_frame is None:
                errors.append(f'{label}: opened but no valid frames during warmup')
                try:
                    capture.release()
                except Exception:
                    pass
                continue

            self.capture = capture
            self.backend_name = label
            self._adaptive_read_count = 0
            self._stop_event.clear()
            self._clear_buffer()
            self._store_frame(first_frame)
            self.adaptive_controller.bind(self.capture)
            self._capture_thread = threading.Thread(
                target=self._capture_worker,
                name=f'camera-capture-{label}',
                daemon=True,
            )
            self._capture_thread.start()
            return

        tried = '; '.join(errors) if errors else 'none'
        raise RuntimeError(f'Unable to open camera source: {self.device}. Tried backends: {tried}')

    def _capture_worker(self) -> None:
        consecutive_failures = 0

        while not self._stop_event.is_set():
            capture = self.capture
            if capture is None:
                break

            try:
                ok, frame = capture.read()
            except Exception as exc:
                logger.debug('Background camera read failed: %s', exc, exc_info=True)
                ok, frame = False, None

            if self._stop_event.is_set():
                break

            if ok and frame is not None and frame.size:
                consecutive_failures = 0
                self._store_frame(frame)
                continue

            consecutive_failures += 1
            time.sleep(0.01 if consecutive_failures < 10 else 0.04)

    def read(self) -> np.ndarray:
        if self.capture is None:
            self.open()
        return self._read_buffered_frame(timeout=1.2, max_staleness=1.0)

    def read_adaptive(self) -> tuple[np.ndarray, CameraAdaptationDecision | None]:
        frame = self.read()
        self._adaptive_read_count += 1
        decision = self._adaptive_decision(frame, self.adaptive_controller, self._adaptive_read_count)
        return frame, decision

    def release(self) -> None:
        self._stop_event.set()

        thread = self._capture_thread
        capture = self.capture

        if thread is not None and thread.is_alive():
            thread.join(timeout=0.6)

        if capture is not None:
            try:
                capture.release()
            except Exception:
                logger.debug('Failed to release OpenCV capture cleanly', exc_info=True)

        if thread is not None and thread.is_alive():
            thread.join(timeout=0.6)

        self._capture_thread = None
        self.capture = None
        self.backend_name = None
        self._clear_buffer()


class _Picamera2ControlAdapter:
    """Small adapter that makes Picamera2 look enough like OpenCV for the adaptive controller."""

    def __init__(self, picam2: Any) -> None:
        self.picam2 = picam2
        self.controls: dict[str, float | int | bool] = {}

    @staticmethod
    def _convert_auto_exposure(value: float) -> bool:
        if value in (0.25, 0.0):
            return False
        if value in (0.75, 1.0):
            return True
        return value >= 0.5

    @staticmethod
    def _convert_exposure(value: float) -> int:
        if value >= 100.0:
            return max(int(round(value)), 100)
        if value >= 0.0:
            return max(int(round(value * 1000.0)), 100)
        return max(int(round((2.0 ** abs(value)) * 100.0)), 100)

    def set(self, prop: int, value: float | int) -> bool:
        try:
            numeric_value = float(value)
        except Exception:
            return False

        mapping: dict[int, tuple[str, float | int | bool]] = {
            cv2.CAP_PROP_AUTO_EXPOSURE: ('AeEnable', self._convert_auto_exposure(numeric_value)),
            cv2.CAP_PROP_EXPOSURE: ('ExposureTime', self._convert_exposure(numeric_value)),
            cv2.CAP_PROP_GAIN: ('AnalogueGain', max(numeric_value, 1.0)),
            cv2.CAP_PROP_BRIGHTNESS: ('Brightness', max(min(numeric_value, 1.0), -1.0)),
            cv2.CAP_PROP_AUTOFOCUS: ('AfMode', 2 if numeric_value >= 0.5 else 0),
            cv2.CAP_PROP_FOCUS: ('LensPosition', max(numeric_value, 0.0)),
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
            logger.debug('Picamera2 control update failed for %s=%s', key, converted, exc_info=True)
            return False


class RaspberryPiCameraSource(_BufferedFrameMixin):
    """Picamera2-backed camera source for Raspberry Pi devices.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        adaptive_controller: AdaptiveCameraController | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.picam2: Any | None = None
        self.adaptive_controller = adaptive_controller or AdaptiveCameraController()
        self._adaptive_read_count = 0
        self._init_buffering()

    def open(self) -> None:
        if self.picam2 is not None:
            return

        self.release()

        try:
            from picamera2 import Picamera2
        except Exception as exc:
            raise RuntimeError('Picamera2 is not installed. Install picamera2 on Raspberry Pi OS to use the Pi camera stack.') from exc

        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={'size': (self.width, self.height), 'format': 'RGB888'},
            controls={'FrameRate': self.fps},
        )
        picam2.configure(config)
        picam2.start()

        self.picam2 = picam2
        self._adaptive_read_count = 0
        self._stop_event.clear()
        self._clear_buffer()
        self.adaptive_controller.bind(_Picamera2ControlAdapter(self.picam2))

        first_frame = self._capture_bgr_frame()
        self._store_frame(first_frame)

        self._capture_thread = threading.Thread(
            target=self._capture_worker,
            name='picamera2-capture',
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_bgr_frame(self) -> np.ndarray:
        if self.picam2 is None:
            raise RuntimeError('Picamera2 is not initialized')
        frame = self.picam2.capture_array()
        if frame is None or not getattr(frame, 'size', 0):
            raise RuntimeError('Failed to read frame from Picamera2')
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _capture_worker(self) -> None:
        consecutive_failures = 0

        while not self._stop_event.is_set():
            if self.picam2 is None:
                break

            try:
                frame = self._capture_bgr_frame()
            except Exception as exc:
                logger.debug('Background Picamera2 read failed: %s', exc, exc_info=True)
                frame = None

            if self._stop_event.is_set():
                break

            if frame is not None:
                consecutive_failures = 0
                self._store_frame(frame)
                continue

            consecutive_failures += 1
            time.sleep(0.01 if consecutive_failures < 10 else 0.04)

    def read(self) -> np.ndarray:
        if self.picam2 is None:
            self.open()
        return self._read_buffered_frame(timeout=1.2, max_staleness=1.0)

    def read_adaptive(self) -> tuple[np.ndarray, CameraAdaptationDecision | None]:
        frame = self.read()
        self._adaptive_read_count += 1
        decision = self._adaptive_decision(frame, self.adaptive_controller, self._adaptive_read_count)
        return frame, decision

    def release(self) -> None:
        self._stop_event.set()

        thread = self._capture_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.6)

        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                logger.debug('Failed to stop Picamera2 cleanly', exc_info=True)

        if thread is not None and thread.is_alive():
            thread.join(timeout=0.4)

        self._capture_thread = None
        self.picam2 = None
        self._clear_buffer()
