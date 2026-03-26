import numpy as np
import qrcode

from qr_onboarding.adaptive_camera import AdaptiveCameraController
from qr_onboarding.enhanced_pipeline import EnhancedQRSystem
from qr_onboarding.evaluation import StatisticalEvaluationLoop
from qr_onboarding.payload_optimizer import PayloadComplexityController
from qr_onboarding.roi_tracking import QRROITracker


def _make_qr(text, size=280):
    return np.array(qrcode.make(text).convert('RGB').resize((size, size)))[:, :, ::-1].copy()


class DummyCap:
    def __init__(self):
        self.values = []

    def set(self, prop, value):
        self.values.append((prop, value))
        return True


def test_adaptive_camera_controller_emits_actions():
    cap = DummyCap()
    controller = AdaptiveCameraController(cap)
    decision = controller.adapt(brightness=40.0, sharpness=20.0)
    assert decision.low_light is True
    assert decision.low_sharpness is True
    assert decision.actions
    assert cap.values


def test_roi_tracker_recovers_local_crop_and_remaps_polygon():
    tracker = QRROITracker(padding=10)
    tracker.update([(20, 20), (80, 20), (80, 80), (20, 80)])
    frame = np.full((150, 150, 3), 255, dtype=np.uint8)
    crop, offset = tracker.crop(frame)
    assert crop.shape[0] < frame.shape[0]
    remapped = tracker.remap_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], offset)
    assert remapped[0][0] == offset[0]
    assert remapped[0][1] == offset[1]


def test_payload_complexity_controller_prefers_smaller_variant():
    payload = {"ssid": "LabWiFi", "registration-id": "rid-a", "registration-token": "tok-1234567890"}
    controller = PayloadComplexityController()
    variants = controller.variants(payload)
    assert len(variants) >= 3
    assert variants[0].estimated_size <= variants[-1].estimated_size


def test_statistical_evaluation_loop_reports_success_rate():
    system = EnhancedQRSystem()
    loop = StatisticalEvaluationLoop(system)
    dataset = [
        {"label": "one", "image": _make_qr('{"ssid":"LabWiFi"}'), "expected_substring": "LabWiFi"},
        {"label": "two", "image": _make_qr('{"ssid":"GuestWiFi"}'), "expected_substring": "GuestWiFi"},
    ]
    report = loop.run(dataset)
    assert report.total == 2
    assert report.successes == 2
    assert report.success_rate == 1.0


def test_enhanced_pipeline_tracks_roi_after_success():
    system = EnhancedQRSystem()
    result = system.scan_stream_frame(_make_qr('{"ssid":"TrackedWiFi","registration-id":"rid-track"}'))
    assert result.success is True
    assert system.roi_tracker.state is not None
