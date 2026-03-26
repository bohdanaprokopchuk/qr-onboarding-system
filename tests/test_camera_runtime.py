import numpy as np

from qr_onboarding.camera import LinuxCameraSource
from qr_onboarding.enhanced_pipeline import EnhancedQRSystem
from qr_onboarding.split_qr import chunk_texts


class _FakeCap:
    def __init__(self, opened=True, read_ok=True):
        self._opened = opened
        self._read_ok = read_ok
        self.released = False
        self.settings = []

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.settings.append((prop, value))
        return True

    def read(self):
        if not self._read_ok:
            return False, None
        return True, np.full((32, 32, 3), 255, dtype=np.uint8)

    def release(self):
        self.released = True


def test_camera_source_normalizes_numeric_device_and_falls_back_to_second_backend(monkeypatch):
    attempts = []

    def fake_capture(device, backend=None):
        attempts.append((device, backend))
        if len(attempts) == 1:
            return _FakeCap(opened=False)
        return _FakeCap(opened=True, read_ok=True)

    monkeypatch.setattr('qr_onboarding.camera.sys.platform', 'win32')
    monkeypatch.setattr('qr_onboarding.camera.cv2.VideoCapture', fake_capture)
    source = LinuxCameraSource(device='0')
    source.open()
    assert source.device == 0
    assert len(attempts) >= 2
    assert attempts[0][0] == 0
    assert source.backend_name == 'CAP_MSMF'


def test_scan_stream_frame_keeps_split_chunk_without_running_ml(monkeypatch):
    payload = b'{"ssid":"LabWiFi","registration-id":"rid-a","registration-token":"rtk-a"}'
    chunk = chunk_texts(payload, 'sess-a', max_chunk_bytes=24)[0]

    import qrcode
    frame = np.array(qrcode.make(chunk).convert('RGB').resize((280, 280)))[:, :, ::-1].copy()
    system = EnhancedQRSystem()

    def fail_enhance(_image):
        raise AssertionError('ML enhancement should not run for a valid split chunk')

    monkeypatch.setattr(system.ml_enhancer, 'enhance', fail_enhance)
    result = system.scan_stream_frame(frame)
    assert result.success is False
    assert result.base_result is not None and result.base_result.success is True
    assert result.error == 'Need more chunks'
