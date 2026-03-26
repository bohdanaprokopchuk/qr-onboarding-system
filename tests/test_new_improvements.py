from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import qrcode

from qr_onboarding.adaptive_thresholds import AdaptiveThresholdCalibrator
from qr_onboarding.enhanced_pipeline import EnhancedQRSystem
from qr_onboarding.pipeline_stats import PipelineStatsCollector
from qr_onboarding.split_qr import SplitQRAssembler, _xor_parity, chunk_texts


def _make_qr(text: str, size: int = 280) -> np.ndarray:
    return np.array(qrcode.make(text).convert('RGB').resize((size, size)))[:, :, ::-1].copy()


def test_xor_parity_computation_basic():
    a = b''
    b = b''
    assert _xor_parity([a, b]) == bytes([0x05, 0x07, 0x05])


def test_split_roundtrip_with_parity_and_single_missing_chunk():
    payload = b'Recovery test payload ' * 20
    texts = chunk_texts(payload, 'rec-session', max_chunk_bytes=30, with_parity=True)
    assembler = SplitQRAssembler()
    assembled = None
    for idx, text in enumerate(texts):
        if idx == 1:
            continue
        assembled = assembler.add_chunk_text(text) or assembled
    assert assembled is not None
    assert assembled.payload == payload
    assert assembled.used_parity is True
    assert assembled.recovered_indices == [1]


def test_split_progress_reports_recoverable_state():
    payload = b'progress payload ' * 10
    texts = chunk_texts(payload, 'progress-session', max_chunk_bytes=25, with_parity=True)
    assembler = SplitQRAssembler()
    for idx, text in enumerate(texts):
        if idx == 0:
            continue
        assembler.add_chunk_text(text)
    progress = assembler.progress('progress-session')
    assert progress is not None
    assert progress.have_parity is True
    assert progress.can_recover is True
    assert progress.done is True
    line = progress.status_line()
    assert '■' in line
    assert 'complete' in line


def test_pipeline_stats_adapts_and_persists():
    stats = PipelineStatsCollector(adapt_after=3)
    fallback = ['clahe', 'adaptive', 'otsu']
    stats.record_win('low_light', 'otsu', latency_ms=12.0)
    stats.record_win('low_light', 'otsu', latency_ms=13.0)
    stats.record_win('low_light', 'clahe', latency_ms=9.0)
    assert stats.is_adapted('low_light') is True
    order = stats.top_stages('low_light', fallback=fallback)
    assert order[0] == 'otsu'
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'stats.json'
        stats.save(path)
        restored = PipelineStatsCollector(adapt_after=99)
        restored.load(path)
        assert restored.is_adapted('low_light') is True
        assert restored.top_stages('low_light', fallback=fallback)[0] == 'otsu'


def test_adaptive_thresholds_switch_to_calibrated_values():
    calibrator = AdaptiveThresholdCalibrator(warmup_frames=5)
    for i in range(5):
        calibrator.update(brightness=60 + i, sharpness=70 + i * 2, contrast=40 + i)
    thresholds = calibrator.thresholds()
    assert calibrator.is_ready is True
    assert thresholds.is_calibrated is True
    assert thresholds.low_light_brightness != 78.0


def test_enhanced_pipeline_reports_partial_success_for_split_chunk():
    payload = b'{"ssid":"LabWiFi","registration-id":"rid-a","registration-token":"rtk-a"}'
    text = chunk_texts(payload, 'partial-session', max_chunk_bytes=20, with_parity=True)[0]
    system = EnhancedQRSystem()
    result = system.scan_stream_frame(_make_qr(text))
    assert result.success is False
    assert result.partial_success is True
    assert result.split_progress is not None


def test_enhanced_pipeline_assembles_split_chunks_with_parity():
    payload = b'{"ssid":"LabWiFi","registration-id":"rid-a","registration-token":"rtk-a"}'
    texts = chunk_texts(payload, 'parity-session', max_chunk_bytes=18, with_parity=True)
    system = EnhancedQRSystem()
    last = None
    for idx, text in enumerate(texts):
        if idx == 1:
            continue
        last = system.scan_stream_frame(_make_qr(text))
    assert last is not None
    assert last.success is True
    assert last.assembled is not None
    assert last.assembled['used_parity'] is True
    assert last.base_result is not None
    assert last.base_result.parsed_payload is not None
    assert last.base_result.parsed_payload.normalized['ssid'] == 'LabWiFi'
