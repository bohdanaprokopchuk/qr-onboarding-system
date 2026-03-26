import numpy as np, qrcode
from qr_onboarding.enhanced_pipeline import EnhancedQRSystem
from qr_onboarding.split_qr import chunk_texts
def _make_qr(text, size=280): return np.array(qrcode.make(text).convert('RGB').resize((size,size)))[:,:,::-1].copy()
def test_enhanced_pipeline_assembles_split_chunks_across_frames():
    payload=b'{"ssid":"LabWiFi","registration-id":"rid-a","registration-token":"rtk-a"}'; texts=chunk_texts(payload,'sess-a',max_chunk_bytes=24); system=EnhancedQRSystem(); last=None
    for text in texts: last=system.scan_stream_frame(_make_qr(text))
    assert last is not None and last.success is True and last.base_result.parsed_payload.normalized['ssid']=='LabWiFi'
def test_enhanced_pipeline_single_frame_direct_decode():
    system=EnhancedQRSystem(); result=system.scan_image(_make_qr('{"ssid":"SoloWiFi","registration-id":"rid-solo"}')); assert result.success is True and result.base_result.parsed_payload.normalized['ssid']=='SoloWiFi'
