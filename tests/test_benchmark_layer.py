from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import qrcode

from qr_onboarding.benchmark import make_payload_truth, save_payload_catalog, write_manifest
from qr_onboarding.benchmark.manifest import StaticBenchmarkCase, load_static_manifest
from qr_onboarding.benchmark.payload_truth import compare_to_truth
from qr_onboarding.benchmark.runner_static import run_static_benchmark
from qr_onboarding.enhanced_pipeline import EnhancedQRSystem
from qr_onboarding.pipeline import QRReader


def _make_qr(text: str, size: int = 320) -> np.ndarray:
    return np.array(qrcode.make(text).convert('RGB').resize((size, size)))[:, :, ::-1].copy()


def test_payload_truth_normalized_match_for_json() -> None:
    truth = make_payload_truth('payload-a', '{"ssid":"Bench","registration-id":"rid-a"}', payload_kind='json-v1')
    result = QRReader().scan_image_direct(_make_qr('{"ssid":"Bench","registration-id":"rid-a"}'))
    comparison = compare_to_truth(result, truth)
    assert comparison.decode_success is True
    assert comparison.normalized_match is True


def test_reader_baselines_and_public_adaptive_modes_decode_clean_qr() -> None:
    image = _make_qr('{"ssid":"Lab","registration-id":"rid-lab"}')
    reader = QRReader()
    assert reader.scan_image_pyzbar_only(image).success is True
    assert reader.scan_image_opencv_only(image).success is True
    system = EnhancedQRSystem()
    assert system.scan_fixed_stage(image, 'proposed_integral').success is True
    system.reset_runtime_state()
    assert system.scan_without_quality_assessment(image).success is True
    system.reset_runtime_state()
    assert system.scan_without_roi(image).success is True


def test_static_benchmark_runner_writes_expected_rows(tmp_path: Path) -> None:
    image = _make_qr('{"ssid":"Tmp","registration-id":"rid-tmp"}')
    image_path = tmp_path / 'sample.png'
    cv2.imwrite(str(image_path), image)

    truth = make_payload_truth('payload_tmp', '{"ssid":"Tmp","registration-id":"rid-tmp"}', payload_kind='json-v1')
    save_payload_catalog({'payload_tmp': truth}, tmp_path / 'payload_catalog.json')
    write_manifest([
        StaticBenchmarkCase(
            case_id='case_tmp',
            image_path='sample.png',
            dataset_group='control',
            scenario='control',
            severity='control',
            payload_id='payload_tmp',
            payload_kind='json-v1',
        )
    ], tmp_path / 'manifest.csv')

    output_csv = tmp_path / 'benchmark.csv'
    run_static_benchmark(load_static_manifest(tmp_path / 'manifest.csv'), {'payload_tmp': truth}, output_csv, ['raw_combined', 'adaptive_full'], manifest_path=tmp_path / 'manifest.csv')

    rows = list(csv.DictReader(output_csv.open('r', encoding='utf-8')))
    assert len(rows) == 2
    assert {row['method'] for row in rows} == {'raw_combined', 'adaptive_full'}
    assert all(row['success'].lower() == 'true' for row in rows)
