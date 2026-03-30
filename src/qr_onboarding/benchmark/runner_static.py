from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2

from ..enhanced_pipeline import EnhancedScanResult
from ..models import ScanResult
from .manifest import StaticBenchmarkCase, resolve_case_path
from .modes import build_method_runner
from .payload_truth import PayloadTruth, compare_to_truth


def _quality_fields(scan: ScanResult | None) -> dict[str, Any]:
    quality = None if scan is None else scan.quality
    return {
        'brightness': None if quality is None else quality.mean_brightness,
        'contrast': None if quality is None else quality.contrast_stddev,
        'sharpness': None if quality is None else quality.laplacian_variance,
        'projected_qr_size_px': None if quality is None else quality.projected_qr_size_px,
        'qr_area_ratio': None if quality is None else quality.qr_area_ratio,
        'operator_hint': None if quality is None else quality.operator_hint,
    }


def _unwrap(result: ScanResult | EnhancedScanResult) -> tuple[ScanResult | None, dict[str, Any]]:
    if isinstance(result, EnhancedScanResult):
        scan = result.base_result
        extra = {
            'overall_success': result.success,
            'partial_success': result.partial_success,
            'scenario': result.scenario,
            'roi_used': result.roi_used,
            'enhancement_stage': result.enhancement_stage or (None if scan is None else scan.stage),
            'notes': ' | '.join(result.notes),
            'split_progress': result.split_progress,
            'assembled': bool(result.assembled),
        }
        return scan, extra
    scan = result
    return scan, {
        'overall_success': scan.success,
        'partial_success': False,
        'scenario': None,
        'roi_used': False,
        'enhancement_stage': scan.stage,
        'notes': '',
        'split_progress': None,
        'assembled': False,
    }


def run_static_benchmark(
    manifest: Iterable[StaticBenchmarkCase],
    payload_catalog: dict[str, PayloadTruth],
    output_csv: str | Path,
    method_names: list[str],
    *,
    private_key: str | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    rows: list[dict[str, Any]] = []
    runners = {method: build_method_runner(method, private_key=private_key) for method in method_names}
    manifest_path = Path(manifest_path or output_csv)
    for case in manifest:
        image_path = resolve_case_path(manifest_path, case.image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Unable to read benchmark image: {image_path}')
        truth = payload_catalog[case.payload_id]
        for method_name, runner in runners.items():
            runner.reset()
            t0 = perf_counter()
            result = runner.run(image)
            elapsed_ms = (perf_counter() - t0) * 1000.0
            scan, extra = _unwrap(result)
            comparison = compare_to_truth(result, truth)
            decoder = None if scan is None else scan.decoder
            stage = extra['enhancement_stage']
            rows.append({
                'case_id': case.case_id,
                'image_path': case.image_path,
                'dataset_group': case.dataset_group,
                'scenario': case.scenario,
                'severity': case.severity,
                'payload_id': case.payload_id,
                'payload_kind': case.payload_kind or truth.payload_kind,
                'method': method_name,
                'success': comparison.decode_success,
                'overall_success': extra['overall_success'],
                'exact_text_match': comparison.exact_text_match,
                'normalized_match': comparison.normalized_match,
                'expected_present': case.expected_present,
                'decoder': decoder,
                'stage': stage,
                'selector_scenario': extra['scenario'],
                'roi_used': extra['roi_used'],
                'partial_success': extra['partial_success'],
                'split_progress': extra['split_progress'],
                'assembled': extra['assembled'],
                'processing_time_ms': round(elapsed_ms, 3),
                'expected_kind': comparison.expected_kind,
                'actual_kind': comparison.actual_kind,
                'expected_hash': comparison.expected_hash,
                'actual_hash': comparison.actual_hash,
                'expected_text': comparison.expected_text,
                'actual_text': comparison.actual_text,
                'notes': extra['notes'],
                **_quality_fields(scan),
            })
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No benchmark rows were produced')
    with output_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_csv
