from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2

from ..enhanced_pipeline import EnhancedQRSystem, EnhancedScanResult
from .manifest import StreamFrameCase, resolve_case_path
from .payload_truth import PayloadTruth, compare_to_truth


def _unwrap(result: EnhancedScanResult) -> dict[str, Any]:
    scan = result.base_result
    return {
        'overall_success': result.success,
        'decode_success': bool(scan and scan.success and (scan.decoded_text is not None or scan.parsed_payload is not None)),
        'stage': result.enhancement_stage or (None if scan is None else scan.stage),
        'decoder': None if scan is None else scan.decoder,
        'roi_used': result.roi_used,
        'partial_success': result.partial_success,
        'notes': ' | '.join(result.notes),
        'selector_scenario': result.scenario,
    }


def run_stream_benchmark(
    manifest: Iterable[StreamFrameCase],
    payload_catalog: dict[str, PayloadTruth],
    output_csv: str | Path,
    *,
    private_key: str | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    rows: list[dict[str, Any]] = []
    manifest_path = Path(manifest_path or output_csv)
    grouped: dict[str, list[StreamFrameCase]] = defaultdict(list)
    for item in manifest:
        grouped[item.sequence_id].append(item)

    for method_name, roi_enabled in [('adaptive_full_stream', True), ('adaptive_no_roi_stream', False)]:
        for sequence_id, items in grouped.items():
            items = sorted(items, key=lambda row: row.frame_index)
            system = EnhancedQRSystem(private_key=private_key)
            first_success = None
            for item in items:
                image_path = resolve_case_path(manifest_path, item.image_path)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f'Unable to read stream benchmark image: {image_path}')
                truth = payload_catalog[item.payload_id]
                t0 = perf_counter()
                result = system.scan_stream_frame(image) if roi_enabled else system.scan_stream_frame_without_roi(image)
                elapsed_ms = (perf_counter() - t0) * 1000.0
                comparison = compare_to_truth(result, truth)
                meta = _unwrap(result)
                if comparison.decode_success and first_success is None:
                    first_success = item.frame_index
                rows.append({
                    'method': method_name,
                    'sequence_id': sequence_id,
                    'frame_index': item.frame_index,
                    'image_path': item.image_path,
                    'scenario': item.scenario,
                    'severity': item.severity,
                    'payload_id': item.payload_id,
                    'success': comparison.decode_success,
                    'overall_success': meta['overall_success'],
                    'exact_text_match': comparison.exact_text_match,
                    'normalized_match': comparison.normalized_match,
                    'first_success_frame_so_far': first_success,
                    'stage': meta['stage'],
                    'decoder': meta['decoder'],
                    'roi_used': meta['roi_used'],
                    'partial_success': meta['partial_success'],
                    'selector_scenario': meta['selector_scenario'],
                    'processing_time_ms': round(elapsed_ms, 3),
                    'notes': meta['notes'],
                })
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No stream benchmark rows were produced')
    with output_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_csv
