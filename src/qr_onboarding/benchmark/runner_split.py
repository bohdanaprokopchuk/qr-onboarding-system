from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2

from ..enhanced_pipeline import EnhancedQRSystem
from .manifest import SplitFrameCase, resolve_case_path
from .payload_truth import (
    PayloadTruth,
    compare_to_truth,
    normalized_hash,
    parse_expected_payload,
)


def _extract_assembled_text(result: Any) -> str | None:
    assembled = getattr(result, 'assembled', None)
    if not isinstance(assembled, dict):
        return None

    candidates = (
        assembled.get('final_text'),
        assembled.get('decoded_text'),
        assembled.get('assembled_text'),
        assembled.get('payload_text'),
        assembled.get('text'),
        assembled.get('payload'),
    )

    for value in candidates:
        if value in (None, ''):
            continue
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8')
            except Exception:
                continue
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
            except Exception:
                continue

    return None


def _compare_split_result_to_truth(result: Any, truth: PayloadTruth) -> dict[str, Any]:
    base_comparison = compare_to_truth(result, truth)

    actual_text = _extract_assembled_text(result)
    if actual_text is None:
        actual_text = base_comparison.actual_text

    actual_kind = base_comparison.actual_kind
    actual_hash = base_comparison.actual_hash
    exact_text_match = base_comparison.exact_text_match
    normalized_match = base_comparison.normalized_match

    if actual_text not in (None, ''):
        exact_text_match = actual_text == truth.expected_text
        try:
            parsed = parse_expected_payload(actual_text)
            actual_kind = parsed.payload_kind
            actual_hash = normalized_hash(parsed.normalized)
            normalized_match = bool(
                actual_hash is not None
                and truth.expected_normalized_hash is not None
                and actual_hash == truth.expected_normalized_hash
            )

            if (
                    truth.payload_kind in {'json-v1', 'plain-text'}
                    and truth.expected_normalized_json is not None
            ):
                from .payload_truth import normalized_json_text
                exact_text_match = normalized_json_text(parsed.normalized) == truth.expected_normalized_json

        except Exception:
            actual_hash = None
            normalized_match = False

    return {
        'decode_success': base_comparison.decode_success,
        'exact_text_match': exact_text_match,
        'normalized_match': normalized_match,
        'actual_text': actual_text,
        'actual_hash': actual_hash,
        'actual_kind': actual_kind,
        'expected_text': truth.expected_text,
        'expected_hash': truth.expected_normalized_hash,
        'expected_kind': truth.payload_kind,
    }


def run_split_benchmark(
    manifest: Iterable[SplitFrameCase],
    payload_catalog: dict[str, PayloadTruth],
    output_csv: str | Path,
    *,
    private_key: str | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    rows: list[dict[str, Any]] = []
    manifest_path = Path(manifest_path or output_csv)
    grouped: dict[str, list[SplitFrameCase]] = defaultdict(list)

    for item in manifest:
        grouped[item.session_id].append(item)

    for session_id, items in grouped.items():
        items = sorted(items, key=lambda row: row.frame_index)
        system = EnhancedQRSystem(private_key=private_key)
        complete_frame: int | None = None

        for item in items:
            image_path = resolve_case_path(manifest_path, item.image_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f'Unable to read split benchmark image: {image_path}')

            truth = payload_catalog[item.payload_id]

            t0 = perf_counter()
            result = system.scan_stream_frame_without_roi(image)
            elapsed_ms = (perf_counter() - t0) * 1000.0

            split_comparison = _compare_split_result_to_truth(result, truth)

            if bool(result.success) and complete_frame is None:
                complete_frame = item.frame_index

            rows.append({
                'session_id': session_id,
                'frame_index': item.frame_index,
                'image_path': item.image_path,
                'payload_id': item.payload_id,
                'chunk_index': item.chunk_index,
                'total_chunks': item.total_chunks,
                'parity_available': item.parity_available,
                'success': split_comparison['decode_success'],
                'overall_success': bool(result.success),
                'partial_success': bool(result.partial_success),
                'exact_text_match': bool(split_comparison['exact_text_match']),
                'normalized_match': bool(split_comparison['normalized_match']),
                'complete_frame_so_far': complete_frame,
                'used_parity': bool(result.assembled and result.assembled.get('used_parity')),
                'recovered_indices': '' if not result.assembled else ','.join(
                    str(idx) for idx in result.assembled.get('recovered_indices', [])
                ),
                'split_progress': result.split_progress,
                'stage': result.enhancement_stage or (
                    None if result.base_result is None else result.base_result.stage
                ),
                'decoder': None if result.base_result is None else result.base_result.decoder,
                'processing_time_ms': round(elapsed_ms, 3),
                'expected_kind': split_comparison['expected_kind'],
                'actual_kind': split_comparison['actual_kind'],
                'expected_hash': split_comparison['expected_hash'],
                'actual_hash': split_comparison['actual_hash'],
                'expected_text': split_comparison['expected_text'],
                'actual_text': split_comparison['actual_text'],
                'notes': ' | '.join(result.notes),
            })

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No split benchmark rows were produced')

    with output_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_csv