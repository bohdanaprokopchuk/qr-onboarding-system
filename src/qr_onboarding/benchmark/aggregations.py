from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open('r', newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def _to_float(value: Any) -> float:
    return float(value) if value not in (None, '') else 0.0


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return round((sum(bool(v) for v in values) / len(values)) * 100.0, 3) if values else 0.0


def summarize_static_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scenario: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_stage: Counter[str] = Counter()
    by_severity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row['method']].append(row)
        by_scenario[(row['scenario'], row['method'])].append(row)
        by_severity[(row['severity'], row['method'])].append(row)
        stage = row.get('stage') or 'unknown'
        if _to_bool(row.get('success')):
            by_stage[stage] += 1

    overall = []
    for method, items in sorted(by_method.items()):
        times = [_to_float(item.get('processing_time_ms')) for item in items]
        overall.append({
            'method': method,
            'success_rate_pct': _rate(_to_bool(item['success']) for item in items),
            'exact_match_rate_pct': _rate(_to_bool(item['exact_text_match']) for item in items),
            'normalized_match_rate_pct': _rate(_to_bool(item['normalized_match']) for item in items),
            'mean_time_ms': round(statistics.mean(times), 3) if times else 0.0,
            'median_time_ms': round(statistics.median(times), 3) if times else 0.0,
            'count': len(items),
        })

    per_scenario = []
    raw_map: dict[str, float] = {}
    for (scenario, method), items in sorted(by_scenario.items()):
        success_rate = _rate(_to_bool(item['success']) for item in items)
        if method == 'raw_combined':
            raw_map[scenario] = success_rate
        per_scenario.append({
            'scenario': scenario,
            'method': method,
            'success_rate_pct': success_rate,
            'exact_match_rate_pct': _rate(_to_bool(item['exact_text_match']) for item in items),
            'normalized_match_rate_pct': _rate(_to_bool(item['normalized_match']) for item in items),
            'count': len(items),
        })
    for row in per_scenario:
        row['gain_vs_raw_combined_pp'] = round(
            row['success_rate_pct'] - raw_map.get(row['scenario'], 0.0),
            3,
        )

    per_severity = []
    for (severity, method), items in sorted(by_severity.items()):
        per_severity.append({
            'severity': severity,
            'method': method,
            'success_rate_pct': _rate(_to_bool(item['success']) for item in items),
            'count': len(items),
        })

    stage_wins = [{'stage': stage, 'wins': count} for stage, count in by_stage.most_common()]
    return {
        'overall_metrics': overall,
        'per_scenario_metrics': per_scenario,
        'per_severity_metrics': per_severity,
        'stage_wins': stage_wins,
    }


def summarize_stream_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row['method'], row['scenario'])].append(row)

    summary = []
    for (method, scenario), items in sorted(grouped.items()):
        first_hits = [
            int(item['first_success_frame_so_far'])
            for item in items
            if item.get('first_success_frame_so_far') not in (None, '', 'None')
        ]
        summary.append({
            'method': method,
            'scenario': scenario,
            'success_rate_pct': _rate(_to_bool(item['success']) for item in items),
            'exact_match_rate_pct': _rate(_to_bool(item['exact_text_match']) for item in items),
            'normalized_match_rate_pct': _rate(_to_bool(item['normalized_match']) for item in items),
            'mean_time_ms': round(
                statistics.mean(_to_float(item['processing_time_ms']) for item in items),
                3,
            ),
            'first_success_frame_mean': round(statistics.mean(first_hits), 3) if first_hits else None,
            'count': len(items),
        })

    return {'stream_metrics': summary}


def summarize_split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row['session_id']].append(row)

    summary = []
    for session_id, items in sorted(grouped.items()):
        items_sorted = sorted(items, key=lambda item: int(item.get('frame_index', 0)))

        completion_candidates = [
            item for item in items_sorted
            if _to_bool(item.get('overall_success'))
        ]

        final_item = completion_candidates[0] if completion_candidates else items_sorted[-1]

        completion_frame = None
        if final_item.get('complete_frame_so_far') not in (None, '', 'None'):
            completion_frame = int(final_item.get('complete_frame_so_far'))
        elif completion_candidates:
            completion_frame = int(final_item.get('frame_index', 0))

        used_parity = _to_bool(final_item.get('used_parity')) or any(
            _to_bool(item.get('used_parity')) for item in items_sorted
        )

        exact_match_value = 100.0 if _to_bool(final_item.get('exact_text_match')) else 0.0

        normalized_match_value = 0.0
        normalized_raw = final_item.get('normalized_match')
        if normalized_raw not in (None, '', 'None'):
            normalized_text = str(normalized_raw).strip().lower()
            if normalized_text in {'true', 'false', 'yes', 'no', '1', '0'}:
                normalized_match_value = 100.0 if _to_bool(normalized_raw) else 0.0
            else:
                try:
                    numeric = float(normalized_raw)
                    normalized_match_value = round(
                        numeric * 100.0 if numeric <= 1.0 else numeric,
                        3,
                    )
                except ValueError:
                    normalized_match_value = 0.0

        summary.append({
            'session_id': session_id,
            'completed': bool(completion_candidates),
            'completion_frame': completion_frame,
            'used_parity': used_parity,
            'exact_match_rate_pct': round(exact_match_value, 3),
            'normalized_match_rate_pct': round(normalized_match_value, 3),
            'count': len(items_sorted),
        })

    return {'split_metrics': summary}


def write_summary_tables(
    summary: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, rows in summary.items():
        if not rows:
            continue
        path = output_dir / f'{name}.csv'
        with path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)

    return written