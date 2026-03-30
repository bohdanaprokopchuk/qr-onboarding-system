from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == '':
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == '':
        return default
    return int(value)


@dataclass(slots=True)
class StaticBenchmarkCase:
    case_id: str
    image_path: str
    dataset_group: str
    scenario: str
    severity: str
    payload_id: str
    payload_kind: str = ''
    expected_present: bool = True
    notes: str = ''

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> 'StaticBenchmarkCase':
        return cls(
            case_id=str(row['case_id']),
            image_path=str(row['image_path']),
            dataset_group=str(row.get('dataset_group', 'static')),
            scenario=str(row.get('scenario', 'unknown')),
            severity=str(row.get('severity', 'unknown')),
            payload_id=str(row['payload_id']),
            payload_kind=str(row.get('payload_kind', '')),
            expected_present=_to_bool(row.get('expected_present'), True),
            notes=str(row.get('notes', '')),
        )


@dataclass(slots=True)
class StreamFrameCase:
    sequence_id: str
    frame_index: int
    image_path: str
    scenario: str
    severity: str
    payload_id: str
    payload_kind: str = ''
    expected_present: bool = True
    notes: str = ''

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> 'StreamFrameCase':
        return cls(
            sequence_id=str(row['sequence_id']),
            frame_index=_to_int(row.get('frame_index')),
            image_path=str(row['image_path']),
            scenario=str(row.get('scenario', 'unknown')),
            severity=str(row.get('severity', 'unknown')),
            payload_id=str(row['payload_id']),
            payload_kind=str(row.get('payload_kind', '')),
            expected_present=_to_bool(row.get('expected_present'), True),
            notes=str(row.get('notes', '')),
        )


@dataclass(slots=True)
class SplitFrameCase:
    session_id: str
    frame_index: int
    image_path: str
    payload_id: str
    chunk_index: int
    total_chunks: int
    parity_available: bool
    payload_kind: str = 'split-chunk'
    expected_present: bool = True
    notes: str = ''

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> 'SplitFrameCase':
        return cls(
            session_id=str(row['session_id']),
            frame_index=_to_int(row.get('frame_index')),
            image_path=str(row['image_path']),
            payload_id=str(row['payload_id']),
            chunk_index=_to_int(row.get('chunk_index')),
            total_chunks=_to_int(row.get('total_chunks')),
            parity_available=_to_bool(row.get('parity_available'), False),
            payload_kind=str(row.get('payload_kind', 'split-chunk')),
            expected_present=_to_bool(row.get('expected_present'), True),
            notes=str(row.get('notes', '')),
        )


def write_manifest(items: Iterable[Any], path: str | Path) -> Path:
    path = Path(path)
    items = list(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        raise ValueError('Manifest requires at least one row')
    fieldnames = [f.name for f in fields(items[0])]
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))
    return path


def _load_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open('r', newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def load_static_manifest(path: str | Path) -> list[StaticBenchmarkCase]:
    return [StaticBenchmarkCase.from_row(row) for row in _load_rows(path)]


def load_stream_manifest(path: str | Path) -> list[StreamFrameCase]:
    return [StreamFrameCase.from_row(row) for row in _load_rows(path)]


def load_split_manifest(path: str | Path) -> list[SplitFrameCase]:
    return [SplitFrameCase.from_row(row) for row in _load_rows(path)]


def resolve_case_path(manifest_path: str | Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return Path(manifest_path).resolve().parent / candidate
