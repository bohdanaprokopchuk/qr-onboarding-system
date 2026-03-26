from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from .enhanced_pipeline import EnhancedQRSystem


@dataclass
class EvaluationSample:
    label: str
    expected_substring: str | None
    success: bool
    partial_success: bool
    decode_ms: float
    stage: str | None
    scenario: str | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    total: int
    successes: int
    partial_successes: int
    success_rate: float
    partial_rate: float
    avg_decode_ms: float
    scenario_counts: dict[str, int]
    stage_counts: dict[str, int]
    samples: list[EvaluationSample]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['samples'] = [sample.to_dict() for sample in self.samples]
        return data


class StatisticalEvaluationLoop:
    def __init__(self, system: EnhancedQRSystem) -> None:
        self.system = system

    def run(self, dataset: Iterable[dict[str, Any]]) -> EvaluationReport:
        samples: list[EvaluationSample] = []
        scenario_counts: dict[str, int] = {}
        stage_counts: dict[str, int] = {}

        for item in dataset:
            label = str(item.get('label', f'sample-{len(samples)}'))
            image = item['image']
            expected_substring = item.get('expected_substring')
            t0 = perf_counter()
            result = self.system.scan_image(np.asarray(image))
            decode_ms = (perf_counter() - t0) * 1000.0
            text = None if result.base_result is None else (result.base_result.decoded_text or '')
            semantic_success = result.success and (expected_substring is None or expected_substring in text)
            scenario = getattr(result, 'scenario', None)
            stage = result.enhancement_stage or (None if result.base_result is None else result.base_result.stage)
            scenario_counts[scenario or 'unknown'] = scenario_counts.get(scenario or 'unknown', 0) + 1
            stage_counts[stage or 'unknown'] = stage_counts.get(stage or 'unknown', 0) + 1
            samples.append(EvaluationSample(label, expected_substring, semantic_success, bool(result.partial_success), decode_ms, stage, scenario, list(result.notes)))

        successes = sum(1 for sample in samples if sample.success)
        partial_successes = sum(1 for sample in samples if sample.partial_success)
        total = len(samples)
        return EvaluationReport(
            total=total,
            successes=successes,
            partial_successes=partial_successes,
            success_rate=(successes / total) if total else 0.0,
            partial_rate=(partial_successes / total) if total else 0.0,
            avg_decode_ms=mean([sample.decode_ms for sample in samples]) if samples else 0.0,
            scenario_counts=scenario_counts,
            stage_counts=stage_counts,
            samples=samples,
        )
