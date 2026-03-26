from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List


ADAPT_AFTER: int = 15


@dataclass
class StageStats:
    wins: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return 0.0 if self.wins == 0 else self.total_latency_ms / self.wins

    def record_win(self, latency_ms: float | None = None) -> None:
        self.wins += 1
        if latency_ms is not None:
            self.total_latency_ms += max(0.0, float(latency_ms))

    def to_dict(self) -> dict:
        return {
            'wins': self.wins,
            'total_latency_ms': round(self.total_latency_ms, 4),
            'avg_latency_ms': round(self.avg_latency_ms, 4),
        }


@dataclass
class ScenarioStats:
    stages: Dict[str, StageStats] = field(default_factory=dict)
    total_decoded: int = 0
    total_failed: int = 0

    def record_win(self, stage: str, latency_ms: float | None = None) -> None:
        bucket = self.stages.setdefault(stage, StageStats())
        bucket.record_win(latency_ms)
        self.total_decoded += 1

    def record_fail(self) -> None:
        self.total_failed += 1

    @property
    def total_attempts(self) -> int:
        return self.total_decoded + self.total_failed

    @property
    def success_rate(self) -> float:
        return 0.0 if self.total_attempts == 0 else self.total_decoded / self.total_attempts

    def top_stages(self, n: int = 8) -> List[str]:
        def _score(item: tuple[str, StageStats]) -> tuple[float, int, float, str]:
            name, stats = item
            latency_penalty = 1.0 + (stats.avg_latency_ms / 1000.0)
            effective = stats.wins / latency_penalty
            return (-effective, -stats.wins, stats.avg_latency_ms, name)

        return [name for name, _ in sorted(self.stages.items(), key=_score)[:n]]

    def to_dict(self) -> dict:
        return {
            'wins': {name: stage.wins for name, stage in self.stages.items()},
            'stage_stats': {name: stage.to_dict() for name, stage in self.stages.items()},
            'total_decoded': self.total_decoded,
            'total_failed': self.total_failed,
            'success_rate': round(self.success_rate, 4),
        }


class PipelineStatsCollector:
    def __init__(self, adapt_after: int = ADAPT_AFTER) -> None:
        self.adapt_after = int(adapt_after)
        self._stats: Dict[str, ScenarioStats] = {}

    def _get(self, scenario: str) -> ScenarioStats:
        if scenario not in self._stats:
            self._stats[scenario] = ScenarioStats()
        return self._stats[scenario]

    def record_win(self, scenario: str, stage: str, latency_ms: float | None = None) -> None:
        self._get(scenario).record_win(stage, latency_ms)

    def record_fail(self, scenario: str) -> None:
        self._get(scenario).record_fail()

    def top_stages(self, scenario: str, fallback: List[str] | None = None, n: int = 8) -> List[str]:
        fallback = list(fallback or [])
        stats = self._stats.get(scenario)
        if stats is None or stats.total_decoded < self.adapt_after:
            return fallback
        ordered = stats.top_stages(n=n)
        seen = set(ordered)
        for stage in fallback:
            if stage not in seen:
                ordered.append(stage)
                seen.add(stage)
        return ordered

    def is_adapted(self, scenario: str) -> bool:
        stats = self._stats.get(scenario)
        return stats is not None and stats.total_decoded >= self.adapt_after

    def summary(self) -> dict:
        return {scenario: stats.to_dict() for scenario, stats in self._stats.items()}

    def save(self, path: str | Path) -> None:
        data = {
            'adapt_after': self.adapt_after,
            'stats': self.summary(),
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding='utf-8'))
        self.adapt_after = int(data.get('adapt_after', self.adapt_after))
        self._stats.clear()
        for scenario, payload in data.get('stats', {}).items():
            stats = ScenarioStats(
                total_decoded=int(payload.get('total_decoded', 0)),
                total_failed=int(payload.get('total_failed', 0)),
            )
            stage_stats = payload.get('stage_stats') or {}
            wins_fallback = payload.get('wins') or {}
            stage_names = set(stage_stats) | set(wins_fallback)
            for name in stage_names:
                data_item = stage_stats.get(name, {})
                stats.stages[name] = StageStats(
                    wins=int(data_item.get('wins', wins_fallback.get(name, 0))),
                    total_latency_ms=float(data_item.get('total_latency_ms', 0.0)),
                )
            self._stats[scenario] = stats
