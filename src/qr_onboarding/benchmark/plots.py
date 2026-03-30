from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_method_success(overall_rows: list[dict], output_path: str | Path) -> Path:
    methods = [row['method'] for row in overall_rows]
    values = [float(row['success_rate_pct']) for row in overall_rows]
    fig = plt.figure(figsize=(10, 4.6))
    ax = fig.add_subplot(111)
    ax.bar(methods, values)
    ax.set_ylabel('Success rate, %')
    ax.set_title('Overall benchmark success by method')
    ax.tick_params(axis='x', rotation=35)
    return _save(fig, output_path)


def plot_scenario_gain(per_scenario_rows: list[dict], output_path: str | Path, method: str = 'adaptive_full') -> Path:
    rows = [row for row in per_scenario_rows if row['method'] == method]
    scenarios = [row['scenario'] for row in rows]
    gains = [float(row['gain_vs_raw_combined_pp']) for row in rows]
    fig = plt.figure(figsize=(10, 4.6))
    ax = fig.add_subplot(111)
    ax.bar(scenarios, gains)
    ax.set_ylabel('Gain vs raw_combined, pp')
    ax.set_title(f'Scenario gain for {method}')
    ax.tick_params(axis='x', rotation=35)
    return _save(fig, output_path)


def plot_severity_curve(per_severity_rows: list[dict], output_path: str | Path, method: str = 'adaptive_full') -> Path:
    ordered = {'control': 0, 'low': 1, 'medium': 2, 'hard': 3}
    rows = sorted([row for row in per_severity_rows if row['method'] == method], key=lambda row: ordered.get(row['severity'], 99))
    severities = [row['severity'] for row in rows]
    values = [float(row['success_rate_pct']) for row in rows]
    fig = plt.figure(figsize=(8, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(severities, values, marker='o')
    ax.set_ylabel('Success rate, %')
    ax.set_title(f'Success rate by severity for {method}')
    return _save(fig, output_path)


def plot_stage_wins(stage_rows: list[dict], output_path: str | Path, top_n: int = 12) -> Path:
    rows = stage_rows[:top_n]
    stages = [row['stage'] for row in rows]
    wins = [int(row['wins']) for row in rows]
    fig = plt.figure(figsize=(10, 4.6))
    ax = fig.add_subplot(111)
    ax.bar(stages, wins)
    ax.set_ylabel('Successful decodes')
    ax.set_title('Successful decode distribution by stage')
    ax.tick_params(axis='x', rotation=35)
    return _save(fig, output_path)
