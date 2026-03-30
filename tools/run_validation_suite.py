from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation_report.csv'


def main() -> int:
    commands = [
        ('pytest', [sys.executable, '-m', 'pytest', 'tests', '-q']),
        ('research-benchmark', [sys.executable, 'tools/evaluate_research_pipeline.py', '--samples', '4']),
        ('binarization-benchmark', [sys.executable, 'tools/benchmark_binarization_methods.py', '--samples', '3']),
        ('generate-static-benchmark-dataset', [sys.executable, 'tools/generate_static_dataset.py', '--payload-count', '2', '--output-root', 'assets/datasets/benchmark_static_smoke']),
        ('run-static-benchmark', [sys.executable, 'tools/run_static_benchmark.py', '--manifest', 'assets/datasets/benchmark_static_smoke/manifest.csv', '--payload-catalog', 'assets/datasets/benchmark_static_smoke/payload_catalog.json', '--methods', 'raw_combined,fixed:proposed_integral,adaptive_full', '--out', 'results/static_benchmark_smoke.csv']),
        ('aggregate-benchmarks', [sys.executable, 'tools/aggregate_benchmarks.py', '--static', 'results/static_benchmark_smoke.csv', '--outdir', 'results/thesis_ready_smoke']),
    ]
    rows = []
    for name, cmd in commands:
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT / 'src')
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
        rows.append({
            'step': name,
            'returncode': proc.returncode,
            'stdout_tail': proc.stdout[-700:],
            'stderr_tail': proc.stderr[-700:],
        })
    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'returncode', 'stdout_tail', 'stderr_tail'])
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(r['returncode'] == 0 for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())

