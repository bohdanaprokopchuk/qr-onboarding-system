from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qr_onboarding.benchmark.aggregations import load_rows, summarize_split_rows, summarize_static_rows, summarize_stream_rows, write_summary_tables
from qr_onboarding.benchmark.plots import plot_method_success, plot_scenario_gain, plot_severity_curve, plot_stage_wins


def main() -> int:
    parser = argparse.ArgumentParser(description='Aggregate benchmark raw CSV outputs into final-ready tables and figures.')
    parser.add_argument('--static')
    parser.add_argument('--stream')
    parser.add_argument('--split')
    parser.add_argument('--outdir', default=str(ROOT / 'results' / 'final_ready'))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    combined_summary = {}

    if args.static:
        static_summary = summarize_static_rows(load_rows(args.static))
        combined_summary.update(static_summary)
        write_summary_tables(static_summary, outdir)
        figures = outdir / 'figures'
        plot_method_success(static_summary['overall_metrics'], figures / 'overall_success.png')
        plot_scenario_gain(static_summary['per_scenario_metrics'], figures / 'scenario_gain_adaptive_full.png')
        plot_severity_curve(static_summary['per_severity_metrics'], figures / 'severity_curve_adaptive_full.png')
        plot_stage_wins(static_summary['stage_wins'], figures / 'stage_wins.png')

    if args.stream:
        stream_summary = summarize_stream_rows(load_rows(args.stream))
        combined_summary.update(stream_summary)
        write_summary_tables(stream_summary, outdir)

    if args.split:
        split_summary = summarize_split_rows(load_rows(args.split))
        combined_summary.update(split_summary)
        write_summary_tables(split_summary, outdir)

    (outdir / 'summary.json').write_text(json.dumps(combined_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print({'outdir': str(outdir), 'tables': sorted(path.name for path in outdir.glob('*.csv')), 'figures': sorted(path.name for path in (outdir / 'figures').glob('*.png')) if (outdir / 'figures').exists() else []})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
