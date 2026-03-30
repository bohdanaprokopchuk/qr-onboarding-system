from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qr_onboarding.benchmark import load_payload_catalog, load_split_manifest, run_split_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the split-QR assembly benchmark.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--payload-catalog', required=True)
    parser.add_argument('--out', default=str(ROOT / 'results' / 'split_benchmark.csv'))
    parser.add_argument('--private-key')
    args = parser.parse_args()

    manifest = load_split_manifest(args.manifest)
    payload_catalog = load_payload_catalog(args.payload_catalog)
    output = run_split_benchmark(manifest, payload_catalog, args.out, private_key=args.private_key, manifest_path=args.manifest)
    print({'rows': len(manifest), 'output_csv': str(output)}, flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    raise SystemExit(main())
