from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qr_onboarding.benchmark import DEFAULT_METHODS, load_payload_catalog, load_static_manifest, run_static_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the static QR benchmark on a manifest-driven dataset.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--payload-catalog', required=True)
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--out', default=str(ROOT / 'results' / 'static_benchmark.csv'))
    parser.add_argument('--private-key')
    args = parser.parse_args()

    manifest = load_static_manifest(args.manifest)
    payload_catalog = load_payload_catalog(args.payload_catalog)
    methods = [item.strip() for item in args.methods.split(',') if item.strip()]
    output = run_static_benchmark(manifest, payload_catalog, args.out, methods, private_key=args.private_key, manifest_path=args.manifest)
    print({'rows': len(manifest) * len(methods), 'methods': methods, 'output_csv': str(output)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
