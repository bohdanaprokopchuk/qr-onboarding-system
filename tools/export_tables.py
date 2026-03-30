from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description='Copy thesis-ready CSV and figure outputs into a single export folder.')
    parser.add_argument('--source-dir', default=str(ROOT / 'results' / 'thesis_ready'))
    parser.add_argument('--export-dir', default=str(ROOT / 'results' / 'thesis_export'))
    args = parser.parse_args()

    source = Path(args.source_dir)
    export = Path(args.export_dir)
    export.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in list(source.glob('*.csv')) + list((source / 'figures').glob('*.png')):
        target = export / path.name
        shutil.copy2(path, target)
        copied.append(target.name)
    print({'export_dir': str(export), 'files': sorted(copied)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
