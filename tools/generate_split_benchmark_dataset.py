from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import qrcode

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qr_onboarding.benchmark import SplitFrameCase, make_payload_truth, save_payload_catalog, write_manifest
from qr_onboarding.split_qr import chunk_texts


def make_qr(text: str, size: int = 400) -> np.ndarray:
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    return np.array(
        qr.make_image(fill_color='black', back_color='white')
        .convert('RGB')
        .resize((size, size))
    )[:, :, ::-1].copy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Generate split-QR assembly benchmark dataset.'
    )
    parser.add_argument(
        '--output-root',
        default=str(ROOT / 'assets' / 'datasets' / 'benchmark_split_v1')
    )
    parser.add_argument('--session-count', type=int, default=5)
    parser.add_argument('--max-chunk-bytes', type=int, default=40)
    parser.add_argument(
        '--force-parity-every',
        type=int,
        default=2,
        help='Force every N-th session to include a dropped data chunk and a parity chunk'
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    payload_catalog = {}
    manifest_rows: list[SplitFrameCase] = []

    for session_idx in range(args.session_count):
        payload_id = f'split_payload_{session_idx:03d}'
        session_id = f'sess-{session_idx:02d}'

        final_text = (
            f'{{"ssid":"Split-{session_idx}",'
            f'"registration-id":"split-rid-{session_idx}",'
            f'"registration-token":"split-rtk-{session_idx}",'
            f'"CC":"US"}}'
        )

        payload_catalog[payload_id] = make_payload_truth(
            payload_id,
            final_text,
            payload_kind='json-v1',
        )

        chunks = chunk_texts(
            final_text.encode('utf-8'),
            session_id,
            max_chunk_bytes=args.max_chunk_bytes,
            with_parity=True,
        )

        if not chunks:
            continue

        data_chunk_count = len(chunks) - 1
        parity_index = len(chunks) - 1
        drop_index: int | None = None

        if (
            args.force_parity_every > 0
            and session_idx % args.force_parity_every == args.force_parity_every - 1
            and data_chunk_count >= 3
        ):
            drop_index = 1 + (session_idx % (data_chunk_count - 1))

        for frame_index, chunk_text in enumerate(chunks):
            is_parity_chunk = frame_index == parity_index

            if drop_index is not None and frame_index == drop_index:
                continue

            image = make_qr(chunk_text)
            rel = Path(f'session_{session_idx:02d}') / f'chunk_{frame_index:03d}.png'
            full = output_root / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(full), image)

            manifest_rows.append(
                SplitFrameCase(
                    session_id=session_id,
                    frame_index=frame_index,
                    image_path=str(rel),
                    payload_id=payload_id,
                    chunk_index=(-1 if is_parity_chunk else frame_index),
                    total_chunks=data_chunk_count,
                    parity_available=is_parity_chunk,
                    notes=(
                        f'Split-QR sequence; '
                        f'kind={"parity" if is_parity_chunk else "data"}; '
                        f'dropped_chunk={drop_index if drop_index is not None else -1}'
                    ),
                )
            )

    save_payload_catalog(payload_catalog, output_root / 'payload_catalog.json')
    write_manifest(manifest_rows, output_root / 'manifest.csv')

    print({
        'output_root': str(output_root),
        'sessions': args.session_count,
        'frames': len(manifest_rows),
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())