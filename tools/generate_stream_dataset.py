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

from qr_onboarding.benchmark import StreamFrameCase, make_payload_truth, save_payload_catalog, write_manifest


def make_qr(text: str, size: int = 420) -> np.ndarray:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
    qr.add_data(text)
    qr.make(fit=True)
    return np.array(qr.make_image(fill_color='black', back_color='white').convert('RGB').resize((size, size)))[:, :, ::-1].copy()


def place(qr_image: np.ndarray, canvas_size: int, scale: float, offset_x: int, offset_y: int) -> np.ndarray:
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    resized = cv2.resize(qr_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    h, w = resized.shape[:2]
    cx = canvas_size // 2 + offset_x
    cy = canvas_size // 2 + offset_y
    x0 = max(0, min(canvas_size - w, cx - w // 2))
    y0 = max(0, min(canvas_size - h, cy - h // 2))
    canvas[y0:y0 + h, x0:x0 + w] = resized
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate deterministic stream benchmark sequences for ROI evaluation.')
    parser.add_argument('--output-root', default=str(ROOT / 'assets' / 'datasets' / 'benchmark_stream_v1'))
    parser.add_argument('--sequence-count', type=int, default=6)
    parser.add_argument('--frames-per-sequence', type=int, default=8)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload_catalog = {}
    manifest_rows: list[StreamFrameCase] = []
    scenarios = ['distance_small_qr', 'low_light', 'glare']

    for seq_idx in range(args.sequence_count):
        payload_id = f'stream_payload_{seq_idx:03d}'
        text = f'{{"ssid":"Stream-{seq_idx}","registration-id":"stream-rid-{seq_idx}","registration-token":"stream-rtk-{seq_idx}"}}'
        payload_catalog[payload_id] = make_payload_truth(payload_id, text, payload_kind='json-v1')
        qr = make_qr(text)
        scenario = scenarios[seq_idx % len(scenarios)]
        sequence_id = f'{scenario}_{seq_idx:02d}'
        for frame_index in range(args.frames_per_sequence):
            progress = frame_index / max(args.frames_per_sequence - 1, 1)
            scale = 0.35 + progress * 0.45 if scenario == 'distance_small_qr' else 0.55 + progress * 0.2
            offset_x = int((1.0 - progress) * 80) - 40
            offset_y = int((1.0 - progress) * 60) - 20
            frame = place(qr, 640, scale=scale, offset_x=offset_x, offset_y=offset_y)
            if scenario == 'low_light':
                frame = np.clip(frame.astype(np.float32) * (0.28 + progress * 0.6), 0, 255).astype(np.uint8)
            elif scenario == 'glare':
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.ellipse(mask, (250 + 20 * frame_index, 170 + 8 * frame_index), (120, 42), 18, 0, 360, 255, -1)
                mask = cv2.GaussianBlur(mask, (0, 0), 21)
                frame = np.clip(frame.astype(np.float32) + (mask[..., None] / 255.0) * (110 - 10 * frame_index), 0, 255).astype(np.uint8)
            rel = Path(sequence_id) / f'frame_{frame_index:03d}.png'
            full = output_root / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(full), frame)
            manifest_rows.append(StreamFrameCase(sequence_id=sequence_id, frame_index=frame_index, image_path=str(rel), scenario=scenario, severity='temporal', payload_id=payload_id, payload_kind='json-v1', notes='Deterministic temporal ROI benchmark frame'))

    save_payload_catalog(payload_catalog, output_root / 'payload_catalog.json')
    write_manifest(manifest_rows, output_root / 'manifest.csv')
    print({'output_root': str(output_root), 'sequences': args.sequence_count, 'frames': len(manifest_rows)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
