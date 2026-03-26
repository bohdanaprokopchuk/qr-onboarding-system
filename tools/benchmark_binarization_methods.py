from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import cv2
import numpy as np
import qrcode

from qr_onboarding.binarization import build_binarization_suite
from qr_onboarding.pipeline import QRReader


def make_clean_qr(text: str, size: int = 320) -> np.ndarray:
    return np.array(qrcode.make(text).convert('RGB').resize((size, size)))[:, :, ::-1].copy()


def add_watermark(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    overlay = np.full_like(out, 255)
    cv2.putText(overlay, 'DEMO', (18, out.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (170, 170, 170), 2, cv2.LINE_AA)
    return cv2.addWeighted(out, 0.82, overlay, 0.18, 0)


def add_screen_like_noise(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    xx = np.linspace(0, 2 * np.pi * 12, w, dtype=np.float32)
    yy = np.linspace(0, 2 * np.pi * 10, h, dtype=np.float32)
    pattern = (8.0 * np.sin(xx)[None, :] + 6.0 * np.sin(yy)[:, None]).astype(np.float32)
    return np.clip(image.astype(np.float32) + pattern[..., None], 0, 255).astype(np.uint8)


def make_dataset(samples: int) -> list[tuple[str, np.ndarray, dict[str, str]]]:
    dataset: list[tuple[str, np.ndarray, dict[str, str]]] = []
    for idx in range(samples):
        payload = {'ssid': f'Bench-{idx}', 'registration-id': f'rid-{idx}'}
        payload_text = f'{{"ssid":"Bench-{idx}","registration-id":"rid-{idx}"}}'
        clean = make_clean_qr(payload_text)
        dataset.append((f'clean_{idx}', clean, payload))
        dataset.append((f'low_light_{idx}', np.clip(clean.astype(np.float32) * 0.35, 0, 255).astype(np.uint8), payload))
        dataset.append((f'blur_{idx}', cv2.GaussianBlur(clean, (0, 0), 2.0), payload))
        h, w = clean.shape[:2]
        gradient = np.tile(np.linspace(0.35, 1.0, w, dtype=np.float32), (h, 1))
        shadow = np.clip(clean.astype(np.float32) * gradient[..., None], 0, 255).astype(np.uint8)
        dataset.append((f'shadow_{idx}', shadow, payload))
        dataset.append((f'watermark_{idx}', add_watermark(clean), payload))
        dataset.append((f'screen_{idx}', add_screen_like_noise(add_watermark(clean)), payload))
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description='Benchmark baseline QR binarization methods')
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--csv', default='binarization_benchmark.csv')
    args = parser.parse_args()

    reader = QRReader()
    dataset = make_dataset(args.samples)
    methods = build_binarization_suite()
    rows: list[dict[str, object]] = []

    for name, image, expected_payload in dataset:
        for method_name, method in methods:
            result = method(image)
            decoded = reader.scan_image(cv2.cvtColor(result.binary, cv2.COLOR_GRAY2BGR))
            normalized = decoded.parsed_payload.normalized if decoded.parsed_payload else {}
            semantic_success = int(decoded.success and all(normalized.get(k) == v for k, v in expected_payload.items()))
            rows.append({
                'sample': name,
                'method': method_name,
                'elapsed_seconds': round(result.elapsed_seconds, 6),
                'success': semantic_success,
                'stage': decoded.stage or '',
                'window_size': result.window_size or '',
            })

    out_path = Path(args.csv)
    with out_path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method_name, _ in methods:
        subset = [r for r in rows if r['method'] == method_name]
        summary[method_name] = {
            'mean_elapsed_seconds': round(statistics.mean(float(r['elapsed_seconds']) for r in subset), 6),
            'recognition_rate_percent': round(100.0 * statistics.mean(int(r['success']) for r in subset), 2),
            'samples': len(subset),
        }
    print(summary)
    print({'csv': str(out_path)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
