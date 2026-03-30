from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import qrcode

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qr_onboarding.benchmark import StaticBenchmarkCase, make_payload_truth, save_payload_catalog, write_manifest

SCENARIOS = ['blur', 'motion_blur', 'low_light', 'distance_small_qr', 'glare', 'perspective', 'noise', 'low_contrast', 'screen_artifact']
SEVERITIES = ['low', 'medium', 'hard']


def make_qr_image(text: str, size: int = 420) -> np.ndarray:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
    qr.add_data(text)
    qr.make(fit=True)
    image = qr.make_image(fill_color='black', back_color='white').convert('RGB').resize((size, size))
    return np.array(image)[:, :, ::-1].copy()


def place_on_canvas(qr_image: np.ndarray, canvas_size: int = 640, scale: float = 1.0, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    if scale != 1.0:
        qr_image = cv2.resize(qr_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    h, w = qr_image.shape[:2]
    cx = canvas_size // 2 + offset[0]
    cy = canvas_size // 2 + offset[1]
    x0 = max(0, min(canvas_size - w, cx - w // 2))
    y0 = max(0, min(canvas_size - h, cy - h // 2))
    canvas[y0:y0 + h, x0:x0 + w] = qr_image
    return canvas


def add_motion_blur(image: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((length / 2, length / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel /= np.sum(kernel)
    return cv2.filter2D(image, -1, kernel)


def add_glare(image: np.ndarray, strength: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    overlay = image.astype(np.float32)
    h, w = image.shape[:2]
    center = (int(w * rng.uniform(0.35, 0.65)), int(h * rng.uniform(0.2, 0.5)))
    axes = (int(w * rng.uniform(0.18, 0.28)), int(h * rng.uniform(0.08, 0.18)))
    angle = float(rng.uniform(-35, 35))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=31)
    overlay += (mask[..., None] / 255.0) * (strength * 255.0)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def apply_scenario(image: np.ndarray, scenario: str, severity: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = {'low': 1, 'medium': 2, 'hard': 3}[severity]
    out = image.copy()
    if scenario == 'blur':
        sigma = {1: 1.0, 2: 2.0, 3: 3.4}[level]
        return cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)
    if scenario == 'motion_blur':
        return add_motion_blur(out, {1: 7, 2: 13, 3: 19}[level], {1: 0, 2: 18, 3: 33}[level])
    if scenario == 'low_light':
        factor = {1: 0.72, 2: 0.5, 3: 0.3}[level]
        shadow = np.linspace(1.0, {1: 0.92, 2: 0.72, 3: 0.48}[level], out.shape[1], dtype=np.float32)[None, :, None]
        return np.clip(out.astype(np.float32) * factor * shadow, 0, 255).astype(np.uint8)
    if scenario == 'distance_small_qr':
        scale = {1: 0.72, 2: 0.52, 3: 0.34}[level]
        offset = {1: (0, 0), 2: (45, -30), 3: (-70, 55)}[level]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray < 250)
        qr_crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return place_on_canvas(qr_crop, canvas_size=image.shape[0], scale=scale, offset=offset)
    if scenario == 'glare':
        return add_glare(out, {1: 0.28, 2: 0.45, 3: 0.62}[level], seed)
    if scenario == 'perspective':
        h, w = out.shape[:2]
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        inset = {1: 28, 2: 58, 3: 92}[level]
        dst = np.float32([[inset, inset // 2], [w - inset, 0], [w - 1, h - inset], [0, h - inset // 2]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(out, matrix, (w, h), borderValue=(255, 255, 255))
    if scenario == 'noise':
        sigma = {1: 8.0, 2: 16.0, 3: 26.0}[level]
        noisy = out.astype(np.float32) + rng.normal(0.0, sigma, out.shape)
        if level >= 2:
            mask = rng.random(out.shape[:2]) < (0.002 if level == 2 else 0.008)
            noisy[mask] = 0
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if scenario == 'low_contrast':
        alpha = {1: 0.78, 2: 0.62, 3: 0.48}[level]
        background = np.full_like(out, 210)
        return cv2.addWeighted(out, alpha, background, 1 - alpha, 0.0)
    if scenario == 'screen_artifact':
        h, w = out.shape[:2]
        scanlines = np.ones((h, w, 1), dtype=np.float32)
        scanlines[::2] *= {1: 0.97, 2: 0.93, 3: 0.87}[level]
        xx = np.linspace(0, math.pi * {1: 8, 2: 12, 3: 18}[level], w, dtype=np.float32)
        moire = (np.sin(xx)[None, :, None] * {1: 5.0, 2: 9.0, 3: 14.0}[level])
        resampled = cv2.resize(cv2.resize(out, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
        mixed = resampled.astype(np.float32) * scanlines + moire
        return np.clip(mixed, 0, 255).astype(np.uint8)
    raise ValueError(f'Unsupported scenario: {scenario}')


def build_payloads(count: int) -> list[tuple[str, str, str]]:
    payloads: list[tuple[str, str, str]] = []
    for idx in range(count):
        mod = idx % 6
        payload_id = f'payload_{idx:03d}'
        if mod == 0:
            text = f'{{"ssid":"Bench-{idx}","registration-id":"rid-{idx}","registration-token":"rtk-{idx:03d}"}}'
            kind = 'json-v1'
        elif mod == 1:
            text = f'https://example.org/bootstrap/{idx}?rid=rid-{idx}&token=rtk-{idx:03d}'
            kind = 'secure-bootstrap-uri'
        elif mod == 2:
            text = f'WIFI:T:WPA;S:LabNet-{idx};P:pass-{idx:03d}-A;H:false;;'
            kind = 'wifi-direct'
        elif mod == 3:
            text = f'{{"ssid":"my;\\"shiny\\";ssid-{idx}","psk":"\\"\\"\\"\\\\;;;buzz-{idx};;;\\\\\\"\\"\\"","CC":"US","registration-id":"rid-special-{idx}","registration-token":"rtk-special-{idx}"}}'
            kind = 'json-v1'
        elif mod == 4:
            text = f'DPP:K:MDkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDIgADemo{idx:02d};M:112233445566;C:81/1;I:Device-{idx};;'
            kind = 'wifi-easy-connect'
        else:
            text = f'Plain-text benchmark payload {idx} with unicode Україна and control sentence.'
            kind = 'plain-text'
        payloads.append((payload_id, kind, text))
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate deterministic static benchmark dataset with manifest and payload catalog.')
    parser.add_argument('--output-root', default=str(ROOT / 'assets' / 'datasets' / 'benchmark_static_v1'))
    parser.add_argument('--payload-count', type=int, default=18)
    parser.add_argument('--canvas-size', type=int, default=640)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload_catalog = {}
    manifest_rows: list[StaticBenchmarkCase] = []
    payloads = build_payloads(args.payload_count)

    for payload_idx, (payload_id, payload_kind, text) in enumerate(payloads):
        payload_catalog[payload_id] = make_payload_truth(payload_id, text, payload_kind=payload_kind)
        clean = place_on_canvas(make_qr_image(text), canvas_size=args.canvas_size)
        clean_rel = Path('control') / f'{payload_id}_clean.png'
        (output_root / clean_rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_root / clean_rel), clean)
        manifest_rows.append(StaticBenchmarkCase(case_id=f'{payload_id}_control', image_path=str(clean_rel), dataset_group='control', scenario='control', severity='control', payload_id=payload_id, payload_kind=payload_kind, notes='Clean reference QR image'))
        for scenario in SCENARIOS:
            for severity_idx, severity in enumerate(SEVERITIES, start=1):
                image = apply_scenario(clean, scenario, severity, seed=payload_idx * 100 + severity_idx * 11 + len(scenario))
                rel = Path('challenging') / scenario / severity / f'{payload_id}_{scenario}_{severity}.png'
                rel_path = output_root / rel
                rel_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(rel_path), image)
                manifest_rows.append(StaticBenchmarkCase(case_id=f'{payload_id}_{scenario}_{severity}', image_path=str(rel), dataset_group='challenging', scenario=scenario, severity=severity, payload_id=payload_id, payload_kind=payload_kind, notes=f'Deterministic {scenario} severity={severity}'))

    save_payload_catalog(payload_catalog, output_root / 'payload_catalog.json')
    write_manifest(manifest_rows, output_root / 'manifest.csv')
    print({'output_root': str(output_root), 'payload_count': len(payloads), 'control_cases': len(payloads), 'challenging_cases': len(manifest_rows) - len(payloads), 'total_cases': len(manifest_rows)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
