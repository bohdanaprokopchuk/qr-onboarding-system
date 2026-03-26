from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict

import cv2
import numpy as np
import qrcode

from qr_onboarding.enhanced_pipeline import EnhancedQRSystem
from qr_onboarding.pipeline import QRReader


def make_clean_qr(text: str, size: int = 256) -> np.ndarray:
    return np.array(qrcode.make(text).convert('RGB').resize((size, size)))[:, :, ::-1].copy()


def apply_scenario(img: np.ndarray, scenario: str, seed: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    if scenario == 'blur':
        return cv2.GaussianBlur(out, (0, 0), 2.2)
    if scenario == 'low_light':
        return np.clip(out.astype(np.float32) * 0.33, 0, 255).astype(np.uint8)
    if scenario == 'small_qr':
        canvas = np.full_like(out, 255)
        scale = 0.42
        qr = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        y0 = (h - qr.shape[0]) // 2
        x0 = (w - qr.shape[1]) // 2
        canvas[y0:y0 + qr.shape[0], x0:x0 + qr.shape[1]] = qr
        return canvas
    if scenario == 'combined':
        out = cv2.GaussianBlur(out, (0, 0), 1.7)
        out = np.clip(out.astype(np.float32) * 0.4 + np.random.default_rng(seed).normal(0, 14, out.shape), 0, 255).astype(np.uint8)
        small = cv2.resize(out, None, fx=0.62, fy=0.62, interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return out
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=6)
    p.add_argument('--csv', default='research_benchmark.csv')
    args = p.parse_args()
    direct_reader = QRReader(); classic_reader = QRReader(); enhanced = EnhancedQRSystem(); scenarios = ['clean', 'blur', 'low_light', 'small_qr', 'combined']
    rows = []; enhanced_stage_counter: Counter[str] = Counter(); grouped = defaultdict(lambda: {'direct': 0, 'classic': 0, 'enhanced': 0, 'total': 0})
    for i in range(args.samples):
        payload = f'{{"ssid":"Bench-{i}","registration-id":"rid-{i}"}}'; clean = make_clean_qr(payload)
        for scenario in scenarios:
            image = clean if scenario == 'clean' else apply_scenario(clean, scenario, i * 13 + len(scenario))
            direct = direct_reader.scan_image_direct(image); classic = classic_reader.scan_image(image); enhanced_result = enhanced.scan_image(image)
            stage = enhanced_result.enhancement_stage or (enhanced_result.base_result.stage if enhanced_result.base_result else '')
            rows.append({'sample': i, 'scenario': scenario, 'direct_success': direct.success, 'classic_success': classic.success, 'enhanced_success': enhanced_result.success, 'enhanced_stage': stage, 'roi_used': enhanced_result.roi_used})
            grouped[scenario]['total'] += 1; grouped[scenario]['direct'] += int(direct.success); grouped[scenario]['classic'] += int(classic.success); grouped[scenario]['enhanced'] += int(enhanced_result.success)
            if stage: enhanced_stage_counter[stage] += 1
    with open(args.csv, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    print({'samples': len(rows), 'direct_successes': sum(int(r['direct_success']) for r in rows), 'classic_successes': sum(int(r['classic_success']) for r in rows), 'enhanced_successes': sum(int(r['enhanced_success']) for r in rows), 'per_scenario': dict(grouped), 'top_enhancement_stages': enhanced_stage_counter.most_common(8), 'csv': args.csv})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
