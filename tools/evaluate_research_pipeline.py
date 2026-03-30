from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_M

from qr_onboarding.enhanced_pipeline import EnhancedQRSystem
from qr_onboarding.pipeline import QRReader


DEFAULT_SCENARIOS = [
    "clean",
    "strong_blur",
    "very_low_light",
    "tiny_qr",
    "perspective",
    "glare",
    "screen_realistic",
    "combined_hard",
]


def make_payload(sample_idx: int, payload_mode: str) -> tuple[str, str]:
    if payload_mode == "standard":
        payload = {
            "ssid": f"Bench-{sample_idx}",
            "registration-id": f"rid-{sample_idx}",
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "standard"

    if payload_mode == "dense":
        payload = {
            "ssid": f"Bench-{sample_idx}",
            "registration-id": f"rid-{sample_idx}",
            "registration-token": f"token-{sample_idx:04d}-abcdef1234567890",
            "device-model": "qr-research-terminal-v2",
            "notes": "payload-density-check-" + ("XYZ123" * 14),
            "capabilities": [
                "wifi",
                "ble",
                "secure-boot",
                "device-attestation",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "dense"

    dense = sample_idx % 3 == 2
    return make_payload(sample_idx, "dense" if dense else "standard")


def make_clean_qr(text: str, size: int = 256) -> np.ndarray:
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    image = image.resize((size, size))
    return np.array(image)[:, :, ::-1].copy()


def darken_with_noise(image: np.ndarray, factor: float, noise_sigma: float, seed: int) -> np.ndarray:
    out = np.clip(image.astype(np.float32) * factor, 0, 255)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        out += rng.normal(0, noise_sigma, size=image.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_motion_blur(image: np.ndarray, ksize: int = 11, angle_deg: float = 0.0) -> np.ndarray:
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    radius = center

    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta) * radius
    dy = np.sin(theta) * radius

    pt1 = (int(round(center - dx)), int(round(center - dy)))
    pt2 = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, pt1, pt2, 1.0, 1)

    s = float(kernel.sum())
    if s <= 0:
        kernel[center, :] = 1.0
        s = float(kernel.sum())
    kernel /= s
    return cv2.filter2D(image, -1, kernel)


def add_perspective_distortion(image: np.ndarray, strength: float, seed: int, bg: int = 245) -> np.ndarray:
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)

    src = np.float32(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ]
    )

    dx = int(w * strength)
    dy = int(h * strength)

    dst = np.float32(
        [
            [rng.integers(0, dx + 1), rng.integers(0, dy + 1)],
            [w - 1 - rng.integers(0, dx + 1), rng.integers(0, dy + 1)],
            [w - 1 - rng.integers(0, dx + 1), h - 1 - rng.integers(0, dy + 1)],
            [rng.integers(0, dx + 1), h - 1 - rng.integers(0, dy + 1)],
        ]
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(bg, bg, bg),
    )


def add_glare(image: np.ndarray, strength: float = 0.48, seed: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)
    overlay = image.astype(np.float32).copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    center1 = (int(w * rng.uniform(0.56, 0.72)), int(h * rng.uniform(0.26, 0.42)))
    axes1 = (int(w * rng.uniform(0.18, 0.25)), int(h * rng.uniform(0.10, 0.15)))
    angle1 = float(rng.integers(-28, 29))
    cv2.ellipse(mask, center1, axes1, angle1, 0, 360, 255, -1)

    center2 = (int(w * rng.uniform(0.30, 0.46)), int(h * rng.uniform(0.52, 0.68)))
    axes2 = (int(w * rng.uniform(0.10, 0.15)), int(h * rng.uniform(0.06, 0.10)))
    angle2 = float(rng.integers(-28, 29))
    cv2.ellipse(mask, center2, axes2, angle2, 0, 360, 150, -1)

    mask = cv2.GaussianBlur(mask, (0, 0), 19)
    alpha = (mask.astype(np.float32) / 255.0) * strength
    overlay = overlay * (1.0 - alpha[..., None]) + 255.0 * alpha[..., None]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def add_screen_artifacts(image: np.ndarray, seed: int, severity: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)
    out = image.astype(np.float32).copy()

    col_band = np.sin(np.linspace(0, 2 * np.pi * 10, w, dtype=np.float32))[None, :]
    row_band = np.sin(np.linspace(0, 2 * np.pi * 14, h, dtype=np.float32))[:, None]
    out += (col_band * (9.0 * severity))[..., None]
    out += (row_band * (6.0 * severity))[..., None]
    out += rng.normal(0, 3.0 * severity, size=out.shape).astype(np.float32)

    if severity >= 0.9:
        blue = out[:, :, 0].copy()
        red = out[:, :, 2].copy()
        out[:, :, 0] = np.roll(blue, 1, axis=1)
        out[:, :, 2] = np.roll(red, -1, axis=1)

    down = cv2.resize(out, None, fx=0.88, fy=0.88, interpolation=cv2.INTER_AREA)
    out = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_textured_canvas(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    h, w, c = shape
    rng = np.random.default_rng(seed)

    canvas = np.full((h, w, c), 242, dtype=np.uint8)
    noise = rng.normal(0, 6, size=(h, w, c)).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 2.3)

    for _ in range(6):
        color = int(rng.integers(222, 247))
        x1 = int(rng.integers(0, max(1, w - 40)))
        y1 = int(rng.integers(0, max(1, h - 40)))
        x2 = int(min(w - 1, x1 + rng.integers(26, 75)))
        y2 = int(min(h - 1, y1 + rng.integers(18, 62)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (color, color, color), -1)

    for _ in range(8):
        color = int(rng.integers(224, 246))
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        radius = int(rng.integers(6, 18))
        cv2.circle(canvas, center, radius, (color, color, color), -1)

    return canvas


def place_small_qr_on_canvas(image: np.ndarray, scale: float, seed: int) -> np.ndarray:
    h, w = image.shape[:2]
    canvas = make_textured_canvas(image.shape, seed)

    qr = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    y0 = (h - qr.shape[0]) // 2
    x0 = (w - qr.shape[1]) // 2
    canvas[y0:y0 + qr.shape[0], x0:x0 + qr.shape[1]] = qr

    return cv2.GaussianBlur(canvas, (0, 0), 0.8)


def add_background_gradient(image: np.ndarray, x_min: float = 0.72, y_min: float = 0.90) -> np.ndarray:
    h, w = image.shape[:2]
    gradient_x = np.linspace(x_min, 1.0, w, dtype=np.float32)[None, :, None]
    gradient_y = np.linspace(y_min, 1.0, h, dtype=np.float32)[:, None, None]
    out = image.astype(np.float32) * (gradient_x * gradient_y)
    return np.clip(out, 0, 255).astype(np.uint8)


def tune_by_payload(payload_variant: str) -> dict[str, float]:
    if payload_variant == "dense":
        return {
            "blur_sigma": 2.5,
            "motion_ksize": 9,
            "tiny_scale": 0.32,
            "perspective_strength": 0.14,
            "glare_strength": 0.38,
            "dark_factor": 0.24,
            "noise_sigma": 11.0,
            "combined_blur_sigma": 2.0,
            "combined_motion_ksize": 9,
            "combined_dark_factor": 0.33,
            "combined_noise_sigma": 9.0,
            "combined_scale": 0.80,
            "screen_severity": 0.80,
        }

    return {
        "blur_sigma": 3.0,
        "motion_ksize": 11,
        "tiny_scale": 0.29,
        "perspective_strength": 0.17,
        "glare_strength": 0.46,
        "dark_factor": 0.21,
        "noise_sigma": 13.0,
        "combined_blur_sigma": 2.4,
        "combined_motion_ksize": 11,
        "combined_dark_factor": 0.30,
        "combined_noise_sigma": 11.0,
        "combined_scale": 0.84,
        "screen_severity": 1.0,
    }


def apply_scenario(img: np.ndarray, scenario: str, seed: int, payload_variant: str = "standard") -> np.ndarray:
    out = img.copy()
    rng = np.random.default_rng(seed)
    p = tune_by_payload(payload_variant)

    if scenario == "clean":
        return out

    if scenario == "strong_blur":
        out = cv2.GaussianBlur(out, (0, 0), p["blur_sigma"])
        out = add_motion_blur(
            out,
            ksize=int(p["motion_ksize"]),
            angle_deg=float(rng.integers(-18, 19)),
        )
        return out

    if scenario == "very_low_light":
        out = darken_with_noise(
            out,
            factor=p["dark_factor"],
            noise_sigma=p["noise_sigma"],
            seed=seed,
        )
        out = add_background_gradient(out, x_min=0.76, y_min=0.92)
        return out

    if scenario == "tiny_qr":
        out = place_small_qr_on_canvas(out, scale=p["tiny_scale"], seed=seed)
        return out

    if scenario == "perspective":
        out = add_perspective_distortion(out, strength=p["perspective_strength"], seed=seed)
        out = cv2.GaussianBlur(out, (0, 0), 0.6)
        return out

    if scenario == "glare":
        out = add_glare(out, strength=p["glare_strength"], seed=seed)
        return out

    if scenario == "screen_realistic":
        out = add_screen_artifacts(out, seed=seed, severity=p["screen_severity"])
        out = add_glare(out, strength=0.12 if payload_variant == "dense" else 0.16, seed=seed + 11)
        return out

    if scenario == "combined_hard":
        out = add_perspective_distortion(
            out,
            strength=max(0.12, p["perspective_strength"] - 0.02),
            seed=seed,
        )
        out = cv2.GaussianBlur(out, (0, 0), p["combined_blur_sigma"])
        out = add_motion_blur(
            out,
            ksize=int(p["combined_motion_ksize"]),
            angle_deg=float(rng.integers(-14, 15)),
        )
        out = add_glare(out, strength=0.18 if payload_variant == "dense" else 0.24, seed=seed + 29)
        out = darken_with_noise(
            out,
            factor=p["combined_dark_factor"],
            noise_sigma=p["combined_noise_sigma"],
            seed=seed + 97,
        )

        h, w = out.shape[:2]
        smaller = cv2.resize(
            out,
            None,
            fx=p["combined_scale"],
            fy=p["combined_scale"],
            interpolation=cv2.INTER_AREA,
        )
        out = cv2.resize(smaller, (w, h), interpolation=cv2.INTER_CUBIC)
        out = add_screen_artifacts(out, seed=seed + 211, severity=0.55 if payload_variant == "dense" else 0.70)
        return out

    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def bool_to_int(value: Any) -> int:
    return int(bool(value))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_successes = sum(bool_to_int(r["direct_success"]) for r in rows)
    classic_successes = sum(bool_to_int(r["classic_success"]) for r in rows)
    enhanced_successes = sum(bool_to_int(r["enhanced_success"]) for r in rows)

    per_scenario: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "direct_successes": 0,
            "classic_successes": 0,
            "enhanced_successes": 0,
        }
    )
    stage_counter: Counter[str] = Counter()

    for row in rows:
        scenario = str(row["scenario"])
        per_scenario[scenario]["total"] += 1
        per_scenario[scenario]["direct_successes"] += bool_to_int(row["direct_success"])
        per_scenario[scenario]["classic_successes"] += bool_to_int(row["classic_success"])
        per_scenario[scenario]["enhanced_successes"] += bool_to_int(row["enhanced_success"])

        stage = str(row["enhanced_stage"] or "")
        if stage:
            stage_counter[stage] += 1

    for scenario, block in per_scenario.items():
        total = block["total"] or 1
        block["direct_rate"] = round(block["direct_successes"] / total, 4)
        block["classic_rate"] = round(block["classic_successes"] / total, 4)
        block["enhanced_rate"] = round(block["enhanced_successes"] / total, 4)

    return {
        "rows": len(rows),
        "direct_successes": direct_successes,
        "classic_successes": classic_successes,
        "enhanced_successes": enhanced_successes,
        "direct_rate": round(direct_successes / max(1, len(rows)), 4),
        "classic_rate": round(classic_successes / max(1, len(rows)), 4),
        "enhanced_rate": round(enhanced_successes / max(1, len(rows)), 4),
        "per_scenario": dict(per_scenario),
        "top_enhancement_stages": stage_counter.most_common(10),
    }


def maybe_dump_example(
    image: np.ndarray,
    dump_dir: Path | None,
    dump_limit: int,
    sample: int,
    scenario: str,
) -> None:
    if dump_dir is None:
        return
    if sample >= dump_limit:
        return
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / f"sample_{sample:02d}_{scenario}.png"
    cv2.imwrite(str(path), image)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Balanced research benchmark for direct, classic, and enhanced QR reading pipelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--csv", default="research_benchmark.csv")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=DEFAULT_SCENARIOS,
        help="Scenario names to evaluate",
    )
    parser.add_argument(
        "--payload-mode",
        choices=["standard", "dense", "mixed"],
        default="mixed",
        help="Controls payload complexity and QR density",
    )
    parser.add_argument(
        "--dump-dir",
        default="",
        help="Optional directory for saving a few generated benchmark images",
    )
    parser.add_argument(
        "--dump-limit",
        type=int,
        default=3,
        help="How many samples per scenario to save into dump-dir",
    )
    args = parser.parse_args()

    scenarios = list(args.scenarios)
    csv_path = Path(args.csv)
    dump_dir = Path(args.dump_dir) if args.dump_dir else None

    direct_reader = QRReader()
    classic_reader = QRReader()
    enhanced = EnhancedQRSystem()

    rows: list[dict[str, Any]] = []

    for sample_idx in range(args.samples):
        payload_text, payload_variant = make_payload(sample_idx, args.payload_mode)
        clean = make_clean_qr(payload_text, size=args.size)

        for scenario_idx, scenario in enumerate(scenarios):
            scenario_seed = args.seed + sample_idx * 7919 + scenario_idx * 131
            image = apply_scenario(clean, scenario, scenario_seed, payload_variant)
            maybe_dump_example(image, dump_dir, args.dump_limit, sample_idx, scenario)

            direct = direct_reader.scan_image_direct(image)
            classic = classic_reader.scan_image(image)
            enhanced_result = enhanced.scan_image(image)

            base_result = getattr(enhanced_result, "base_result", None)
            enhanced_stage = getattr(enhanced_result, "enhancement_stage", "") or getattr(base_result, "stage", "")
            base_stage = getattr(base_result, "stage", "")
            base_decoder = getattr(base_result, "decoder", "")
            roi_used = bool(getattr(enhanced_result, "roi_used", False))

            row = {
                "sample": sample_idx,
                "scenario": scenario,
                "seed": scenario_seed,
                "payload_variant": payload_variant,
                "payload_length": len(payload_text),
                "direct_success": bool(getattr(direct, "success", False)),
                "classic_success": bool(getattr(classic, "success", False)),
                "enhanced_success": bool(getattr(enhanced_result, "success", False)),
                "enhanced_stage": enhanced_stage,
                "base_stage": base_stage,
                "base_decoder": base_decoder,
                "roi_used": roi_used,
            }
            rows.append(row)

    if not rows:
        print("No benchmark rows were generated.")
        return 1

    ensure_parent(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows)
    summary["csv"] = str(csv_path)
    summary["scenarios"] = scenarios
    summary["payload_mode"] = args.payload_mode
    summary["size"] = args.size
    summary["seed"] = args.seed
    if dump_dir is not None:
        summary["dump_dir"] = str(dump_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
