from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2

from qr_onboarding.enhanced_pipeline import EnhancedQRSystem
from qr_onboarding.pipeline import QRReader


def iter_files(root: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(p for p in root.rglob(pattern) if p.is_file())
    return sorted(p for p in root.glob(pattern) if p.is_file())


def bool_to_int(value: Any) -> int:
    return int(bool(value))


def safe_stage(result: Any) -> str:
    if result is None:
        return ""
    stage = getattr(result, "enhancement_stage", "") or ""
    if stage:
        return stage
    base = getattr(result, "base_result", None)
    return getattr(base, "stage", "") or ""


def safe_decoder(result: Any) -> str:
    if result is None:
        return ""
    base = getattr(result, "base_result", None)
    if base is not None:
        return getattr(base, "decoder", "") or ""
    return getattr(result, "decoder", "") or ""


def has_any_hit(enhanced: EnhancedQRSystem, result: Any) -> bool:
    if result is None:
        return False
    try:
        return bool(enhanced._has_any_qr_hit(result))
    except Exception:
        return bool(getattr(result, "success", False))


def classify_path(root: Path, path: Path) -> tuple[str, str]:
    try:
        rel = path.relative_to(root)
    except Exception:
        return "unknown_dataset", path.parent.name or "unknown_scenario"

    parts = rel.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return root.name, parts[0]
    if len(parts) == 1:
        return root.name, path.parent.name or parts[0]
    return root.name, path.parent.name or "unknown_scenario"


def stage_family(stage: str) -> str:
    s = (stage or "").lower().strip()
    if not s:
        return ""

    if "watermark" in s or "screen" in s or "moire" in s or "halftone" in s:
        return "screen_print_cleanup"
    if "gamma" in s or "clahe" in s or "equalized" in s or "glare" in s or "contrast" in s:
        return "illumination_contrast_recovery"
    if "upscaled" in s or "sharp" in s or "detail" in s or "blur" in s:
        return "blur_small_qr_recovery"
    if (
        "adaptive" in s
        or "otsu" in s
        or "niblack" in s
        or "yao" in s
        or "di" in s
        or "proposed_integral" in s
    ):
        return "adaptive_binarization"
    if "rect" in s or "perspective" in s or "warp" in s:
        return "geometry_recovery"
    if "ml" in s or "research" in s or "neural" in s:
        return "ml_recovery"
    if s == "gray":
        return "minimal_preprocessing"
    return "other_custom_stage"


def is_novelty_stage(stage: str) -> bool:
    fam = stage_family(stage)
    s = (stage or "").lower().strip()
    if not s:
        return False
    if s == "gray":
        return False
    if fam == "minimal_preprocessing":
        return False
    return True


def to_bgr_gray(image: Any) -> Any:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def timed_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float, str]:
    started = perf_counter()
    try:
        result = fn(*args, **kwargs)
        err = ""
    except Exception as exc:
        result = None
        err = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (perf_counter() - started) * 1000.0
    return result, elapsed_ms, err


def make_failure_row(root: Path, path: Path, reason: str) -> dict[str, Any]:
    dataset_name, scenario_group = classify_path(root, path)
    return {
        "file": str(path.relative_to(root)),
        "dataset_name": dataset_name,
        "scenario_group": scenario_group,
        "selector_scenario": "",
        "raw_success": False,
        "raw_stage": "",
        "raw_decoder": "",
        "raw_decode_ms": 0.0,
        "raw_error": reason,
        "minimal_success": False,
        "minimal_stage": "",
        "minimal_decoder": "",
        "minimal_decode_ms": 0.0,
        "minimal_error": reason,
        "classic_success": False,
        "classic_stage": "",
        "classic_decoder": "",
        "classic_decode_ms": 0.0,
        "classic_error": reason,
        "switch_only_success": False,
        "switch_only_hit": False,
        "switch_only_stage": "",
        "switch_only_family": "",
        "switch_only_ms": 0.0,
        "switch_only_error": reason,
        "ml_only_success": False,
        "ml_only_hit": False,
        "ml_only_stage": "",
        "ml_only_family": "",
        "ml_only_ms": 0.0,
        "ml_only_error": reason,
        "custom_only_success": False,
        "custom_only_hit": False,
        "custom_only_stage": "",
        "custom_only_family": "",
        "custom_only_ms": 0.0,
        "custom_only_error": reason,
        "enhanced_success": False,
        "enhanced_hit": False,
        "enhanced_stage": "",
        "enhanced_family": "",
        "enhanced_decoder": "",
        "enhanced_roi_used": False,
        "enhanced_partial_success": False,
        "enhanced_decode_ms": 0.0,
        "enhanced_error": reason,
        "novelty_only_hit": False,
        "novelty_beats_raw": False,
        "novelty_beats_minimal": False,
        "novelty_beats_classic": False,
        "custom_beats_raw": False,
        "custom_beats_minimal": False,
        "custom_beats_classic": False,
        "enhanced_beats_raw": False,
        "enhanced_beats_minimal": False,
        "enhanced_beats_classic": False,
    }


def make_row(
    *,
    root: Path,
    path: Path,
    raw_result: Any,
    raw_ms: float,
    raw_error: str,
    minimal_result: Any,
    minimal_ms: float,
    minimal_error: str,
    classic_result: Any,
    classic_ms: float,
    classic_error: str,
    switch_result: Any,
    switch_ms: float,
    switch_hit: bool,
    switch_error: str,
    ml_result: Any,
    ml_ms: float,
    ml_hit: bool,
    ml_error: str,
    custom_result: Any,
    custom_ms: float,
    custom_hit: bool,
    custom_error: str,
    enhanced_result: Any,
    enhanced_ms: float,
    enhanced_hit: bool,
    enhanced_error: str,
    selector_scenario: str,
) -> dict[str, Any]:
    rel = str(path.relative_to(root))
    dataset_name, scenario_group = classify_path(root, path)

    raw_success = bool(getattr(raw_result, "success", False))
    minimal_success = bool(getattr(minimal_result, "success", False))
    classic_success = bool(getattr(classic_result, "success", False))
    custom_success = bool(getattr(custom_result, "success", False))
    enhanced_success = bool(getattr(enhanced_result, "success", False))

    custom_stage = safe_stage(custom_result)
    enhanced_stage = safe_stage(enhanced_result)
    custom_family = stage_family(custom_stage)
    novelty_hit = custom_hit and is_novelty_stage(custom_stage)

    return {
        "file": rel,
        "dataset_name": dataset_name,
        "scenario_group": scenario_group,
        "selector_scenario": selector_scenario,
        "raw_success": raw_success,
        "raw_stage": getattr(raw_result, "stage", "") or "",
        "raw_decoder": getattr(raw_result, "decoder", "") or "",
        "raw_decode_ms": round(raw_ms, 2),
        "raw_error": raw_error,
        "minimal_success": minimal_success,
        "minimal_stage": getattr(minimal_result, "stage", "") or "",
        "minimal_decoder": getattr(minimal_result, "decoder", "") or "",
        "minimal_decode_ms": round(minimal_ms, 2),
        "minimal_error": minimal_error,
        "classic_success": classic_success,
        "classic_stage": getattr(classic_result, "stage", "") or "",
        "classic_decoder": getattr(classic_result, "decoder", "") or "",
        "classic_decode_ms": round(classic_ms, 2),
        "classic_error": classic_error,
        "switch_only_success": bool(getattr(switch_result, "success", False)),
        "switch_only_hit": switch_hit,
        "switch_only_stage": safe_stage(switch_result),
        "switch_only_family": stage_family(safe_stage(switch_result)),
        "switch_only_ms": round(switch_ms, 2),
        "switch_only_error": switch_error,
        "ml_only_success": bool(getattr(ml_result, "success", False)),
        "ml_only_hit": ml_hit,
        "ml_only_stage": safe_stage(ml_result),
        "ml_only_family": stage_family(safe_stage(ml_result)),
        "ml_only_ms": round(ml_ms, 2),
        "ml_only_error": ml_error,
        "custom_only_success": custom_success,
        "custom_only_hit": custom_hit,
        "custom_only_stage": custom_stage,
        "custom_only_family": custom_family,
        "custom_only_decoder": safe_decoder(custom_result),
        "custom_only_ms": round(custom_ms, 2),
        "custom_only_error": custom_error,
        "enhanced_success": enhanced_success,
        "enhanced_hit": enhanced_hit,
        "enhanced_stage": enhanced_stage,
        "enhanced_family": stage_family(enhanced_stage),
        "enhanced_decoder": safe_decoder(enhanced_result),
        "enhanced_roi_used": bool(getattr(enhanced_result, "roi_used", False)),
        "enhanced_partial_success": bool(getattr(enhanced_result, "partial_success", False)),
        "enhanced_decode_ms": round(enhanced_ms, 2),
        "enhanced_error": enhanced_error,
        "novelty_only_hit": novelty_hit,
        "novelty_beats_raw": novelty_hit and not raw_success,
        "novelty_beats_minimal": novelty_hit and not minimal_success,
        "novelty_beats_classic": novelty_hit and not classic_success,
        "custom_beats_raw": custom_hit and not raw_success,
        "custom_beats_minimal": custom_hit and not minimal_success,
        "custom_beats_classic": custom_hit and not classic_success,
        "enhanced_beats_raw": enhanced_hit and not raw_success,
        "enhanced_beats_minimal": enhanced_hit and not minimal_success,
        "enhanced_beats_classic": enhanced_hit and not classic_success,
    }


def add_case_preview(rows: list[dict[str, Any]], predicate: str, limit: int = 20) -> list[dict[str, Any]]:
    picked = []
    for row in rows:
        if not row.get(predicate):
            continue
        picked.append(
            {
                "file": row["file"],
                "scenario_group": row["scenario_group"],
                "selector_scenario": row["selector_scenario"],
                "stage": row["custom_only_stage"] or row["enhanced_stage"],
                "family": row["custom_only_family"] or row["enhanced_family"],
                "decoder": row["custom_only_decoder"] or row["enhanced_decoder"],
            }
        )
        if len(picked) >= limit:
            break
    return picked


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "files": len(rows),
        "raw_successes": sum(bool_to_int(r["raw_success"]) for r in rows),
        "minimal_successes": sum(bool_to_int(r["minimal_success"]) for r in rows),
        "classic_successes": sum(bool_to_int(r["classic_success"]) for r in rows),
        "switch_only_hits": sum(bool_to_int(r["switch_only_hit"]) for r in rows),
        "ml_only_hits": sum(bool_to_int(r["ml_only_hit"]) for r in rows),
        "custom_only_hits": sum(bool_to_int(r["custom_only_hit"]) for r in rows),
        "enhanced_hits": sum(bool_to_int(r["enhanced_hit"]) for r in rows),
        "novelty_only_hits": sum(bool_to_int(r["novelty_only_hit"]) for r in rows),
        "novelty_beats_raw": sum(bool_to_int(r["novelty_beats_raw"]) for r in rows),
        "novelty_beats_minimal": sum(bool_to_int(r["novelty_beats_minimal"]) for r in rows),
        "novelty_beats_classic": sum(bool_to_int(r["novelty_beats_classic"]) for r in rows),
        "custom_beats_raw": sum(bool_to_int(r["custom_beats_raw"]) for r in rows),
        "custom_beats_minimal": sum(bool_to_int(r["custom_beats_minimal"]) for r in rows),
        "custom_beats_classic": sum(bool_to_int(r["custom_beats_classic"]) for r in rows),
        "enhanced_beats_raw": sum(bool_to_int(r["enhanced_beats_raw"]) for r in rows),
        "enhanced_beats_minimal": sum(bool_to_int(r["enhanced_beats_minimal"]) for r in rows),
        "enhanced_beats_classic": sum(bool_to_int(r["enhanced_beats_classic"]) for r in rows),
    }

    total = max(1, len(rows))
    summary["raw_rate"] = round(summary["raw_successes"] / total, 4)
    summary["minimal_rate"] = round(summary["minimal_successes"] / total, 4)
    summary["classic_rate"] = round(summary["classic_successes"] / total, 4)
    summary["switch_only_hit_rate"] = round(summary["switch_only_hits"] / total, 4)
    summary["ml_only_hit_rate"] = round(summary["ml_only_hits"] / total, 4)
    summary["custom_only_hit_rate"] = round(summary["custom_only_hits"] / total, 4)
    summary["enhanced_hit_rate"] = round(summary["enhanced_hits"] / total, 4)
    summary["novelty_only_hit_rate"] = round(summary["novelty_only_hits"] / total, 4)

    by_group = defaultdict(lambda: {
        "total": 0,
        "raw_successes": 0,
        "minimal_successes": 0,
        "classic_successes": 0,
        "custom_only_hits": 0,
        "enhanced_hits": 0,
        "novelty_only_hits": 0,
        "novelty_beats_minimal": 0,
        "novelty_beats_classic": 0,
    })

    novelty_stage_counter = Counter()
    novelty_family_counter = Counter()

    for row in rows:
        group = row["scenario_group"]
        by_group[group]["total"] += 1
        by_group[group]["raw_successes"] += bool_to_int(row["raw_success"])
        by_group[group]["minimal_successes"] += bool_to_int(row["minimal_success"])
        by_group[group]["classic_successes"] += bool_to_int(row["classic_success"])
        by_group[group]["custom_only_hits"] += bool_to_int(row["custom_only_hit"])
        by_group[group]["enhanced_hits"] += bool_to_int(row["enhanced_hit"])
        by_group[group]["novelty_only_hits"] += bool_to_int(row["novelty_only_hit"])
        by_group[group]["novelty_beats_minimal"] += bool_to_int(row["novelty_beats_minimal"])
        by_group[group]["novelty_beats_classic"] += bool_to_int(row["novelty_beats_classic"])

        if row["novelty_only_hit"]:
            st = row["custom_only_stage"]
            fam = row["custom_only_family"]
            if st:
                novelty_stage_counter[st] += 1
            if fam:
                novelty_family_counter[fam] += 1

    for group, block in by_group.items():
        n = max(1, block["total"])
        block["raw_rate"] = round(block["raw_successes"] / n, 4)
        block["minimal_rate"] = round(block["minimal_successes"] / n, 4)
        block["classic_rate"] = round(block["classic_successes"] / n, 4)
        block["custom_only_hit_rate"] = round(block["custom_only_hits"] / n, 4)
        block["enhanced_hit_rate"] = round(block["enhanced_hits"] / n, 4)
        block["novelty_only_hit_rate"] = round(block["novelty_only_hits"] / n, 4)

    summary["by_group"] = dict(by_group)
    summary["top_novelty_stages"] = novelty_stage_counter.most_common(12)
    summary["top_novelty_families"] = novelty_family_counter.most_common(12)
    summary["novelty_beats_minimal_cases"] = add_case_preview(rows, "novelty_beats_minimal", limit=20)
    summary["novelty_beats_classic_cases"] = add_case_preview(rows, "novelty_beats_classic", limit=20)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Novelty-oriented ablation benchmark for raw, minimal, classic, custom-only, and full enhanced QR pipelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="Root directory with benchmark images")
    parser.add_argument("--pattern", default="**/*.png", help="Glob pattern for images")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for matching files")
    parser.add_argument("--csv", default="results/ablation_results.csv", help="Output CSV path")
    parser.add_argument("--summary-json", default="results/ablation_summary.json", help="Output summary JSON path")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of files")
    args = parser.parse_args()

    root = Path(args.dataset).resolve()
    files = iter_files(root, args.pattern, args.recursive or ("**" in args.pattern))
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        print(json.dumps({"error": f"No files matched in {root} with pattern {args.pattern}"}, ensure_ascii=False, indent=2))
        return 1

    csv_path = Path(args.csv)
    summary_path = Path(args.summary_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    direct_reader = QRReader()
    classic_reader = QRReader()
    rows = []

    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append(make_failure_row(root, path, "Image could not be loaded"))
            continue

        enhanced = EnhancedQRSystem()
        gray_bgr = to_bgr_gray(image)

        raw_result, raw_ms, raw_error = timed_call(direct_reader.scan_image_direct, image)
        minimal_result, minimal_ms, minimal_error = timed_call(direct_reader.scan_image_direct, gray_bgr)
        classic_result, classic_ms, classic_error = timed_call(classic_reader.scan_image, image)

        try:
            thresholds = enhanced.calibrator.thresholds()
            selector_scenario, scenario_notes, points, _quality = enhanced.selector.classify(image, thresholds)
            notes = list(scenario_notes)
        except Exception:
            selector_scenario = ""
            notes = []
            points = None

        switch_result, switch_ms, switch_error = timed_call(
            enhanced._scan_candidate_order,
            image,
            notes,
            selector_scenario,
            False,
            points,
        )
        switch_hit = has_any_hit(enhanced, switch_result)

        ml_result, ml_ms, ml_error = timed_call(
            enhanced._scan_ml_stages,
            image,
            notes,
            selector_scenario,
            False,
        )
        ml_hit = has_any_hit(enhanced, ml_result)

        if switch_hit:
            custom_result = switch_result
            custom_ms = switch_ms
            custom_hit = switch_hit
            custom_error = switch_error
        else:
            custom_result = ml_result
            custom_ms = switch_ms + ml_ms
            custom_hit = ml_hit
            custom_error = "; ".join(x for x in [switch_error, ml_error] if x)

        enhanced_result, enhanced_ms, enhanced_error = timed_call(enhanced.scan_image, image)
        enhanced_hit = has_any_hit(enhanced, enhanced_result)

        rows.append(
            make_row(
                root=root,
                path=path,
                raw_result=raw_result,
                raw_ms=raw_ms,
                raw_error=raw_error,
                minimal_result=minimal_result,
                minimal_ms=minimal_ms,
                minimal_error=minimal_error,
                classic_result=classic_result,
                classic_ms=classic_ms,
                classic_error=classic_error,
                switch_result=switch_result,
                switch_ms=switch_ms,
                switch_hit=switch_hit,
                switch_error=switch_error,
                ml_result=ml_result,
                ml_ms=ml_ms,
                ml_hit=ml_hit,
                ml_error=ml_error,
                custom_result=custom_result,
                custom_ms=custom_ms,
                custom_hit=custom_hit,
                custom_error=custom_error,
                enhanced_result=enhanced_result,
                enhanced_ms=enhanced_ms,
                enhanced_hit=enhanced_hit,
                enhanced_error=enhanced_error,
                selector_scenario=selector_scenario,
            )
        )

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["dataset"] = str(root)
    summary["csv"] = str(csv_path)
    summary["summary_json"] = str(summary_path)

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
