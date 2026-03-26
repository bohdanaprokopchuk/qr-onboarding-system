from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

import cv2
import numpy as np

from .adaptive_thresholds import AdaptiveThresholdCalibrator, CalibratedThresholds
from .ml_models import MLEnhancer
from .models import ScanResult
from .multi_frame import MultiFrameBuffer
from .payload_codecs import PayloadError, classify_text_payload, decode_versioned_payload
from .pipeline import QRReader
from .pipeline_stats import PipelineStatsCollector
from .preprocessing import build_candidates, evaluate_quality, screen_artifact_score
from .provisioning import ProvisioningManager
from .roi_tracking import QRROITracker
from .split_qr import SplitQRAssembler


@dataclass
class EnhancedScanResult:
    success: bool
    base_result: Optional[ScanResult] = None
    assembled: Optional[dict[str, Any]] = None
    provisioned: Optional[dict[str, Any]] = None
    enhancement_stage: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None
    scenario: Optional[str] = None
    roi_used: bool = False
    camera_adaptation: Optional[dict[str, Any]] = None
    split_progress: Optional[str] = None
    partial_success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'success': self.success,
            'base_result': None if not self.base_result else self.base_result.to_dict(),
            'assembled': self.assembled,
            'provisioned': self.provisioned,
            'enhancement_stage': self.enhancement_stage,
            'notes': self.notes,
            'error': self.error,
            'scenario': self.scenario,
            'roi_used': self.roi_used,
            'camera_adaptation': self.camera_adaptation,
            'split_progress': self.split_progress,
            'partial_success': self.partial_success,
        }


class OnlinePipelineSelector:
    STATIC_PRIORITY: dict[str, list[str]] = {
        'screen_capture': ['watermark_suppressed', 'screen_clean', 'screen_sharp', 'screen_proposed_integral', 'watermark_proposed_integral', 'dynamic_equalized', 'clahe_sharp'],
        'low_light': ['proposed_integral', 'dynamic_equalized', 'clahe', 'clahe_sharp', 'adaptive', 'yao', 'adaptive_sharp', 'upscaled'],
        'motion_or_defocus': ['median', 'sharp', 'screen_sharp', 'clahe_sharp', 'proposed_integral', 'upscaled'],
        'small_qr': ['upscaled', 'upscaled_adaptive', 'screen_proposed_integral', 'proposed_integral', 'clahe_sharp', 'adaptive'],
        'oversized_qr': ['gray', 'sharp', 'otsu', 'clahe', 'rectified', 'rectified_clahe', 'rectified_proposed_integral'],
        'glare_or_low_contrast': ['watermark_suppressed', 'screen_clean', 'dynamic_equalized', 'clahe_sharp', 'otsu', 'proposed_integral'],
        'distance_or_soft_focus': ['upscaled', 'upscaled_adaptive', 'screen_proposed_integral', 'median', 'clahe_sharp', 'proposed_integral'],
        'balanced': ['gray', 'proposed_integral', 'watermark_suppressed', 'clahe_sharp', 'adaptive', 'upscaled'],
    }

    def __init__(self, reader: QRReader, stats: PipelineStatsCollector) -> None:
        self.reader = reader
        self.stats = stats

    def classify(self, image: np.ndarray, thresholds: CalibratedThresholds | None = None) -> tuple[str, list[str], Any | None, Any]:
        t = thresholds or CalibratedThresholds()
        points = self.reader.locator.detect(image)
        quality = evaluate_quality(image, points)
        artifact_score = screen_artifact_score(image)
        scenario = 'balanced'
        notes: list[str] = []

        if artifact_score >= 11.5:
            scenario = 'screen_capture'
            notes.append('screen-like moire or watermark artifacts detected')
        if quality.mean_brightness < t.low_light_brightness:
            scenario = 'low_light'
            notes.append(f'low-light detected ({quality.mean_brightness:.1f} < {t.low_light_brightness:.1f})')
        if quality.laplacian_variance < t.motion_defocus_sharpness:
            scenario = 'motion_or_defocus'
            notes.append(f'low sharpness detected ({quality.laplacian_variance:.1f} < {t.motion_defocus_sharpness:.1f})')
        if quality.projected_qr_size_px is not None and quality.projected_qr_size_px < t.small_qr_size_px:
            scenario = 'small_qr'
            notes.append(f'small QR projection detected ({quality.projected_qr_size_px:.0f}px)')
        if quality.qr_area_ratio is not None and quality.qr_area_ratio > 0.45:
            scenario = 'oversized_qr'
            notes.append('oversized QR crop detected')
        if quality.mean_brightness > t.glare_brightness and quality.contrast_stddev < t.glare_low_contrast:
            scenario = 'glare_or_low_contrast'
            notes.append('glare or low local contrast detected')
        if points is None and quality.laplacian_variance < 120.0 and quality.mean_brightness > 115.0:
            scenario = 'distance_or_soft_focus'
            notes.append('no locator match, trying distance or soft-focus strategy')
        if t.is_calibrated:
            notes.append('using calibrated thresholds')
        return scenario, notes, points, quality

    def candidate_order(self, image: np.ndarray, scenario: str, points: Any | None) -> list[tuple[str, np.ndarray]]:
        candidates = build_candidates(image, points)
        static_order = self.STATIC_PRIORITY.get(scenario, self.STATIC_PRIORITY['balanced'])
        names = self.stats.top_stages(scenario, fallback=static_order)
        ordered: list[tuple[str, np.ndarray]] = []
        seen: set[str] = set()
        for target in names:
            for name, candidate in candidates:
                if name == target and name not in seen:
                    ordered.append((name, candidate))
                    seen.add(name)
        for name, candidate in candidates:
            if name not in seen:
                ordered.append((name, candidate))
                seen.add(name)
        return ordered


class EnhancedQRSystem:
    def __init__(
        self,
        private_key: Optional[str] = None,
        ml_enhancer: Optional[MLEnhancer] = None,
        provisioning_manager: Optional[ProvisioningManager] = None,
        frame_buffer_size: int = 8,
        roi_padding: int = 32,
        stats_path: Optional[str] = None,
        calibration_warmup: int = 40,
        adapt_after: int = 15,
    ) -> None:
        self.reader = QRReader(private_key=private_key)
        self.ml_enhancer = ml_enhancer or MLEnhancer()
        self.provisioning_manager = provisioning_manager
        self.frames = MultiFrameBuffer(maxlen=frame_buffer_size)
        self.assembler = SplitQRAssembler()
        self.roi_tracker = QRROITracker(padding=roi_padding)
        self.stats = PipelineStatsCollector(adapt_after=adapt_after)
        if stats_path:
            self.stats.load(stats_path)
        self._stats_path = stats_path
        self.calibrator = AdaptiveThresholdCalibrator(warmup_frames=calibration_warmup)
        self.selector = OnlinePipelineSelector(self.reader, self.stats)

    @staticmethod
    def _merge_notes(*groups: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for note in group:
                if note not in seen:
                    merged.append(note)
                    seen.add(note)
        return merged

    def _maybe_save_stats(self) -> None:
        if not self._stats_path:
            return
        try:
            self.stats.save(self._stats_path)
        except Exception:
            return

    def _post(
        self,
        result: ScanResult,
        notes: list[str],
        scenario: str | None = None,
        roi_used: bool = False,
        camera_adaptation: dict[str, Any] | None = None,
    ) -> EnhancedScanResult:
        if not result.success or not result.parsed_payload:
            return EnhancedScanResult(
                success=False,
                base_result=result,
                notes=list(notes),
                error=result.error or 'Decode failed',
                scenario=scenario,
                roi_used=roi_used,
                camera_adaptation=camera_adaptation,
            )

        payload = result.parsed_payload
        assembled_meta = None
        split_progress_line: Optional[str] = None

        if payload.payload_kind == 'split-chunk':
            chunk_text = result.decoded_text if (result.decoded_text and result.decoded_text.startswith('QRC1|')) else bytes.fromhex(result.raw_hex or '').decode('utf-8', errors='ignore')
            assembled = self.assembler.add_chunk_text(chunk_text)
            session_id = payload.normalized.get('session_id')
            progress = self.assembler.progress(session_id) if session_id else None
            if progress is not None:
                split_progress_line = progress.status_line()
                notes = self._merge_notes(notes, [split_progress_line])
            if assembled is None:
                return EnhancedScanResult(
                    success=False,
                    partial_success=True,
                    base_result=result,
                    notes=list(notes),
                    enhancement_stage='chunk-buffering',
                    error='Need more chunks',
                    scenario=scenario,
                    roi_used=roi_used,
                    camera_adaptation=camera_adaptation,
                    split_progress=split_progress_line,
                )
            try:
                final = decode_versioned_payload(assembled.payload, private_key=self.reader.private_key)
            except PayloadError:
                final = classify_text_payload(assembled.payload.decode('utf-8', errors='ignore'))
            result.parsed_payload = final
            try:
                result.decoded_text = json.dumps(final.normalized, ensure_ascii=False)
            except Exception:
                pass
            completion_note = 'Split-QR chunks assembled successfully'
            if assembled.used_parity and assembled.recovered_indices:
                completion_note += f'; recovered chunk(s): {assembled.recovered_indices}'
            notes = self._merge_notes(notes, [completion_note])
            assembled_meta = {
                'session_id': assembled.session_id,
                'total': assembled.total,
                'digest': assembled.digest,
                'used_parity': assembled.used_parity,
                'recovered_indices': assembled.recovered_indices,
            }

        provisioned = None
        if self.provisioning_manager:
            try:
                provisioned = self.provisioning_manager.provision(result.parsed_payload.normalized)
                notes = self._merge_notes(notes, ['Provisioning pipeline executed'])
            except Exception as exc:
                notes = self._merge_notes(notes, [f'Provisioning failed: {exc}'])

        return EnhancedScanResult(
            success=True,
            base_result=result,
            assembled=assembled_meta,
            provisioned=provisioned,
            notes=list(notes),
            scenario=scenario,
            roi_used=roi_used,
            camera_adaptation=camera_adaptation,
            split_progress=split_progress_line,
        )

    def _scan_direct(self, image: np.ndarray, notes: list[str], scenario: str, roi_used: bool = False) -> tuple[EnhancedScanResult, float]:
        t0 = perf_counter()
        base = self.reader.scan_image_direct(image)
        elapsed_ms = (perf_counter() - t0) * 1000.0
        if base.success:
            result = self._post(base, self._merge_notes(notes, ['Direct decode succeeded']), scenario=scenario, roi_used=roi_used)
            result.enhancement_stage = result.enhancement_stage or 'direct'
            return result, elapsed_ms
        return EnhancedScanResult(
            success=False,
            base_result=base,
            notes=list(notes),
            error=base.error or 'Direct decode failed',
            scenario=scenario,
            roi_used=roi_used,
        ), elapsed_ms

    def _scan_candidate_order(
        self,
        image: np.ndarray,
        notes: list[str],
        scenario: str,
        roi_used: bool = False,
        points: Any | None = None,
    ) -> EnhancedScanResult:
        attempts: list[str] = []
        for name, candidate in self.selector.candidate_order(image, scenario, points):
            stage_input = candidate if candidate.ndim == 3 else cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR)
            t0 = perf_counter()
            trial = self.reader.scan_image_direct(stage_input)
            elapsed_ms = (perf_counter() - t0) * 1000.0
            attempts.append(name)
            if trial.success:
                out = self._post(trial, self._merge_notes(notes, [f'Decode succeeded after online pipeline switch: {name}']), scenario=scenario, roi_used=roi_used)
                out.enhancement_stage = name
                if out.success:
                    self.stats.record_win(scenario, name, latency_ms=elapsed_ms)
                return out
        exhausted = f'Online pipeline candidates exhausted: {", ".join(attempts[:8])}' if attempts else 'Online pipeline candidates exhausted'
        return EnhancedScanResult(
            success=False,
            notes=self._merge_notes(notes, [exhausted]),
            error='Online pipeline switch failed',
            scenario=scenario,
            roi_used=roi_used,
        )

    def _scan_ml_stages(self, image: np.ndarray, notes: list[str], scenario: str, roi_used: bool = False) -> EnhancedScanResult:
        enh = self.ml_enhancer.enhance(image)
        candidates = [
            ('ml_deblur', cv2.cvtColor(enh.deblurred, cv2.COLOR_GRAY2BGR)),
            ('ml_super_res', cv2.cvtColor(enh.super_res, cv2.COLOR_GRAY2BGR)),
            ('ml_masked_super_res', cv2.cvtColor(enh.masked_super_res, cv2.COLOR_GRAY2BGR)),
        ]
        for stage, candidate in candidates:
            t0 = perf_counter()
            trial = self.reader.scan_image_direct(candidate)
            elapsed_ms = (perf_counter() - t0) * 1000.0
            if trial.success:
                out = self._post(trial, self._merge_notes(notes, [f'Decode succeeded after {stage}']), scenario=scenario, roi_used=roi_used)
                out.enhancement_stage = stage
                if out.success:
                    self.stats.record_win(scenario, stage, latency_ms=elapsed_ms)
                return out
        return EnhancedScanResult(
            success=False,
            notes=self._merge_notes(notes, ['ML enhancement stages exhausted']),
            error='Enhanced decode failed',
            scenario=scenario,
            roi_used=roi_used,
        )

    def _scan_with_roi(self, frame: np.ndarray, notes: list[str], scenario: str) -> EnhancedScanResult:
        roi, offset = self.roi_tracker.crop(frame)
        if offset == (0, 0) and roi.shape == frame.shape:
            return EnhancedScanResult(False, notes=list(notes), error='ROI unavailable', scenario=scenario)
        direct, latency_ms = self._scan_direct(roi, notes, scenario, roi_used=True)
        if self._has_any_qr_hit(direct):
            if direct.base_result is not None and direct.base_result.polygon is not None:
                direct.base_result.polygon = self.roi_tracker.remap_polygon(direct.base_result.polygon, offset)
                self.roi_tracker.update(direct.base_result.polygon)
            direct.roi_used = True
            if direct.success:
                self.stats.record_win(scenario, 'direct', latency_ms=latency_ms)
            direct.notes = self._merge_notes(direct.notes, ['Decode recovered inside tracked ROI'])
            return direct
        switched = self._scan_candidate_order(roi, notes, scenario, roi_used=True)
        if self._has_any_qr_hit(switched):
            if switched.base_result is not None and switched.base_result.polygon is not None:
                switched.base_result.polygon = self.roi_tracker.remap_polygon(switched.base_result.polygon, offset)
                self.roi_tracker.update(switched.base_result.polygon)
            switched.roi_used = True
            switched.notes = self._merge_notes(switched.notes, ['Decode recovered inside tracked ROI'])
            return switched
        return EnhancedScanResult(False, notes=self._merge_notes(notes, switched.notes), error='ROI decode failed', scenario=scenario, roi_used=True)

    @staticmethod
    def _has_any_qr_hit(result: EnhancedScanResult) -> bool:
        return bool(result.success or result.partial_success or (result.base_result is not None and result.base_result.success))

    def scan_image(self, image: np.ndarray) -> EnhancedScanResult:
        thresholds = self.calibrator.thresholds()
        scenario, scenario_notes, points, _quality = self.selector.classify(image, thresholds)
        notes = list(scenario_notes)

        if self.roi_tracker.state is not None:
            roi_result = self._scan_with_roi(image, notes, scenario)
            if self._has_any_qr_hit(roi_result):
                return roi_result
            notes = self._merge_notes(notes, roi_result.notes)

        direct, latency_ms = self._scan_direct(image, notes, scenario)
        if self._has_any_qr_hit(direct):
            if direct.base_result is not None and direct.base_result.polygon is not None:
                self.roi_tracker.update(direct.base_result.polygon)
            if direct.success:
                self.stats.record_win(scenario, 'direct', latency_ms=latency_ms)
            return direct

        notes = self._merge_notes(notes, direct.notes)
        switched = self._scan_candidate_order(image, notes, scenario, points=points)
        if self._has_any_qr_hit(switched):
            if switched.base_result is not None and switched.base_result.polygon is not None:
                self.roi_tracker.update(switched.base_result.polygon)
            return switched

        notes = self._merge_notes(notes, switched.notes)
        ml_result = self._scan_ml_stages(image, notes, scenario)
        if self._has_any_qr_hit(ml_result):
            if ml_result.base_result is not None and ml_result.base_result.polygon is not None:
                self.roi_tracker.update(ml_result.base_result.polygon)
            return ml_result

        self.stats.record_fail(scenario)
        return EnhancedScanResult(
            success=False,
            base_result=direct.base_result,
            notes=self._merge_notes(notes, ml_result.notes),
            error=direct.error or switched.error or ml_result.error or 'Enhanced decode failed',
            scenario=scenario,
        )

    def scan_stream_frame(self, frame: np.ndarray, camera_adaptation: dict[str, Any] | None = None) -> EnhancedScanResult:
        quality = evaluate_quality(frame)
        self.calibrator.update(quality.mean_brightness, quality.laplacian_variance, quality.contrast_stddev)
        self.frames.push(frame)
        single = self.scan_image(frame)
        if self._has_any_qr_hit(single):
            single.camera_adaptation = camera_adaptation
            if single.success:
                self._maybe_save_stats()
            return single
        if len(self.frames) >= 3:
            fused = self.frames.fused()
            if fused is not None:
                fused_result = self.scan_image(fused)
                if self._has_any_qr_hit(fused_result):
                    fused_result.notes = self._merge_notes(fused_result.notes, ['Decode recovered via multi-frame fusion'])
                    fused_result.enhancement_stage = fused_result.enhancement_stage or 'multi-frame-fusion'
                    fused_result.camera_adaptation = camera_adaptation
                    if fused_result.success:
                        self._maybe_save_stats()
                    return fused_result
        self.roi_tracker.mark_miss()
        single.camera_adaptation = camera_adaptation
        return single

    def calibration_status(self) -> str:
        return self.calibrator.thresholds().describe() if self.calibrator.is_ready else self.calibrator.progress_line()

    def pipeline_stats_summary(self) -> dict:
        return self.stats.summary()
