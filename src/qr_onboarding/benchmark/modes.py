from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..enhanced_pipeline import EnhancedQRSystem, EnhancedScanResult
from ..models import ScanResult
from ..pipeline import QRReader

DEFAULT_METHODS = [
    'raw_pyzbar',
    'raw_opencv',
    'raw_combined',
    'fixed:proposed_integral',
    'adaptive_full',
    'adaptive_no_roi',
    'adaptive_no_ml',
    'adaptive_no_switch',
    'adaptive_no_quality',
    'adaptive_ml_only',
]


@dataclass(slots=True)
class MethodRunner:
    name: str
    run: Callable[[np.ndarray], ScanResult | EnhancedScanResult]
    reset: Callable[[], None]


def build_method_runner(method_name: str, *, private_key: str | None = None) -> MethodRunner:
    if method_name == 'raw_pyzbar':
        reader = QRReader(private_key=private_key, try_opencv_text_fallback=False)
        return MethodRunner(method_name, reader.scan_image_pyzbar_only, lambda: None)
    if method_name == 'raw_opencv':
        reader = QRReader(private_key=private_key, try_opencv_text_fallback=True)
        return MethodRunner(method_name, reader.scan_image_opencv_only, lambda: None)
    if method_name == 'raw_combined':
        reader = QRReader(private_key=private_key, try_opencv_text_fallback=True)
        return MethodRunner(method_name, reader.scan_image_raw_combined, lambda: None)

    system = EnhancedQRSystem(private_key=private_key)
    if method_name.startswith('fixed:'):
        stage_name = method_name.split(':', 1)[1]
        return MethodRunner(method_name, lambda image, stage_name=stage_name: system.scan_fixed_stage(image, stage_name), system.reset_runtime_state)
    if method_name == 'adaptive_full':
        return MethodRunner(method_name, system.scan_image, system.reset_runtime_state)
    if method_name == 'adaptive_no_roi':
        return MethodRunner(method_name, system.scan_without_roi, system.reset_runtime_state)
    if method_name == 'adaptive_no_ml':
        return MethodRunner(method_name, system.scan_without_ml, system.reset_runtime_state)
    if method_name == 'adaptive_no_switch':
        return MethodRunner(method_name, system.scan_without_switch, system.reset_runtime_state)
    if method_name == 'adaptive_no_quality':
        return MethodRunner(method_name, system.scan_without_quality_assessment, system.reset_runtime_state)
    if method_name == 'adaptive_switch_only':
        return MethodRunner(method_name, system.scan_switch_only, system.reset_runtime_state)
    if method_name == 'adaptive_ml_only':
        return MethodRunner(method_name, system.scan_ml_only, system.reset_runtime_state)
    raise ValueError(f'Unsupported benchmark method: {method_name}')
