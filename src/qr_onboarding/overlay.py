from __future__ import annotations
import cv2, numpy as np
from .models import ScanResult

def draw_overlay(frame: np.ndarray, result: ScanResult) -> np.ndarray:
    canvas=frame.copy()
    if result.polygon and len(result.polygon)>=4:
        pts=np.array(result.polygon,dtype=np.int32).reshape((-1,1,2)); cv2.polylines(canvas,[pts],True,(0,255,0) if result.success else (0,0,255),2)
    lines=[]
    if result.success:
        lines.append(f'decoder: {result.decoder} | stage: {result.stage}')
        if result.parsed_payload:
            lines.append(f'payload: {result.parsed_payload.payload_kind}')
            if result.parsed_payload.sensitive: lines.append('warning: payload contains sensitive data')
    else:
        lines.append('QR not decoded')
        if result.error: lines.append(result.error)
    if result.quality:
        q=result.quality; lines.append(f'brightness={q.mean_brightness:.1f} contrast={q.contrast_stddev:.1f} sharpness={q.laplacian_variance:.1f}')
        if q.projected_qr_size_px is not None: lines.append(f'projected_qr_size={q.projected_qr_size_px:.1f}px')
        lines.append(f'hint: {q.operator_hint}')
    x,y=15,25
    for line in lines:
        cv2.putText(canvas,line,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(canvas,line,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(20,20,20),1,cv2.LINE_AA)
        y+=24
    return canvas
