from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional
import cv2, numpy as np
from .preprocessing import to_gray
def _lap(image): return float(cv2.Laplacian(to_gray(image), cv2.CV_64F).var())
@dataclass
class FrameScore: sharpness: float; contrast: float; brightness: float; total: float
@dataclass
class MultiFrameBuffer:
    maxlen: int = 8
    frames: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=8))
    def push(self, frame: np.ndarray)->None:
        if self.frames.maxlen!=self.maxlen: self.frames=deque(self.frames, maxlen=self.maxlen)
        self.frames.append(frame.copy())
    def __len__(self)->int: return len(self.frames)
    def fused(self)->Optional[np.ndarray]: return None if not self.frames else fuse_frames(list(self.frames))
def score_frame(frame: np.ndarray)->FrameScore:
    g=to_gray(frame); s=_lap(g); c=float(np.std(g)); b=float(np.mean(g)); return FrameScore(s,c,b,s*0.7+c*0.2-abs(b-128)*0.1)
def _align(ref, cand):
    if ref.shape!=cand.shape: cand=cv2.resize(cand, (ref.shape[1], ref.shape[0]))
    warp=np.eye(2,3,dtype=np.float32); criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,1e-4)
    try:
        cv2.findTransformECC(ref, cand, warp, cv2.MOTION_EUCLIDEAN, criteria)
        return cv2.warpAffine(cand, warp, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
    except Exception: return cand
def fuse_frames(frames):
    if len(frames)==1: return frames[0].copy()
    ref=max(frames, key=lambda f: score_frame(f).total); rg=to_gray(ref); aligned=[rg]
    for f in frames:
        if f is ref: continue
        aligned.append(_align(rg, to_gray(f)))
    stack=np.stack(aligned, axis=0).astype(np.float32); fused=np.clip(0.6*np.median(stack, axis=0)+0.4*np.mean(stack, axis=0),0,255).astype(np.uint8)
    return cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR)
