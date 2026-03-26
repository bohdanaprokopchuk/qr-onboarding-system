from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class ROIState:
    bbox: tuple[int, int, int, int]
    padding: int = 32
    age: int = 0
    hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QRROITracker:
    def __init__(self, padding: int = 32, max_age: int = 8) -> None:
        self.padding = padding
        self.max_age = max_age
        self.state: Optional[ROIState] = None

    @staticmethod
    def _normalize_polygon(points: Any) -> Optional[np.ndarray]:
        if points is None:
            return None
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(arr) < 4:
            return None
        return arr

    def update(self, points: Any) -> Optional[ROIState]:
        arr = self._normalize_polygon(points)
        if arr is None:
            return self.mark_miss()
        x, y, w, h = cv2.boundingRect(arr.astype(np.int32))
        self.state = ROIState((int(x), int(y), int(w), int(h)), self.padding, age=0, hits=(self.state.hits + 1 if self.state else 1))
        return self.state

    def mark_miss(self) -> Optional[ROIState]:
        if self.state is None:
            return None
        self.state.age += 1
        if self.state.age > self.max_age:
            self.state = None
        return self.state

    def crop(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        if self.state is None:
            return frame, (0, 0)
        x, y, w, h = self.state.bbox
        pad = self.state.padding
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + w + pad)
        y1 = min(frame.shape[0], y + h + pad)
        if x1 <= x0 or y1 <= y0:
            return frame, (0, 0)
        return frame[y0:y1, x0:x1].copy(), (x0, y0)

    def remap_polygon(self, points: Any, offset: tuple[int, int]) -> list[tuple[int, int]] | None:
        arr = self._normalize_polygon(points)
        if arr is None:
            return None
        ox, oy = offset
        arr[:, 0] += ox
        arr[:, 1] += oy
        return [(int(round(x)), int(round(y))) for x, y in arr]
