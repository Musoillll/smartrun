from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


Box = Tuple[int, int, int, int]


@dataclass
class ROIState:
    box: Box
    last_frame: int
    score: float = 1.0


class DynamicROIManager:
    """
    为非固定摄像头场景生成动态 ROI。

    核心思路：
    1. 以人物检测框为基础生成 ROI；
    2. 对单人 ROI 做 margin 扩展，给姿态估计预留上下文；
    3. 对同一 track 的 ROI 做 EMA 平滑，减小镜头抖动；
    4. 可输出单人 ROI 或多人的 union ROI。
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        expand_ratio_x: float = 0.22,
        expand_ratio_y: float = 0.15,
        min_size: int = 160,
        smooth_alpha: float = 0.65,
        ttl_frames: int = 20,
    ) -> None:
        self.w = int(frame_width)
        self.h = int(frame_height)
        self.expand_ratio_x = expand_ratio_x
        self.expand_ratio_y = expand_ratio_y
        self.min_size = int(min_size)
        self.smooth_alpha = smooth_alpha
        self.ttl_frames = int(ttl_frames)
        self.states: Dict[int, ROIState] = {}

    def update_person_roi(self, person_id: int, det_box: Box, frame_idx: int) -> Box:
        raw = self._expand_box(det_box)
        prev = self.states.get(person_id)
        if prev is None:
            box = raw
        else:
            box = self._smooth_box(prev.box, raw)
        self.states[person_id] = ROIState(box=box, last_frame=frame_idx)
        self._cleanup(frame_idx)
        return box

    def get_person_roi(self, person_id: int) -> Optional[Box]:
        state = self.states.get(person_id)
        return state.box if state else None

    def build_union_roi(self, person_ids: list[int], padding: int = 40) -> Optional[Box]:
        boxes = [self.states[pid].box for pid in person_ids if pid in self.states]
        if not boxes:
            return None
        x1 = max(0, min(b[0] for b in boxes) - padding)
        y1 = max(0, min(b[1] for b in boxes) - padding)
        x2 = min(self.w, max(b[2] for b in boxes) + padding)
        y2 = min(self.h, max(b[3] for b in boxes) + padding)
        return (x1, y1, x2, y2)

    def _expand_box(self, box: Box) -> Box:
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        dx = max(int(bw * self.expand_ratio_x), self.min_size // 8)
        dy_top = max(int(bh * (self.expand_ratio_y + 0.1)), self.min_size // 8)
        dy_bottom = max(int(bh * self.expand_ratio_y), self.min_size // 8)

        nx1 = max(0, x1 - dx)
        ny1 = max(0, y1 - dy_top)
        nx2 = min(self.w, x2 + dx)
        ny2 = min(self.h, y2 + dy_bottom)

        if nx2 - nx1 < self.min_size:
            extra = (self.min_size - (nx2 - nx1)) // 2 + 1
            nx1 = max(0, nx1 - extra)
            nx2 = min(self.w, nx2 + extra)

        if ny2 - ny1 < self.min_size:
            extra = (self.min_size - (ny2 - ny1)) // 2 + 1
            ny1 = max(0, ny1 - extra)
            ny2 = min(self.h, ny2 + extra)

        return (int(nx1), int(ny1), int(nx2), int(ny2))

    def _smooth_box(self, prev: Box, cur: Box) -> Box:
        a = self.smooth_alpha
        return tuple(int(round(a * p + (1 - a) * c)) for p, c in zip(prev, cur))  # type: ignore[return-value]

    def _cleanup(self, frame_idx: int) -> None:
        expired = [pid for pid, s in self.states.items() if frame_idx - s.last_frame > self.ttl_frames]
        for pid in expired:
            self.states.pop(pid, None)
