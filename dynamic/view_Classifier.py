from __future__ import annotations

from typing import Dict, Optional, Tuple


Point = Tuple[int, int]


def _valid(p) -> bool:
    return isinstance(p, tuple) and len(p) == 2


def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class RunningViewClassifier:
    """
    基于 2D 关键点做轻量视角判断：front / back / side / unknown。

    经验规则：
    - 肩宽/身高、髋宽/身高较小，更可能是 side；
    - front/back 的区分优先依赖鼻子是否可见，以及鼻子相对双肩中点的位置；
    - 单帧不稳定时用最近若干帧投票平滑。
    """

    def __init__(self, history_size: int = 8) -> None:
        self.history_size = history_size
        self.history: Dict[int, list[str]] = {}

    def classify(self, person_id: int, kps: Dict[str, Point]) -> str:
        label = self._classify_once(kps)
        h = self.history.setdefault(person_id, [])
        h.append(label)
        if len(h) > self.history_size:
            h.pop(0)
        return self._vote(h)

    def _classify_once(self, kps: Dict[str, Point]) -> str:
        ls = kps.get('left_shoulder')
        rs = kps.get('right_shoulder')
        lh = kps.get('left_hip')
        rh = kps.get('right_hip')
        nose = kps.get('nose')
        la = kps.get('left_ankle')
        ra = kps.get('right_ankle')

        if not all(_valid(p) for p in [ls, rs, lh, rh]):
            return 'unknown'

        shoulder_w = _dist(ls, rs)
        hip_w = _dist(lh, rh)

        top_y = min(ls[1], rs[1])
        bottom_candidates = [p[1] for p in [la, ra, lh, rh] if _valid(p)]
        if not bottom_candidates:
            return 'unknown'
        body_h = max(bottom_candidates) - top_y
        if body_h <= 1:
            return 'unknown'

        width_ratio = max(shoulder_w, hip_w) / body_h
        if width_ratio < 0.16:
            return 'side'

        if _valid(nose):
            shoulder_mid_x = (ls[0] + rs[0]) / 2.0
            shoulder_w_safe = max(shoulder_w, 1.0)
            nose_offset = (nose[0] - shoulder_mid_x) / shoulder_w_safe
            if abs(nose_offset) <= 0.45:
                return 'front'
            return 'side'

        return 'back'

    def _vote(self, items: list[str]) -> str:
        priority = ['front', 'back', 'side', 'unknown']
        counts = {k: items.count(k) for k in priority}
        best = max(priority, key=lambda k: (counts[k], -priority.index(k)))
        return best
