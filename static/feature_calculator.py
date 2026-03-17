import numpy as np
from collections import deque
from typing import Optional


# 基础几何工具

def calc_angle_3points(a, b, c) -> Optional[float]:
    if not all(is_valid(p) for p in [a, b, c]):
        return None

    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cos_val = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def calc_angle_to_vertical(p1, p2) -> Optional[float]:
    if not all(is_valid(p) for p in [p1, p2]):
        return None

    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    vec = p2 - p1
    norm = np.linalg.norm(vec)

    if norm < 1e-6:
        return None

    cos_val = abs(vec[1]) / norm
    return float(np.degrees(np.arccos(np.clip(cos_val, 0.0, 1.0))))


def calc_angle_to_horizontal(p1, p2) -> Optional[float]:
    if not all(is_valid(p) for p in [p1, p2]):
        return None

    p1, p2 = np.array(p1), np.array(p2)
    vec = p2 - p1
    norm = np.linalg.norm(vec)

    if norm < 1e-6:
        return None

    cos_val = abs(vec[0]) / norm
    return float(np.degrees(np.arccos(np.clip(cos_val, 0.0, 1.0))))


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def is_valid(kp) -> bool:
    return isinstance(kp, tuple) and len(kp) == 2


# 单帧特征

class FrameFeatureCalculator:

    def compute(self, kps: dict) -> Optional[dict]:
        if not isinstance(kps, dict) or len(kps) == 0:
            return None

        features = {
            'trunk_lean_angle': self._trunk_lean(kps),
            'left_thigh_angle': self._thigh_angle(kps, 'left'),
            'right_thigh_angle': self._thigh_angle(kps, 'right'),
            'left_knee_angle': self._knee_angle(kps, 'left'),
            'right_knee_angle': self._knee_angle(kps, 'right'),
            'left_elbow_angle': self._elbow_angle(kps, 'left'),
            'right_elbow_angle': self._elbow_angle(kps, 'right'),
            'shoulder_tilt': self._shoulder_tilt(kps),
            'hip_tilt': self._hip_tilt(kps),
        }

        # 派生特征
        lt, rt = features['left_thigh_angle'], features['right_thigh_angle']
        features['thigh_asymmetry'] = abs(lt - rt) if lt is not None and rt is not None else None

        # 如果全部为 None → 无效
        if all(v is None for v in features.values()):
            return None

        return features

    def _trunk_lean(self, kps):
        ls, rs = kps.get('left_shoulder'), kps.get('right_shoulder')
        lh, rh = kps.get('left_hip'), kps.get('right_hip')

        if not all(is_valid(p) for p in [ls, rs, lh, rh]):
            return None

        return calc_angle_to_vertical(midpoint(ls, rs), midpoint(lh, rh))

    def _thigh_angle(self, kps, side):
        return calc_angle_to_vertical(
            kps.get(f'{side}_hip'),
            kps.get(f'{side}_knee')
        )

    def _knee_angle(self, kps, side):
        return calc_angle_3points(
            kps.get(f'{side}_hip'),
            kps.get(f'{side}_knee'),
            kps.get(f'{side}_ankle')
        )

    def _elbow_angle(self, kps, side):
        return calc_angle_3points(
            kps.get(f'{side}_shoulder'),
            kps.get(f'{side}_elbow'),
            kps.get(f'{side}_wrist')
        )

    def _shoulder_tilt(self, kps):
        return calc_angle_to_horizontal(
            kps.get('left_shoulder'),
            kps.get('right_shoulder')
        )

    def _hip_tilt(self, kps):
        return calc_angle_to_horizontal(
            kps.get('left_hip'),
            kps.get('right_hip')
        )


# 时序特征（多人独立）

class TemporalFeatureCalculator:

    def __init__(self, fps: float, window_seconds: float = 2.0):
        self.fps = fps
        self.window_size = int(fps * window_seconds)

        self.ankle_history: dict[int, deque] = {}

    def update(self, person_id: int, frame_idx: int, kps: dict):
        if person_id not in self.ankle_history:
            self.ankle_history[person_id] = deque(maxlen=self.window_size)

        la = kps.get('left_ankle')
        ra = kps.get('right_ankle')

        self.ankle_history[person_id].append({
            'frame': frame_idx,
            'left': la[1] if is_valid(la) else None,
            'right': ra[1] if is_valid(ra) else None
        })

    def compute_cadence(self, person_id: int) -> Optional[float]:
        history = self.ankle_history.get(person_id)
        if not history or len(history) < int(self.fps):
            return None

        y_series = [
            h['left'] if h['left'] is not None else h['right']
            for h in history
        ]

        valid = [y for y in y_series if y is not None]
        if len(valid) < int(self.fps):
            return None

        peaks = self._count_peaks(valid)
        duration = len(valid) / self.fps

        if duration < 0.5:
            return None

        return round((peaks / duration) * 60, 1)

    def _count_peaks(self, y, min_prominence=5):
        peaks = 0
        for i in range(1, len(y) - 1):
            if y[i] < y[i-1] and y[i] < y[i+1]:
                left = max(y[max(0, i-5):i])
                right = max(y[i+1:min(len(y), i+6)])
                if min(left, right) - y[i] >= min_prominence:
                    peaks += 1
        return peaks


# 规则引擎

class GaitRuleEngine:

    THRESHOLDS = {
        'trunk': (3, 25),
        'thigh': (20, 80),
        'knee': (130, 175),
        'elbow': (70, 120),
        'tilt_max': 8,
        'cadence': (150, 200),
    }

    def evaluate(self, features: Optional[dict]) -> Optional[dict]:
        if features is None:
            return None

        result = {}

        result['trunk'] = self._check(features.get('trunk_lean_angle'), *self.THRESHOLDS['trunk'])
        result['thigh'] = self._check(
            max(filter(lambda x: x is not None,
                       [features.get('left_thigh_angle'), features.get('right_thigh_angle')]),
                default=None),
            *self.THRESHOLDS['thigh']
        )

        result['knee'] = self._check(
            max(filter(lambda x: x is not None,
                       [features.get('left_knee_angle'), features.get('right_knee_angle')]),
                default=None),
            *self.THRESHOLDS['knee']
        )

        return result

    def evaluate_cadence(self, cadence: Optional[float]) -> Optional[dict]:
        if cadence is None:
            return None
        return self._check(cadence, *self.THRESHOLDS['cadence'])

    def _check(self, val, min_v, max_v):
        if val is None:
            return None
        return {
            'value': round(val, 1),
            'status': 'good' if min_v <= val <= max_v else 'warning'
        }

class FeatureSmoother:
    """
    对每个人的特征做时序平滑（EMA + 滑动窗口）
    """

    def __init__(self, alpha=0.6, window_size=5):
        """
        alpha: EMA权重（越大越灵敏，越小越稳定）
        window_size: 滑动窗口大小
        """
        self.alpha = alpha
        self.window_size = window_size

        # person_id -> {feature_name: deque}
        self.history = {}

        # person_id -> {feature_name: smoothed_value}
        self.ema_state = {}

    def update(self, person_id: int, features: dict) -> dict:
        if features is None:
            return None

        if person_id not in self.history:
            self.history[person_id] = {}
            self.ema_state[person_id] = {}

        smoothed = {}

        for key, val in features.items():

            # 跳过 None
            if val is None:
                smoothed[key] = None
                continue

            # 初始化窗口
            if key not in self.history[person_id]:
                from collections import deque
                self.history[person_id][key] = deque(maxlen=self.window_size)

            self.history[person_id][key].append(val)

            # 滑动平均（去噪）
            window_avg = sum(self.history[person_id][key]) / len(self.history[person_id][key])

            # EMA
            if key not in self.ema_state[person_id]:
                ema = window_avg
            else:
                ema = self.alpha * window_avg + (1 - self.alpha) * self.ema_state[person_id][key]

            self.ema_state[person_id][key] = ema

            smoothed[key] = round(ema, 2)

        return smoothed
