import math
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

Point = Tuple[float, float]


# ─────────────────────────────────────────────
# 基础几何工具
# ─────────────────────────────────────────────

def is_valid(kp) -> bool:
    return isinstance(kp, tuple) and len(kp) == 2 and all(v is not None for v in kp)


def calc_angle_3points(a, b, c) -> Optional[float]:
    if not all(is_valid(p) for p in [a, b, c]):
        return None

    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
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

    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return None

    cos_val = abs(vec[0]) / norm
    return float(np.degrees(np.arccos(np.clip(cos_val, 0.0, 1.0))))


def calc_distance(p1, p2) -> Optional[float]:
    if not all(is_valid(p) for p in [p1, p2]):
        return None
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def midpoint(p1, p2) -> Optional[Point]:
    if not all(is_valid(p) for p in [p1, p2]):
        return None
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def safe_abs_diff(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(a - b)


# ─────────────────────────────────────────────
# 关键点标准化工具
# ─────────────────────────────────────────────

class KeypointAdapter:
    """
    把动态 ROI / 多视角输入统一成可供特征计算的标准关键点。
    - normalize=True 时将关键点映射到 ROI 内 [0,1]
    - canonicalize=True 且运动方向向左时，做水平翻转，使“前进方向”统一为向右
    """

    @staticmethod
    def normalize_keypoints(kps: Dict, roi_box=None) -> Dict:
        if not isinstance(kps, dict):
            return {}

        if roi_box is None:
            # 若已经像归一化坐标（基本位于 0~1），则直接返回
            values = [v for v in kps.values() if is_valid(v)]
            if values and all(0.0 <= p[0] <= 1.2 and 0.0 <= p[1] <= 1.2 for p in values):
                return dict(kps)
            return dict(kps)

        x1, y1, x2, y2 = roi_box
        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)

        out = {}
        for name, pt in kps.items():
            if is_valid(pt):
                out[name] = ((pt[0] - x1) / w, (pt[1] - y1) / h)
            else:
                out[name] = pt
        return out

    @staticmethod
    def infer_motion_direction(kps: Dict, prev_kps: Optional[Dict] = None) -> str:
        """
        返回 right / left / unknown
        优先使用当前帧的人体左右结构；若不稳定，再参考上一帧髋中心位移。
        """
        ls, rs = kps.get('left_shoulder'), kps.get('right_shoulder')
        lh, rh = kps.get('left_hip'), kps.get('right_hip')

        cues = []
        if is_valid(ls) and is_valid(rs):
            cues.append(rs[0] - ls[0])
        if is_valid(lh) and is_valid(rh):
            cues.append(rh[0] - lh[0])

        if cues:
            avg = sum(cues) / len(cues)
            if abs(avg) > 0.02:
                return 'right' if avg > 0 else 'left'

        if prev_kps:
            curr_mid = midpoint(lh, rh)
            prev_mid = midpoint(prev_kps.get('left_hip'), prev_kps.get('right_hip'))
            if is_valid(curr_mid) and is_valid(prev_mid):
                dx = curr_mid[0] - prev_mid[0]
                if abs(dx) > 0.01:
                    return 'right' if dx > 0 else 'left'

        return 'unknown'

    @staticmethod
    def canonicalize_direction(kps: Dict, motion_direction: str = 'unknown') -> Dict:
        if motion_direction != 'left':
            return dict(kps)

        out = {}
        for name, pt in kps.items():
            if is_valid(pt):
                out[name] = (1.0 - float(pt[0]), float(pt[1]))
            else:
                out[name] = pt
        return out


# ─────────────────────────────────────────────
# 单帧特征
# ─────────────────────────────────────────────

class FrameFeatureCalculator:
    """
    向后兼容：
        compute(kps)
    扩展支持：
        compute(kps, roi_box=..., view_type='side', prev_kps=..., motion_direction='auto')
    """

    def compute(
        self,
        kps: Dict,
        roi_box=None,
        view_type: str = 'unknown',
        prev_kps: Optional[Dict] = None,
        motion_direction: str = 'auto',
        normalize: bool = True,
        canonicalize: bool = True,
    ) -> Optional[Dict]:
        if not isinstance(kps, dict) or len(kps) == 0:
            return None

        proc = dict(kps)
        prev_proc = dict(prev_kps) if isinstance(prev_kps, dict) else None

        if normalize:
            proc = KeypointAdapter.normalize_keypoints(proc, roi_box)
            if prev_proc is not None:
                prev_proc = KeypointAdapter.normalize_keypoints(prev_proc, roi_box)

        inferred_direction = motion_direction
        if motion_direction == 'auto':
            inferred_direction = KeypointAdapter.infer_motion_direction(proc, prev_proc)

        if canonicalize:
            proc = KeypointAdapter.canonicalize_direction(proc, inferred_direction)
            if prev_proc is not None:
                prev_proc = KeypointAdapter.canonicalize_direction(prev_proc, inferred_direction)

        features = {
            'view_type': view_type,
            'motion_direction': inferred_direction,
            'trunk_lean_angle': self._trunk_lean(proc),
            'left_thigh_angle': self._thigh_angle(proc, 'left'),
            'right_thigh_angle': self._thigh_angle(proc, 'right'),
            'left_knee_angle': self._knee_angle(proc, 'left'),
            'right_knee_angle': self._knee_angle(proc, 'right'),
            'left_elbow_angle': self._elbow_angle(proc, 'left'),
            'right_elbow_angle': self._elbow_angle(proc, 'right'),
            'shoulder_tilt': self._shoulder_tilt(proc),
            'hip_tilt': self._hip_tilt(proc),
            'hip_width': calc_distance(proc.get('left_hip'), proc.get('right_hip')),
            'shoulder_width': calc_distance(proc.get('left_shoulder'), proc.get('right_shoulder')),
        }

        lt, rt = features['left_thigh_angle'], features['right_thigh_angle']
        lk, rk = features['left_knee_angle'], features['right_knee_angle']
        le, re = features['left_elbow_angle'], features['right_elbow_angle']

        features['thigh_asymmetry'] = safe_abs_diff(lt, rt)
        features['knee_asymmetry'] = safe_abs_diff(lk, rk)
        features['elbow_asymmetry'] = safe_abs_diff(le, re)
        features['support_knee_angle'] = self._support_knee_angle(features, view_type)
        features['swing_thigh_angle'] = self._swing_thigh_angle(features, view_type)
        features['body_symmetry_score'] = self._body_symmetry_score(features)

        # 保留兼容字段，方便旧版 pipeline 少改甚至不改
        features['trunk'] = features['trunk_lean_angle']
        features['thigh'] = features['swing_thigh_angle']
        features['knee'] = features['support_knee_angle']

        numeric_values = [v for v in features.values() if isinstance(v, (int, float))]
        if not numeric_values:
            return None

        return features

    def _trunk_lean(self, kps):
        shoulder_mid = midpoint(kps.get('left_shoulder'), kps.get('right_shoulder'))
        hip_mid = midpoint(kps.get('left_hip'), kps.get('right_hip'))
        return calc_angle_to_vertical(shoulder_mid, hip_mid)

    def _thigh_angle(self, kps, side):
        return calc_angle_to_vertical(kps.get(f'{side}_hip'), kps.get(f'{side}_knee'))

    def _knee_angle(self, kps, side):
        return calc_angle_3points(
            kps.get(f'{side}_hip'),
            kps.get(f'{side}_knee'),
            kps.get(f'{side}_ankle'),
        )

    def _elbow_angle(self, kps, side):
        return calc_angle_3points(
            kps.get(f'{side}_shoulder'),
            kps.get(f'{side}_elbow'),
            kps.get(f'{side}_wrist'),
        )

    def _shoulder_tilt(self, kps):
        return calc_angle_to_horizontal(kps.get('left_shoulder'), kps.get('right_shoulder'))

    def _hip_tilt(self, kps):
        return calc_angle_to_horizontal(kps.get('left_hip'), kps.get('right_hip'))

    def _support_knee_angle(self, features, view_type: str):
        lk = features.get('left_knee_angle')
        rk = features.get('right_knee_angle')
        if view_type == 'side':
            # 侧面以更伸展的一侧作为支撑腿近似
            vals = [v for v in [lk, rk] if v is not None]
            return max(vals) if vals else None
        vals = [v for v in [lk, rk] if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _swing_thigh_angle(self, features, view_type: str):
        lt = features.get('left_thigh_angle')
        rt = features.get('right_thigh_angle')
        vals = [v for v in [lt, rt] if v is not None]
        if not vals:
            return None
        if view_type == 'side':
            return max(vals)
        return sum(vals) / len(vals)

    def _body_symmetry_score(self, features):
        parts = [
            features.get('thigh_asymmetry'),
            features.get('knee_asymmetry'),
            features.get('elbow_asymmetry'),
            features.get('shoulder_tilt'),
            features.get('hip_tilt'),
        ]
        vals = [v for v in parts if v is not None]
        if not vals:
            return None
        # 分数越高越对称，满分 100
        penalty = min(sum(vals) / len(vals), 100.0)
        return round(max(0.0, 100.0 - penalty), 2)


# ─────────────────────────────────────────────
# 时序特征（多人独立）
# ─────────────────────────────────────────────

class TemporalFeatureCalculator:
    def __init__(self, fps: float, window_seconds: float = 2.0):
        self.fps = fps
        self.window_size = max(5, int(fps * window_seconds))
        self.ankle_history: Dict[int, deque] = {}

    def update(self, person_id: int, frame_idx: int, kps: Dict, roi_box=None):
        if person_id not in self.ankle_history:
            self.ankle_history[person_id] = deque(maxlen=self.window_size)

        norm_kps = KeypointAdapter.normalize_keypoints(kps, roi_box)
        la = norm_kps.get('left_ankle')
        ra = norm_kps.get('right_ankle')

        self.ankle_history[person_id].append({
            'frame': frame_idx,
            'left': float(la[1]) if is_valid(la) else None,
            'right': float(ra[1]) if is_valid(ra) else None,
        })

    def compute_cadence(self, person_id: int) -> Optional[float]:
        history = self.ankle_history.get(person_id)
        if not history or len(history) < max(6, int(self.fps)):
            return None

        y_series = []
        for h in history:
            if h['left'] is not None and h['right'] is not None:
                y_series.append(min(h['left'], h['right']))
            elif h['left'] is not None:
                y_series.append(h['left'])
            elif h['right'] is not None:
                y_series.append(h['right'])

        if len(y_series) < max(6, int(self.fps)):
            return None

        amplitude = max(y_series) - min(y_series)
        prominence = max(0.01, amplitude * 0.15)
        peaks = self._count_valleys(y_series, min_prominence=prominence)
        duration = len(y_series) / self.fps
        if duration < 0.5 or peaks <= 0:
            return None

        return round((peaks / duration) * 60.0, 1)

    def _count_valleys(self, y, min_prominence=0.02):
        peaks = 0
        for i in range(1, len(y) - 1):
            if y[i] < y[i - 1] and y[i] < y[i + 1]:
                left = max(y[max(0, i - 5):i], default=y[i])
                right = max(y[i + 1:min(len(y), i + 6)], default=y[i])
                if min(left, right) - y[i] >= min_prominence:
                    peaks += 1
        return peaks


# ─────────────────────────────────────────────
# 规则引擎
# ─────────────────────────────────────────────

class GaitRuleEngine:
    """
    输出兼容旧/新两种字段：
    - trunk / thigh / knee
    - trunk_lean / thigh_swing / knee_support / symmetry / cadence
    """

    THRESHOLDS = {
        'side': {
            'trunk': (5, 25),
            'thigh': (20, 85),
            'knee': (135, 178),
            'elbow': (60, 125),
            'tilt_max': 10,
            'cadence': (150, 210),
        },
        'front': {
            'trunk': (0, 20),
            'thigh': (15, 70),
            'knee': (130, 175),
            'elbow': (60, 125),
            'tilt_max': 8,
            'cadence': (150, 210),
            'symmetry_min': 70,
        },
        'back': {
            'trunk': (0, 20),
            'thigh': (15, 70),
            'knee': (130, 175),
            'elbow': (60, 125),
            'tilt_max': 8,
            'cadence': (150, 210),
            'symmetry_min': 70,
        },
        'unknown': {
            'trunk': (3, 25),
            'thigh': (20, 80),
            'knee': (130, 175),
            'elbow': (70, 120),
            'tilt_max': 8,
            'cadence': (150, 200),
            'symmetry_min': 65,
        },
    }

    def evaluate(self, features: Optional[dict], view_type: Optional[str] = None) -> Optional[dict]:
        if features is None:
            return None

        view = view_type or features.get('view_type') or 'unknown'
        cfg = self.THRESHOLDS.get(view, self.THRESHOLDS['unknown'])
        result = {}

        trunk_res = self._check_range(
            features.get('trunk_lean_angle'),
            *cfg['trunk'],
            label='躯干前倾',
            unit='°',
            low_msg='躯干过直，前倾不足',
            high_msg='躯干前倾过大',
        )
        thigh_res = self._check_range(
            features.get('swing_thigh_angle'),
            *cfg['thigh'],
            label='摆动腿幅度',
            unit='°',
            low_msg='摆腿不足',
            high_msg='摆腿过大',
        )
        knee_res = self._check_range(
            features.get('support_knee_angle'),
            *cfg['knee'],
            label='支撑膝角',
            unit='°',
            low_msg='支撑腿屈膝偏多',
            high_msg='支撑腿过伸',
        )

        result['trunk'] = trunk_res
        result['thigh'] = thigh_res
        result['knee'] = knee_res

        result['trunk_lean'] = trunk_res
        result['thigh_swing'] = thigh_res
        result['knee_support'] = knee_res

        for side_cn, side_en in [('左肘', 'left'), ('右肘', 'right')]:
            result[f'elbow_{side_cn}'] = self._check_range(
                features.get(f'{side_en}_elbow_angle'),
                *cfg['elbow'],
                label=side_cn,
                unit='°',
                low_msg='摆臂收得过小',
                high_msg='摆臂张得过大',
            )

        if view in {'front', 'back', 'unknown'}:
            result['symmetry'] = self._check_min(
                features.get('body_symmetry_score'),
                cfg.get('symmetry_min', 65),
                label='身体对称性',
                unit='',
                low_msg='左右摆动不够对称',
            )
            result['shoulder_tilt'] = self._check_max(
                features.get('shoulder_tilt'),
                cfg['tilt_max'],
                label='肩部倾斜',
                unit='°',
                high_msg='肩部左右倾斜偏大',
            )
            result['hip_tilt'] = self._check_max(
                features.get('hip_tilt'),
                cfg['tilt_max'],
                label='髋部倾斜',
                unit='°',
                high_msg='骨盆左右倾斜偏大',
            )

        return result

    def evaluate_cadence(self, cadence: Optional[float], view_type: str = 'unknown') -> Optional[dict]:
        if cadence is None:
            return None
        cfg = self.THRESHOLDS.get(view_type, self.THRESHOLDS['unknown'])
        return self._check_range(
            cadence,
            *cfg['cadence'],
            label='步频',
            unit='spm',
            low_msg='步频偏低',
            high_msg='步频偏高',
        )

    def _check_range(self, val, min_v, max_v, label='', unit='', low_msg='', high_msg=''):
        if val is None:
            return {
                'label': label,
                'value': None,
                'unit': unit,
                'status': 'unknown',
                'message': '关键点不足',
            }
        status = 'good' if min_v <= val <= max_v else 'warning'
        if status == 'good':
            message = '正常'
        else:
            message = low_msg if val < min_v else high_msg
        return {
            'label': label,
            'value': round(float(val), 1),
            'unit': unit,
            'status': status,
            'message': message,
        }

    def _check_min(self, val, min_v, label='', unit='', low_msg=''):
        if val is None:
            return {
                'label': label,
                'value': None,
                'unit': unit,
                'status': 'unknown',
                'message': '关键点不足',
            }
        status = 'good' if val >= min_v else 'warning'
        return {
            'label': label,
            'value': round(float(val), 1),
            'unit': unit,
            'status': status,
            'message': '正常' if status == 'good' else low_msg,
        }

    def _check_max(self, val, max_v, label='', unit='', high_msg=''):
        if val is None:
            return {
                'label': label,
                'value': None,
                'unit': unit,
                'status': 'unknown',
                'message': '关键点不足',
            }
        status = 'good' if val <= max_v else 'warning'
        return {
            'label': label,
            'value': round(float(val), 1),
            'unit': unit,
            'status': status,
            'message': '正常' if status == 'good' else high_msg,
        }


# ─────────────────────────────────────────────
# 特征平滑
# ─────────────────────────────────────────────

class FeatureSmoother:
    """对每个人的特征做 EMA + 滑动窗口 平滑。"""

    def __init__(self, alpha=0.6, window_size=5):
        self.alpha = alpha
        self.window_size = window_size
        self.history: Dict[int, Dict[str, deque]] = {}
        self.ema_state: Dict[int, Dict[str, float]] = {}

    def update(self, person_id: int, features: dict) -> Optional[dict]:
        if features is None:
            return None

        if person_id not in self.history:
            self.history[person_id] = {}
            self.ema_state[person_id] = {}

        smoothed = {}
        for key, val in features.items():
            if not isinstance(val, (int, float)):
                smoothed[key] = val
                continue

            if key not in self.history[person_id]:
                self.history[person_id][key] = deque(maxlen=self.window_size)

            self.history[person_id][key].append(float(val))
            window_avg = sum(self.history[person_id][key]) / len(self.history[person_id][key])

            prev = self.ema_state[person_id].get(key, window_avg)
            ema = self.alpha * window_avg + (1 - self.alpha) * prev
            self.ema_state[person_id][key] = ema
            smoothed[key] = round(ema, 2)

        return smoothed
