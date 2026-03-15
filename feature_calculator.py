"""
跑姿特征计算模块
针对俯角侧面视角优化

包含：
1. 躯干前倾角
2. 抬腿高度（大腿与竖直方向夹角）
3. 膝关节角度
4. 肘关节角度
5. 步频（基于踝点运动周期）
6. 左右对称性（正面视角）
"""

import numpy as np
from collections import deque
from typing import Optional


# ─────────────────────────────────────────────
# 基础几何工具
# ─────────────────────────────────────────────

def calc_angle_3points(a, b, c) -> float:
    """
    计算三点夹角，b 为顶点
    返回角度 0-180°
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def calc_angle_to_vertical(p1, p2) -> float:
    """
    Calculate angle between p1->p2 vector and vertical axis.
    Returns 0-90 degrees (0=perfectly vertical, 90=horizontal).
    Handles overhead camera: takes absolute value so direction does not matter.
    """
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    vec = p2 - p1
    if np.linalg.norm(vec) < 1e-6:
        return 0.0
    cos_val = abs(vec[1]) / (np.linalg.norm(vec) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, 0.0, 1.0))))


def calc_angle_to_horizontal(p1, p2) -> float:
    """
    计算 p1→p2 连线与水平方向的夹角
    返回 0-90°
    """
    p1, p2 = np.array(p1), np.array(p2)
    vec = p2 - p1
    horizontal = np.array([1, 0])
    cos_val = np.dot(vec, horizontal) / (np.linalg.norm(vec) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(abs(cos_val), 0, 1.0))))


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def is_valid(kp) -> bool:
    """检查关键点是否有效（tuple类型）"""
    return isinstance(kp, tuple) and len(kp) == 2


# ─────────────────────────────────────────────
# 单帧特征计算
# ─────────────────────────────────────────────

class FrameFeatureCalculator:
    """
    对单帧的关键点字典计算所有可用特征
    输入：MediaPipePoseAnalyzer.analyze_person() 返回的 keypoints 字典
    """

    def compute(self, kps: dict) -> dict:
        """
        计算所有特征，返回字典
        缺失关键点的特征返回 None
        """
        features = {}

        features['trunk_lean_angle'] = self._trunk_lean(kps)
        features['left_thigh_angle'] = self._thigh_angle(kps, side='left')
        features['right_thigh_angle'] = self._thigh_angle(kps, side='right')
        features['left_knee_angle'] = self._knee_angle(kps, side='left')
        features['right_knee_angle'] = self._knee_angle(kps, side='right')
        features['left_elbow_angle'] = self._elbow_angle(kps, side='left')
        features['right_elbow_angle'] = self._elbow_angle(kps, side='right')
        features['shoulder_tilt'] = self._shoulder_tilt(kps)
        features['hip_tilt'] = self._hip_tilt(kps)

        # 派生特征
        lt = features['left_thigh_angle']
        rt = features['right_thigh_angle']
        if lt is not None and rt is not None:
            features['thigh_asymmetry'] = abs(lt - rt)
        else:
            features['thigh_asymmetry'] = None

        return features

    def _trunk_lean(self, kps) -> Optional[float]:
        """
        躯干前倾角：肩中点 → 髋中点 连线与竖直方向夹角
        俯角视频中有系统误差，但相对值仍然有意义
        0° = 完全直立, 角度越大 = 越前倾
        """
        ls = kps.get('left_shoulder')
        rs = kps.get('right_shoulder')
        lh = kps.get('left_hip')
        rh = kps.get('right_hip')

        if not all(is_valid(p) for p in [ls, rs, lh, rh]):
            return None

        shoulder_mid = midpoint(ls, rs)
        hip_mid = midpoint(lh, rh)
        return calc_angle_to_vertical(shoulder_mid, hip_mid)

    def _thigh_angle(self, kps, side='left') -> Optional[float]:
        """
        大腿抬起角度：髋 → 膝 连线与竖直方向夹角
        0° = 大腿垂直向下（支撑相）
        角度越大 = 大腿抬得越高（摆动相）
        """
        hip = kps.get(f'{side}_hip')
        knee = kps.get(f'{side}_knee')

        if not (is_valid(hip) and is_valid(knee)):
            return None

        return calc_angle_to_vertical(hip, knee)

    def _knee_angle(self, kps, side='left') -> Optional[float]:
        """
        膝关节弯曲角度：髋 - 膝 - 踝 三点夹角
        180° = 完全伸直
        角度越小 = 弯曲越多
        """
        hip = kps.get(f'{side}_hip')
        knee = kps.get(f'{side}_knee')
        ankle = kps.get(f'{side}_ankle')

        if not all(is_valid(p) for p in [hip, knee, ankle]):
            return None

        return calc_angle_3points(hip, knee, ankle)

    def _elbow_angle(self, kps, side='left') -> Optional[float]:
        """
        肘关节角度：肩 - 肘 - 腕 三点夹角
        90° 左右为良好摆臂角度
        """
        shoulder = kps.get(f'{side}_shoulder')
        elbow = kps.get(f'{side}_elbow')
        wrist = kps.get(f'{side}_wrist')

        if not all(is_valid(p) for p in [shoulder, elbow, wrist]):
            return None

        return calc_angle_3points(shoulder, elbow, wrist)

    def _shoulder_tilt(self, kps) -> Optional[float]:
        """
        肩部倾斜角：左右肩连线与水平方向夹角
        0° = 完全水平（对称）
        正面视角时更有意义
        """
        ls = kps.get('left_shoulder')
        rs = kps.get('right_shoulder')

        if not (is_valid(ls) and is_valid(rs)):
            return None

        return calc_angle_to_horizontal(ls, rs)

    def _hip_tilt(self, kps) -> Optional[float]:
        """
        髋部倾斜角：左右髋连线与水平方向夹角
        用于检测塌髋
        """
        lh = kps.get('left_hip')
        rh = kps.get('right_hip')

        if not (is_valid(lh) and is_valid(rh)):
            return None

        return calc_angle_to_horizontal(lh, rh)


# ─────────────────────────────────────────────
# 时序特征计算（步频）
# ─────────────────────────────────────────────

class TemporalFeatureCalculator:
    """
    跨帧时序特征计算
    主要用于步频检测
    """

    def __init__(self, fps: float, window_seconds: float = 2.0):
        """
        fps: 视频帧率
        window_seconds: 用于计算步频的时间窗口
        """
        self.fps = fps
        self.window_size = int(fps * window_seconds)

        # 每个人的踝点历史轨迹
        # person_id -> deque of (frame_idx, left_ankle_y, right_ankle_y)
        self.ankle_history: dict[int, deque] = {}

        # 步频结果缓存
        self.cadence_cache: dict[int, Optional[float]] = {}

    def update(self, person_id: int, frame_idx: int, kps: dict):
        """
        更新某人的踝点轨迹
        """
        if person_id not in self.ankle_history:
            self.ankle_history[person_id] = deque(maxlen=self.window_size)

        la = kps.get('left_ankle')
        ra = kps.get('right_ankle')

        la_y = la[1] if is_valid(la) else None
        ra_y = ra[1] if is_valid(ra) else None

        self.ankle_history[person_id].append({
            'frame': frame_idx,
            'left_ankle_y': la_y,
            'right_ankle_y': ra_y
        })

    def compute_cadence(self, person_id: int) -> Optional[float]:
        """
        计算步频（步/分钟）
        原理：踝点Y坐标周期性上下运动，检测波峰数量
        """
        if person_id not in self.ankle_history:
            return None

        history = list(self.ankle_history[person_id])
        if len(history) < int(self.fps):  # 至少需要1秒数据
            return None

        # 合并左右踝的Y坐标（取有效值）
        y_series = []
        for h in history:
            y = None
            if h['left_ankle_y'] is not None:
                y = h['left_ankle_y']
            elif h['right_ankle_y'] is not None:
                y = h['right_ankle_y']
            y_series.append(y)

        # 过滤掉None
        valid_y = [y for y in y_series if y is not None]
        if len(valid_y) < int(self.fps):
            return None

        # 简单峰值检测（踝点抬起 = Y值减小）
        peaks = self._count_peaks(valid_y)

        # 时间窗口长度（秒）
        duration_seconds = len(valid_y) / self.fps

        if duration_seconds < 0.5:
            return None

        # 步频 = 峰值数 / 时间 * 60
        cadence = (peaks / duration_seconds) * 60
        return round(cadence, 1)

    def _count_peaks(self, y_series: list, min_prominence: int = 5) -> int:
        """
        计算序列中的波峰数量
        min_prominence: 最小波峰显著度（像素），过滤噪声
        """
        peaks = 0
        n = len(y_series)

        for i in range(1, n - 1):
            # 局部最小值（Y减小 = 抬腿）
            if y_series[i] < y_series[i - 1] and y_series[i] < y_series[i + 1]:
                # 检查显著度
                left_max = max(y_series[max(0, i - 5):i])
                right_max = max(y_series[i + 1:min(n, i + 6)])
                prominence = min(left_max, right_max) - y_series[i]
                if prominence >= min_prominence:
                    peaks += 1

        return peaks


# ─────────────────────────────────────────────
# 规则引擎（判断对错）
# ─────────────────────────────────────────────

class GaitRuleEngine:
    """
    基于角度阈值的跑姿判断规则
    针对俯角视频做了保守调整（阈值范围比标准侧面更宽松）
    """

    # 阈值配置（可根据实际数据调整）
    THRESHOLDS = {
        # 躯干前倾角（俯角视频有系统误差，阈值放宽）
        'trunk_lean': {'min': 3, 'max': 25, 'ideal': 10},

        # 大腿抬起角度（摆动相）
        'thigh_swing': {'min': 20, 'max': 80, 'ideal': 45},

        # 支撑相膝关节角度（触地时不应完全伸直）
        'knee_support': {'min': 130, 'max': 175},

        # 肘关节角度
        'elbow': {'min': 70, 'max': 120, 'ideal': 90},

        # 肩部/髋部倾斜（对称性）
        'tilt_max': 8,

        # 步频范围（步/分钟）
        'cadence': {'min': 150, 'max': 200, 'ideal': 170},
    }

    def evaluate(self, features: dict) -> dict:
        """
        对一帧的特征进行规则判断
        返回每个维度的评估结果
        """
        results = {}

        # 躯干前倾
        trunk = features.get('trunk_lean_angle')
        results['trunk_lean'] = self._check_range(
            trunk,
            self.THRESHOLDS['trunk_lean']['min'],
            self.THRESHOLDS['trunk_lean']['max'],
            label='Trunk Lean',
            unit='deg',
            low_msg='Too upright, lean forward slightly',
            high_msg='Leaning too far forward'
        )

        # Thigh angle (take the swing leg = larger angle)
        lt = features.get('left_thigh_angle')
        rt = features.get('right_thigh_angle')
        swing_thigh = max(filter(lambda x: x is not None, [lt, rt]), default=None)
        results['thigh_swing'] = self._check_range(
            swing_thigh,
            self.THRESHOLDS['thigh_swing']['min'],
            self.THRESHOLDS['thigh_swing']['max'],
            label='Thigh Swing',
            unit='deg',
            low_msg='Knee lift too low',
            high_msg='Knee lift too high'
        )

        # Knee angle (take support leg = larger angle)
        lk = features.get('left_knee_angle')
        rk = features.get('right_knee_angle')
        support_knee = max(filter(lambda x: x is not None, [lk, rk]), default=None)
        results['knee_support'] = self._check_range(
            support_knee,
            self.THRESHOLDS['knee_support']['min'],
            self.THRESHOLDS['knee_support']['max'],
            label='Knee Angle',
            unit='deg',
            low_msg='Knee too bent at landing',
            high_msg='Knee too straight, high impact'
        )

        # Elbow angles
        le = features.get('left_elbow_angle')
        re = features.get('right_elbow_angle')
        for side, val in [('L_elbow', le), ('R_elbow', re)]:
            key = f'elbow_{side}'
            results[key] = self._check_range(
                val,
                self.THRESHOLDS['elbow']['min'],
                self.THRESHOLDS['elbow']['max'],
                label=f'{side} Angle',
                unit='deg',
                low_msg=f'{side} too bent',
                high_msg=f'{side} too straight'
            )

        # Shoulder symmetry
        shoulder_tilt = features.get('shoulder_tilt')
        results['shoulder_symmetry'] = self._check_max(
            shoulder_tilt,
            self.THRESHOLDS['tilt_max'],
            label='Shoulder Tilt',
            unit='deg',
            bad_msg='Shoulders uneven, check arm swing'
        )

        return results

    def evaluate_cadence(self, cadence: Optional[float]) -> dict:
        """Evaluate cadence (steps/min)"""
        return self._check_range(
            cadence,
            self.THRESHOLDS['cadence']['min'],
            self.THRESHOLDS['cadence']['max'],
            label='Cadence',
            unit='spm',
            low_msg='Cadence low, aim for 160+ spm',
            high_msg='Cadence high, relax the pace'
        )

    # ── 内部工具 ──

    def _check_range(self, value, min_val, max_val, label, unit,
                     low_msg, high_msg) -> dict:
        if value is None:
            return {'status': 'unknown', 'value': None, 'label': label, 'message': 'No data'}

        value = round(value, 1)
        if value < min_val:
            status = 'warning'
            message = low_msg
        elif value > max_val:
            status = 'warning'
            message = high_msg
        else:
            status = 'good'
            message = 'OK'

        return {
            'status': status,
            'value': value,
            'label': label,
            'unit': unit,
            'range': f'{min_val}-{max_val}{unit}',
            'message': message
        }

    def _check_max(self, value, max_val, label, unit, bad_msg) -> dict:
        if value is None:
            return {'status': 'unknown', 'value': None, 'label': label, 'message': 'No data'}

        value = round(value, 1)
        status = 'good' if value <= max_val else 'warning'
        message = 'OK' if status == 'good' else bad_msg

        return {
            'status': status,
            'value': value,
            'label': label,
            'unit': unit,
            'message': message
        }


# ─────────────────────────────────────────────
# 快速测试入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # 用假数据验证模块是否正常工作
    calc = FrameFeatureCalculator()
    engine = GaitRuleEngine()

    mock_kps = {
        'left_shoulder':  (400, 300),
        'right_shoulder': (450, 295),
        'left_hip':       (410, 430),
        'right_hip':      (455, 425),
        'left_knee':      (405, 560),
        'right_knee':     (460, 530),
        'left_ankle':     (400, 680),
        'right_ankle':    (465, 640),
        'left_elbow':     (370, 390),
        'right_elbow':    (490, 385),
        'left_wrist':     (355, 480),
        'right_wrist':    (505, 470),
        'head_center':    (425, 220),
    }

    features = calc.compute(mock_kps)
    print("── 计算特征 ──")
    for k, v in features.items():
        print(f"  {k:25s}: {v}")

    print("\n── 规则评估 ──")
    results = engine.evaluate(features)
    for k, v in results.items():
        status_icon = '✅' if v['status'] == 'good' else ('⚠️' if v['status'] == 'warning' else '❓')
        val_str = f"{v['value']}{v.get('unit','')}" if v['value'] is not None else 'N/A'
        print(f"  {status_icon} {v['label']:15s}: {val_str:10s} → {v['message']}")