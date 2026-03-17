"""
非固定摄像头动态 ROI 跑步姿态分析 Pipeline

流程：
YOLO 人体检测 -> 动态 ROI 生成 -> MediaPipe 姿态估计 -> 视角分类 ->
特征计算/平滑 -> 视角自适应规则评估 -> 可视化与结果导出

用法：
python pipeline_dynamic_camera.py --video your_video.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Optional

import cv2
import numpy as np

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_PATH)

from ROI import DynamicROIManager
from feature_calculator_dyn import (
    FeatureSmoother,
    FrameFeatureCalculator,
    GaitRuleEngine,
    TemporalFeatureCalculator,
)
from mediapipe_analyzer_dyn import MediaPipePoseAnalyzerDynamic
from view_Classifier import RunningViewClassifier
from yolo_detector_dyn import YOLOPersonDetectorDynamic


COLORS = [
    (255, 80, 80),
    (80, 200, 80),
    (80, 80, 255),
    (255, 200, 0),
    (200, 0, 255),
    (0, 220, 220),
]

STATUS_COLOR = {
    'good': (0, 220, 0),
    'warning': (0, 140, 255),
    'unknown': (160, 160, 160),
}


class MultiViewRuleEngine:
    """
    在原有规则基础上增加“视角感知”。

    side: 重点看躯干前倾、大腿摆动、膝角、步频
    front/back: 重点看肩髋倾斜与左右对称性
    """

    def __init__(self) -> None:
        self.base = GaitRuleEngine()

    def evaluate(self, features: Optional[dict], view: str, cadence: Optional[float]) -> Dict[str, dict]:
        if not features:
            return {}

        result: Dict[str, dict] = {}

        def pack(label, value, ok_range=None, max_only=None, unit='°', warning_msg=''):
            if value is None:
                return {'label': label, 'value': None, 'status': 'unknown', 'unit': unit, 'message': '关键点不足'}
            if ok_range is not None:
                status = 'good' if ok_range[0] <= value <= ok_range[1] else 'warning'
            elif max_only is not None:
                status = 'good' if value <= max_only else 'warning'
            else:
                status = 'unknown'
            return {
                'label': label,
                'value': round(float(value), 1),
                'status': status,
                'unit': unit,
                'message': '' if status == 'good' else warning_msg,
            }

        if view == 'side':
            result['trunk_lean'] = pack('躯干前倾', features.get('trunk_lean_angle'), (3, 28), warning_msg='前倾角度异常')
            thigh_peak = _max_valid(features.get('left_thigh_angle'), features.get('right_thigh_angle'))
            knee_peak = _max_valid(features.get('left_knee_angle'), features.get('right_knee_angle'))
            result['thigh_swing'] = pack('大腿摆动', thigh_peak, (20, 85), warning_msg='摆动幅度异常')
            result['knee_support'] = pack('支撑腿膝角', knee_peak, (125, 178), warning_msg='支撑期膝角异常')
            result['cadence'] = pack('步频', cadence, (150, 205), unit='spm', warning_msg='步频异常') if cadence is not None else {
                'label': '步频', 'value': None, 'status': 'unknown', 'unit': 'spm', 'message': '时序不足'
            }
        elif view in {'front', 'back'}:
            result['shoulder_tilt'] = pack('肩部倾斜', features.get('shoulder_tilt'), max_only=10, warning_msg='左右肩不平衡')
            result['hip_tilt'] = pack('骨盆倾斜', features.get('hip_tilt'), max_only=10, warning_msg='左右骨盆不平衡')
            result['thigh_asym'] = pack('左右大腿不对称', features.get('thigh_asymmetry'), max_only=18, warning_msg='左右摆动不对称')
            result['cadence'] = pack('步频', cadence, (150, 205), unit='spm', warning_msg='步频异常') if cadence is not None else {
                'label': '步频', 'value': None, 'status': 'unknown', 'unit': 'spm', 'message': '时序不足'
            }
        else:
            base_eval = self.base.evaluate(features) or {}
            for key, item in base_eval.items():
                result[key] = {
                    'label': key,
                    'value': item.get('value'),
                    'status': item.get('status', 'unknown'),
                    'unit': '°',
                    'message': '' if item.get('status') == 'good' else '建议结合视角复核',
                }
        return result


def _max_valid(*values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return max(vals) if vals else None


def draw_skeleton(img, kps, color):
    connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        ('head_center', 'left_shoulder'),
        ('head_center', 'right_shoulder'),
    ]
    for a, b in connections:
        pa, pb = kps.get(a), kps.get(b)
        if isinstance(pa, tuple) and isinstance(pb, tuple):
            cv2.line(img, pa, pb, color, 2)


def draw_eval_panel(img, person_id, view_label, evaluation, box, color):
    x1, y1, x2, y2 = box
    panel_x = x2 + 8
    panel_y = y1

    items = [v for v in evaluation.values()]
    if not items:
        return

    line_h = 22
    panel_w = 250
    panel_h = (len(items) + 2) * line_h + 10
    img_h, img_w = img.shape[:2]
    if panel_x + panel_w > img_w:
        panel_x = max(0, x1 - panel_w - 8)
    if panel_y + panel_h > img_h:
        panel_y = max(0, img_h - panel_h - 5)

    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x - 4, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    cv2.putText(img, f'P{person_id} | view: {view_label}', (panel_x, panel_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for i, item in enumerate(items, start=1):
        y = panel_y + i * line_h + 18
        sc = STATUS_COLOR.get(item.get('status', 'unknown'), (200, 200, 200))
        value = item.get('value')
        unit = item.get('unit', '')
        text = f"{item.get('label', '')}: {'N/A' if value is None else str(value) + unit}"
        cv2.putText(img, text, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sc, 1)


def run_pipeline(video_path: str, output_dir: Optional[str] = None, sample_interval: int = 3, max_frames: Optional[int] = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'无法打开视频: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), f'{base_name}_dynamic_eval')
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    detector = YOLOPersonDetectorDynamic(model_path='yolov8n.pt', confidence_threshold=0.35)
    roi_manager = DynamicROIManager(width, height)
    analyzer = MediaPipePoseAnalyzerDynamic(model_complexity=1, detection_confidence=0.4, tracking_confidence=0.4)
    feat_calc = FrameFeatureCalculator()
    smoother = FeatureSmoother(alpha=0.5, window_size=5)
    time_calc = TemporalFeatureCalculator(fps=fps, window_seconds=2.0)
    view_classifier = RunningViewClassifier(history_size=8)
    rule_engine = MultiViewRuleEngine()

    print(f'视频: {width}x{height}, {fps:.1f}fps, {total}帧, {total/max(fps,1):.1f}秒')
    print('[模式] 非固定摄像头 | 动态ROI | 多视角(front/back/side)\n')

    frame_idx = 0
    saved = 0
    track_summary = defaultdict(lambda: {
        'frames': 0,
        'views': defaultdict(int),
        'warning_count': 0,
        'last_features': None,
        'last_cadence': None,
    })

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and saved >= max_frames:
            break
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue

        vis = frame.copy()
        people = detector.detect_people(frame, frame_idx)
        active_ids = []

        for person in people:
            pid = person['id']
            color = COLORS[(pid - 1) % len(COLORS)]
            det_box = person['box']
            roi_box = roi_manager.update_person_roi(pid, det_box, frame_idx)
            active_ids.append(pid)

            result = analyzer.analyze_person(frame, roi_box)
            cv2.rectangle(vis, (det_box[0], det_box[1]), (det_box[2], det_box[3]), color, 2)
            cv2.rectangle(vis, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (255, 255, 0), 1)
            cv2.putText(vis, f'P{pid} det {person["confidence"]:.2f}', (det_box[0], max(15, det_box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if result is None:
                cv2.putText(vis, 'Pose not detected', (det_box[0], det_box[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
                continue

            kps = result['keypoints']
            draw_skeleton(vis, kps, color)
            for coord in kps.values():
                if isinstance(coord, tuple):
                    cv2.circle(vis, coord, 3, color, -1)

            features = feat_calc.compute(kps)
            features = smoother.update(pid, features) if features else None
            time_calc.update(pid, frame_idx, kps)
            cadence = time_calc.compute_cadence(pid)
            view_label = view_classifier.classify(pid, kps)
            evaluation = rule_engine.evaluate(features, view_label, cadence)
            draw_eval_panel(vis, pid, view_label, evaluation, det_box, color)

            warn_count = sum(1 for v in evaluation.values() if v.get('status') == 'warning')
            track_summary[pid]['frames'] += 1
            track_summary[pid]['views'][view_label] += 1
            track_summary[pid]['warning_count'] += warn_count
            track_summary[pid]['last_features'] = features
            track_summary[pid]['last_cadence'] = cadence

        union_roi = roi_manager.build_union_roi(active_ids, padding=30)
        if union_roi:
            cv2.rectangle(vis, (union_roi[0], union_roi[1]), (union_roi[2], union_roi[3]), (0, 255, 255), 2)
            cv2.putText(vis, 'Global Dynamic ROI', (union_roi[0], max(20, union_roi[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        header = f'Frame {frame_idx} | {frame_idx/max(fps,1):.2f}s | persons={len(people)} | sample={sample_interval}f'
        cv2.rectangle(vis, (0, 0), (width, 34), (0, 0, 0), -1)
        cv2.putText(vis, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out_path = os.path.join(frames_dir, f'frame_{frame_idx:05d}.jpg')
        cv2.imwrite(out_path, vis)
        saved += 1
        if saved % 10 == 0:
            print(f'已处理 {saved} 帧')

        frame_idx += 1

    cap.release()

    summary_path = os.path.join(output_dir, 'summary.json')
    final_summary = {}
    for pid, item in track_summary.items():
        dominant_view = max(item['views'], key=item['views'].get) if item['views'] else 'unknown'
        final_summary[pid] = {
            'frames': item['frames'],
            'dominant_view': dominant_view,
            'view_histogram': dict(item['views']),
            'warning_count': item['warning_count'],
            'last_features': item['last_features'],
            'last_cadence': item['last_cadence'],
        }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 60)
    print('动态ROI多视角 Pipeline 完成')
    print('=' * 60)
    print(f'输出目录: {output_dir}')
    print(f'可视化帧: {frames_dir}')
    print(f'摘要结果: {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='视频路径')
    parser.add_argument('--sample_interval', type=int, default=3, help='采样间隔')
    parser.add_argument('--max_frames', type=int, default=None, help='最多处理帧数')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        sample_interval=args.sample_interval,
        max_frames=args.max_frames,
    )
