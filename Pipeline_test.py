"""
整合Pipeline：YOLO → MediaPipe → 特征计算 → 规则评估 → 结果可视化
用法：python pipeline_test.py --video 视频路径.mp4
"""

import cv2
import numpy as np
import argparse
import os
import sys

PROJECT_PATH = ""#视频路径
sys.path.append(PROJECT_PATH)

from yolo_detector import YOLOPersonDetector
from mediapipe_analyzer import MediaPipePoseAnalyzer
from feature_calculator import FrameFeatureCalculator, TemporalFeatureCalculator, GaitRuleEngine

#ROI
CAMERA_CONFIG = {
    'roi1': (500, 380, 2100, 1260),   # 第一段镜头 (0~5s)：起跑区域
    'roi2': (802, 10, 1994, 956),      # 第二段镜头 (5s+)
    'switch_sec':  5.0,                # 镜头切换时间（秒）
    'sprint_sec':  6.0,                # 冲刺阶段开始时间（秒）
    'sprint_interval': 2,              # 冲刺阶段采样间隔（帧）
    'normal_interval': 5,              # 正常阶段采样间隔（帧）
}
#可视化的颜色
COLORS = [
    (255, 80,  80),   # 红
    (80,  200, 80),   # 绿
    (80,  80,  255),  # 蓝
    (255, 200, 0),    # 黄
    (200, 0,   255),  # 紫
]

STATUS_COLOR = {
    'good':    (0, 220, 0),
    'warning': (0, 140, 255),
    'unknown': (160, 160, 160),
}

#前端完成之后这块可以删
def draw_eval_panel(img, person_id, evaluation, cadence_result, box, color):
    x1, y1, x2, y2 = box
    panel_x = x2 + 8
    panel_y = y1

    # 收集要显示的条目（优先级排序）
    items = []
    for key in ['trunk_lean', 'thigh_swing', 'knee_support']:
        if key in evaluation:
            items.append(evaluation[key])
    if cadence_result:
        items.append(cadence_result)
    for key in ['elbow_左肘', 'elbow_右肘']:
        if key in evaluation:
            items.append(evaluation[key])

    if not items:
        return

    line_h   = 26
    panel_w  = 230
    panel_h  = len(items) * line_h + 20

    # 面板超出右边界时移到左侧
    img_h, img_w = img.shape[:2]
    if panel_x + panel_w > img_w:
        panel_x = x1 - panel_w - 8

    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x - 4, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # 标题行
    cv2.putText(img, f"P{person_id} Gait Eval",
                (panel_x, panel_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # 各指标行
    for i, item in enumerate(items):
        y  = panel_y + (i + 1) * line_h + 8
        sc = STATUS_COLOR.get(item.get('status', 'unknown'))

        icon    = 'OK' if item['status'] == 'good' else ('!!' if item['status'] == 'warning' else '--')
        val     = item.get('value')
        unit    = item.get('unit', '')
        val_str = f"{val}{unit}" if val is not None else 'N/A'
        label   = item.get('label', '')

        cv2.putText(img, f"[{icon}] {label}: {val_str}",
                    (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, sc, 1)

        # warning 时在下一行显示建议
        if item['status'] == 'warning':
            msg = item.get('message', '')
            if msg and msg != '正常':
                cv2.putText(img, f"     -> {msg}",
                            (panel_x, y + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 180, 255), 1)



# 非儿童端固定那几个摄像头的时候可以用
def auto_detect_roi(video_path, sample_frames=40, padding=60):
    """
    自动检测跑道中运动区域，作为分析ROI。
    原理：对前 N 帧做背景差分，找到所有运动前景区域，
         取最大包围框 + padding 作为最终 ROI。

    摄像头固定时这个方法非常稳定，无需用户操作。
    返回 (x1, y1, x2, y2) 或 None（检测失败时分析全画面）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 用 MOG2 背景建模，对运动敏感、对光线变化鲁棒
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=sample_frames,
        varThreshold=40,
        detectShadows=False
    )

    # 跳过第一帧（通常是静态起始帧），从第3帧开始采样
    step = max(1, total // sample_frames)
    all_contour_points = []

    print("[ROI] 自动检测运动区域...")

    frame_idx = 0
    processed = 0
    while cap.isOpened() and processed < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0 and frame_idx > 2:
            # 背景差分
            fg_mask = bg_subtractor.apply(frame)

            # 形态学处理去噪
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)

            # 找轮廓
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 2000:  # 过滤小噪声，只保留人体大小的运动区域
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    all_contour_points.extend([(x, y), (x + cw, y + ch)])

            processed += 1
        frame_idx += 1

    cap.release()

    if not all_contour_points:
        print("[ROI] 未检测到运动区域，将分析全画面\n")
        return None

    # 所有运动点的最大包围框
    xs = [p[0] for p in all_contour_points]
    ys = [p[1] for p in all_contour_points]

    x1 = max(0,     min(xs) - padding)
    y1 = max(0,     min(ys) - padding)
    x2 = min(w,     max(xs) + padding)
    y2 = min(h,     max(ys) + padding)

    print(f"[ROI] 自动识别运动区域: ({x1},{y1}) -> ({x2},{y2})")
    print(f"      覆盖画面比例: {(x2-x1)*(y2-y1)/(w*h)*100:.0f}%\n")
    return (x1, y1, x2, y2)


def person_in_roi(box, roi):
    """
    判断人物检测框的中心点是否在 ROI 内
    box: (x1, y1, x2, y2)
    roi: (rx1, ry1, rx2, ry2) 或 None
    """
    if roi is None:
        return True
    bx1, by1, bx2, by2 = box
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


# 主 Pipeline
def run_pipeline(video_path, sample_interval=5, max_frames=200,
                 roi1=None, roi2=None, switch_sec=5.0,
                 sprint_sec=6.0, sprint_interval=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {width}x{height}, {fps:.0f}fps, {total}帧, {total/fps:.1f}秒\n")

    # ROI 根据当前时间戳自动切换
    # switch_sec 之前用 roi1，之后用 roi2

    # 输出目录
    base    = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(os.path.dirname(video_path), f"{base}_eval")
    os.makedirs(out_dir, exist_ok=True)

    # 初始化所有模块
    detector    = YOLOPersonDetector(model_path='yolov8n.pt', confidence_threshold=0.35)
    analyzer    = MediaPipePoseAnalyzer(model_complexity=1,
                                        detection_confidence=0.4,
                                        tracking_confidence=0.4)
    feat_calc   = FrameFeatureCalculator()
    time_calc   = TemporalFeatureCalculator(fps=fps, window_seconds=2.0)
    rule_engine = GaitRuleEngine()

    # 统计
    stats     = {"frames": 0, "total_persons": 0, "pose_ok": 0, "warnings": 0}
    frame_idx = 0
    saved     = 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 根据时间段选择采样间隔
        # 第二镜头冲刺阶段（sprint_sec之后）采样密度大
        current_sec = frame_idx / fps
        if current_sec >= switch_sec and current_sec >= sprint_sec:
            active_interval = sprint_interval   # 冲刺阶段：每2帧采一次
        else:
            active_interval = sample_interval   # 正常阶段：每5帧采一次

        if frame_idx % active_interval != 0:
            frame_idx += 1
            continue

        vis = frame.copy()

        # ── Step 1: YOLO 检测所有人 ──
        people = detector.detect_people(frame)
        stats["total_persons"] += len(people)
        person_results = []

        # 根据当前时间戳选择对应 ROI
        current_roi = roi1 if (frame_idx / fps) < switch_sec else roi2
        people = [p for p in people if person_in_roi(p['box'], current_roi)]
        people.sort(key=lambda p: (p['box'][0] + p['box'][2]) / 2)

        for idx, p in enumerate(people):
            p['roi_id'] = idx + 1

        for person in people:
            pid = person['roi_id']
            box   = person['box']
            color = COLORS[(pid - 1) % len(COLORS)]

            # 画检测框和ID
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(vis, f"P{pid} {person['confidence']:.2f}",
                        (box[0], box[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ── Step 2: MediaPipe 姿态分析 ──
            result = analyzer.analyze_person(frame, box)
            if not result:
                cv2.putText(vis, "Pose not detected", (box[0], box[3] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                continue

            stats["pose_ok"] += 1
            kps = result['keypoints']

            # 画骨架和关键点
            draw_skeleton(vis, kps, color)
            for coord in kps.values():
                if isinstance(coord, tuple):
                    cv2.circle(vis, coord, 4, color, -1)

            # ── Step 3: 单帧特征计算 ──
            features = feat_calc.compute(kps)

            # ── Step 4: 时序更新 → 步频 ──
            time_calc.update(pid, frame_idx, kps)
            cadence        = time_calc.compute_cadence(pid)

            # ── Step 5: 规则评估 ──
            evaluation     = rule_engine.evaluate(features)
            cadence_result = rule_engine.evaluate_cadence(cadence) if cadence else None

            warn_count = sum(1 for v in evaluation.values() if v.get('status') == 'warning')
            stats["warnings"] += warn_count

            # ── Step 6: 绘制评估面板 ──
            draw_eval_panel(vis, pid, evaluation, cadence_result, box, color)

            person_results.append({
                'roi_id': pid,  # ROI编号
                'box': box,
                'features': features,
                'cadence': cadence,
                'warnings': warn_count,
            })

            person_results.sort(key=lambda x: x['roi_id'])

        # Draw current ROI boundary
        current_roi = roi1 if (frame_idx / fps) < switch_sec else roi2
        if current_roi:
            rx1, ry1, rx2, ry2 = current_roi
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            label = f"Zone 1 (0~{switch_sec:.0f}s)" if (frame_idx/fps) < switch_sec else f"Zone 2 ({switch_sec:.0f}s+)"
            cv2.putText(vis, label, (rx1, ry1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 顶部信息栏
        current_sec = frame_idx / fps
        active_interval = sprint_interval if (current_sec >= switch_sec and current_sec >= sprint_sec) else sample_interval
        phase = "SPRINT" if (current_sec >= switch_sec and current_sec >= sprint_sec) else ("CAM2" if current_sec >= switch_sec else "CAM1")
        info = (f"Frame {frame_idx} | {current_sec:.1f}s | {phase} @{active_interval}f | "
                f"Det:{len(people)} | Pose:{len(person_results)}")
        cv2.rectangle(vis, (0, 0), (width, 36), (0, 0, 0), -1)
        cv2.putText(vis, info, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 保存帧
        out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, vis)
        saved += 1
        stats["frames"] += 1

        if saved % 10 == 0:
            print(f"  已处理 {saved}/{max_frames} 帧...")

        frame_idx += 1

    cap.release()

    # ── 最终报告 ──
    print("\n" + "=" * 50)
    print("Pipeline 评估报告")
    print("=" * 50)
    print(f"Frames processed : {stats['frames']}")
    print(f"Total detections : {stats['total_persons']}")
    print(f"Pose success rate: {100*stats['pose_ok']/max(stats['total_persons'],1):.0f}%")
    print(f"Avg persons/frame: {stats['total_persons']/max(stats['frames'],1):.1f}")
    print(f"Total warnings   : {stats['warnings']}")
    print(f"\nSaved to: {out_dir}")


# 工具函数
def draw_skeleton(img, kps, color):
    connections = [
        ('left_shoulder',  'right_shoulder'),
        ('left_shoulder',  'left_elbow'),
        ('left_elbow',     'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow',    'right_wrist'),
        ('left_shoulder',  'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip',       'right_hip'),
        ('left_hip',       'left_knee'),
        ('left_knee',      'left_ankle'),
        ('right_hip',      'right_knee'),
        ('right_knee',     'right_ankle'),
        ('head_center',    'left_shoulder'),
        ('head_center',    'right_shoulder'),
    ]
    for a, b in connections:
        pa, pb = kps.get(a), kps.get(b)
        if isinstance(pa, tuple) and isinstance(pb, tuple):
            cv2.line(img, pa, pb, color, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True,  help="视频文件路径")
    parser.add_argument("--max_frames", type=int, default=200, help="最多处理帧数")
    # 以下参数可选，不填则使用顶部 CAMERA_CONFIG 里的默认值
    parser.add_argument("--roi1",       type=str, default=None, help="覆盖第一段ROI: x1,y1,x2,y2")
    parser.add_argument("--roi2",       type=str, default=None, help="覆盖第二段ROI: x1,y1,x2,y2")
    args = parser.parse_args()

    def parse_roi(s):
        if not s:
            return None
        try:
            parts = [int(v.strip()) for v in s.split(",")]
            assert len(parts) == 4
            return tuple(parts)
        except Exception as e:
            print(f"[ROI] 参数解析失败: {e}")
            return None

    # CLI参数优先，没传则用 CAMERA_CONFIG
    roi1        = parse_roi(args.roi1) or CAMERA_CONFIG['roi1']
    roi2        = parse_roi(args.roi2) or CAMERA_CONFIG['roi2']
    switch_sec  = CAMERA_CONFIG['switch_sec']
    sprint_sec  = CAMERA_CONFIG['sprint_sec']
    sprint_int  = CAMERA_CONFIG['sprint_interval']
    normal_int  = CAMERA_CONFIG['normal_interval']

    print(f"[ROI] Zone1 (0~{switch_sec}s): {roi1 or '全画面'}")
    print(f"[ROI] Zone2 ({switch_sec}s+):  {roi2 or '全画面（待配置）'}")

    run_pipeline(args.video, normal_int, args.max_frames,
                 roi1=roi1, roi2=roi2, switch_sec=switch_sec,
                 sprint_sec=sprint_sec, sprint_interval=sprint_int)
