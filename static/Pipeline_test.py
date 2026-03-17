
import cv2
import json
import numpy as np
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_PATH = ""#视频路径
sys.path.append(PROJECT_PATH)

from yolo_detector import YOLOPersonDetector
from mediapipe_analyzer import MediaPipePoseAnalyzer
from feature_calculator import (
    FrameFeatureCalculator,
    TemporalFeatureCalculator,
    GaitRuleEngine,
    FeatureSmoother,
)

Point = Tuple[int, int]
Polygon = List[Point]


# ─────────────────────────────────────────────
# 固定摄像头配置
# ─────────────────────────────────────────────
CAMERA_CONFIG = {
    "roi1": (500, 380, 2100, 1260),
    "roi2": (802, 10, 1994, 956),
    "switch_sec": 5.0,
    "sprint_sec": 6.0,
    "sprint_interval": 2,
    "normal_interval": 5,

    # 第一段镜头：从左到右 4 条跑道示例
    "roi1_lanes": {
        1: [(439, 1228), (915, 1167), (681, 441), (416, 446)],
        2: [(1010, 1150), (1406, 1122), (1287, 365), (922, 375)],
        3: [(1436, 1124), (1836, 1058), (1649, 301), (1335, 323)],
        4: [(1872, 1060), (2243, 1005), (2150, 290), (1830, 287)],
    },

    # 第二段镜头：从左到右 4 条跑道示例
    "roi2_lanes": {
        1: [(799, 739), (1133, 727), (1031, 47), (915, 46)],
        2: [(1144, 727), (1476, 712), (1153, 42), (1043, 46)],
        3: [(1487, 714), (1823, 708), (1274, 44), (1164, 44)],
        4: [(1843, 709), (1996, 696), (1396, 43), (1281, 46)],
    },
}

COLORS = [
    (255, 80, 80),
    (80, 200, 80),
    (80, 80, 255),
    (255, 200, 0),
    (200, 0, 255),
    (0, 220, 220),
]

STATUS_COLOR = {
    "good": (0, 220, 0),
    "warning": (0, 140, 255),
    "unknown": (160, 160, 160),
}


# ─────────────────────────────────────────────
# 评估面板
# ─────────────────────────────────────────────

def enrich_evaluation(evaluation: Optional[dict], cadence_result: Optional[dict]) -> List[dict]:
    if not evaluation:
        evaluation = {}

    items = []

    mapping = [
        ("trunk", "躯干前倾", "°", "躯干前倾异常"),
        ("thigh", "大腿摆动", "°", "大腿摆动不足或过大"),
        ("knee", "膝关节角", "°", "膝关节角度异常"),
    ]

    for key, label, unit, warn_msg in mapping:
        item = evaluation.get(key)
        if not item:
            continue
        one = dict(item)
        one.setdefault("label", label)
        one.setdefault("unit", unit)
        one.setdefault("message", "正常" if one.get("status") == "good" else warn_msg)
        items.append(one)

    if cadence_result:
        one = dict(cadence_result)
        one.setdefault("label", "步频")
        one.setdefault("unit", "spm")
        one.setdefault("message", "正常" if one.get("status") == "good" else "步频异常")
        items.append(one)

    return items


def draw_eval_panel(img, lane_id, evaluation, cadence_result, box, color):
    items = enrich_evaluation(evaluation, cadence_result)
    if not items:
        return

    x1, y1, x2, y2 = box
    panel_x = x2 + 8
    panel_y = y1
    line_h = 24
    panel_w = 220
    panel_h = len(items) * line_h + 20

    img_h, img_w = img.shape[:2]
    if panel_x + panel_w > img_w:
        panel_x = x1 - panel_w - 8

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (panel_x - 4, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    cv2.putText(
        img,
        f"Lane {lane_id} Eval",
        (panel_x, panel_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
    )

    for i, item in enumerate(items):
        y = panel_y + (i + 1) * line_h + 8
        status = item.get("status", "unknown")
        sc = STATUS_COLOR.get(status, STATUS_COLOR["unknown"])

        icon = "OK" if status == "good" else ("!!" if status == "warning" else "--")
        val = item.get("value")
        unit = item.get("unit", "")
        val_str = f"{val}{unit}" if val is not None else "N/A"
        label = item.get("label", "")

        cv2.putText(
            img,
            f"[{icon}] {label}: {val_str}",
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            sc,
            1,
        )


# ─────────────────────────────────────────────
# 几何与跑道归属
# ─────────────────────────────────────────────

def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0


def polygon_center(polygon: Polygon) -> Point:
    pts = np.array(polygon, dtype=np.float32)
    return int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))


def point_to_segment_distance(point: Point, a: Point, b: Point) -> float:
    p = np.array(point, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-6:
        return float(np.linalg.norm(p - a))
    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def get_person_anchor_point(box, kps=None) -> Point:
    if kps:
        la = kps.get("left_ankle")
        ra = kps.get("right_ankle")
        if isinstance(la, tuple) and isinstance(ra, tuple):
            return int((la[0] + ra[0]) / 2), int((la[1] + ra[1]) / 2)
        if isinstance(la, tuple):
            return la
        if isinstance(ra, tuple):
            return ra

    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def assign_lane(box, lane_polygons: Dict[int, Polygon], kps=None):
    anchor = get_person_anchor_point(box, kps)

    for lane_id, polygon in lane_polygons.items():
        if point_in_polygon(anchor, polygon):
            return lane_id, anchor

    # 容错：如果锚点没落进任何 polygon，就找最近的跑道下边界中心线
    best_lane = None
    best_dist = 1e18
    for lane_id, polygon in lane_polygons.items():
        if len(polygon) < 2:
            continue
        dist = point_to_segment_distance(anchor, polygon[0], polygon[1])
        if dist < best_dist:
            best_dist = dist
            best_lane = lane_id

    return best_lane, anchor


def person_in_roi(box, roi):
    if roi is None:
        return True
    bx1, by1, bx2, by2 = box
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def deduplicate_by_lane(people: List[dict]) -> List[dict]:
    lane_best = {}
    for person in people:
        lane_id = person["lane_id"]
        x1, y1, x2, y2 = person["box"]
        area = max(1, (x2 - x1) * (y2 - y1))
        if lane_id not in lane_best or area > lane_best[lane_id]["area"]:
            lane_best[lane_id] = {"person": person, "area": area}
    return [v["person"] for v in lane_best.values()]


def draw_lane_polygons(img, lane_polygons: Dict[int, Polygon]):
    for lane_id, polygon in lane_polygons.items():
        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 255), 2)
        cx, cy = polygon_center(polygon)
        cv2.putText(
            img,
            f"Lane {lane_id}",
            (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )


def draw_skeleton(img, kps, color):
    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("head_center", "left_shoulder"),
        ("head_center", "right_shoulder"),
    ]
    for a, b in connections:
        pa, pb = kps.get(a), kps.get(b)
        if isinstance(pa, tuple) and isinstance(pb, tuple):
            cv2.line(img, pa, pb, color, 2)


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def run_pipeline(
    video_path,
    sample_interval=5,
    max_frames=200,
    roi1=None,
    roi2=None,
    switch_sec=5.0,
    sprint_sec=6.0,
    sprint_interval=2,
    roi1_lanes: Optional[Dict[int, Polygon]] = None,
    roi2_lanes: Optional[Dict[int, Polygon]] = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {width}x{height}, {fps:.0f}fps, {total}帧, {total / fps:.1f}秒\n")

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(os.path.dirname(video_path), f"{base}_lane_eval")
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "lane_results.jsonl")

    detector = YOLOPersonDetector(model_path="yolov8n.pt", confidence_threshold=0.35)
    analyzer = MediaPipePoseAnalyzer(
        model_complexity=1,
        detection_confidence=0.4,
        tracking_confidence=0.4,
    )
    feat_calc = FrameFeatureCalculator()
    time_calc = TemporalFeatureCalculator(fps=fps, window_seconds=2.0)
    rule_engine = GaitRuleEngine()
    smoother = FeatureSmoother(alpha=0.5, window_size=5)

    track_stability = {}
    STABLE_FRAMES = 4
    stats = {"frames": 0, "total_persons": 0, "pose_ok": 0, "warnings": 0}
    frame_idx = 0
    saved = 0

    with open(jsonl_path, "w", encoding="utf-8") as fout:
        while cap.isOpened() and saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps
            if current_sec >= switch_sec and current_sec >= sprint_sec:
                active_interval = sprint_interval
            else:
                active_interval = sample_interval

            if frame_idx % active_interval != 0:
                frame_idx += 1
                continue

            vis = frame.copy()
            people = detector.detect_people(frame)
            stats["total_persons"] += len(people)
            person_results = []

            current_roi = roi1 if current_sec < switch_sec else roi2
            current_lane_polygons = roi1_lanes if current_sec < switch_sec else roi2_lanes

            people = [p for p in people if person_in_roi(p["box"], current_roi)]

            valid_people = []
            current_ids = []

            for p in people:
                box = p["box"]
                result = analyzer.analyze_person(frame, box)
                if not result:
                    continue

                stats["pose_ok"] += 1
                kps = result["keypoints"]
                lane_id, anchor = assign_lane(box, current_lane_polygons, kps)
                if lane_id is None:
                    continue

                p["pose_result"] = result
                p["kps"] = kps
                p["lane_id"] = lane_id
                p["anchor"] = anchor
                valid_people.append(p)

            valid_people = deduplicate_by_lane(valid_people)
            valid_people.sort(key=lambda x: x["lane_id"])

            for p in valid_people:
                lane_id = p["lane_id"]
                current_ids.append(lane_id)
                track_stability[lane_id] = track_stability.get(lane_id, 0) + 1

            for person in valid_people:
                lane_id = person["lane_id"]
                if track_stability.get(lane_id, 0) < STABLE_FRAMES:
                    continue

                box = person["box"]
                color = COLORS[(lane_id - 1) % len(COLORS)]
                kps = person["kps"]
                anchor = person["anchor"]

                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(
                    vis,
                    f"L{lane_id} {person['confidence']:.2f}",
                    (box[0], box[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                cv2.circle(vis, anchor, 5, (255, 255, 255), -1)

                draw_skeleton(vis, kps, color)
                for coord in kps.values():
                    if isinstance(coord, tuple):
                        cv2.circle(vis, coord, 4, color, -1)

                features = feat_calc.compute(kps)
                features = smoother.update(lane_id, features)
                if features is None:
                    continue

                for k, v in list(features.items()):
                    if isinstance(v, (int, float)) and abs(v) > 300:
                        features[k] = None

                time_calc.update(lane_id, frame_idx, kps)
                cadence = time_calc.compute_cadence(lane_id)

                evaluation = rule_engine.evaluate(features) or {}
                cadence_result = rule_engine.evaluate_cadence(cadence) if cadence else None

                warn_count = sum(
                    1 for v in evaluation.values() if isinstance(v, dict) and v.get("status") == "warning"
                )
                if cadence_result and cadence_result.get("status") == "warning":
                    warn_count += 1
                stats["warnings"] += warn_count

                draw_eval_panel(vis, lane_id, evaluation, cadence_result, box, color)

                person_results.append(
                    {
                        "lane_id": lane_id,
                        "box": [int(v) for v in box],
                        "anchor": [int(anchor[0]), int(anchor[1])],
                        "features": features,
                        "evaluation": enrich_evaluation(evaluation, cadence_result),
                        "cadence": cadence,
                        "warnings": warn_count,
                    }
                )

            person_results.sort(key=lambda x: x["lane_id"])

            disappeared = set(track_stability.keys()) - set(current_ids)
            for d in disappeared:
                track_stability.pop(d, None)

            if current_roi:
                rx1, ry1, rx2, ry2 = current_roi
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
                label = f"Zone 1 (0~{switch_sec:.0f}s)" if current_sec < switch_sec else f"Zone 2 ({switch_sec:.0f}s+)"
                cv2.putText(
                    vis,
                    label,
                    (rx1, ry1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

            draw_lane_polygons(vis, current_lane_polygons)

            phase = "SPRINT" if (current_sec >= switch_sec and current_sec >= sprint_sec) else ("CAM2" if current_sec >= switch_sec else "CAM1")
            info = (
                f"Frame {frame_idx} | {current_sec:.1f}s | {phase} @{active_interval}f | "
                f"Det:{len(people)} | LanePose:{len(person_results)}"
            )
            cv2.rectangle(vis, (0, 0), (width, 36), (0, 0, 0), -1)
            cv2.putText(
                vis,
                info,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, vis)

            frame_record = {
                "frame_idx": frame_idx,
                "time_sec": round(current_sec, 3),
                "phase": phase,
                "zone": 1 if current_sec < switch_sec else 2,
                "results_by_lane": person_results,
            }
            fout.write(json.dumps(frame_record, ensure_ascii=False) + "\n")

            saved += 1
            stats["frames"] += 1
            if saved % 10 == 0:
                print(f"  已处理 {saved}/{max_frames} 帧...")

            frame_idx += 1

    cap.release()

    print("\n" + "=" * 50)
    print("Pipeline 跑道版评估报告")
    print("=" * 50)
    print(f"Frames processed : {stats['frames']}")
    print(f"Total detections : {stats['total_persons']}")
    print(f"Pose success rate: {100 * stats['pose_ok'] / max(stats['total_persons'], 1):.0f}%")
    print(f"Avg persons/frame: {stats['total_persons'] / max(stats['frames'], 1):.1f}")
    print(f"Total warnings   : {stats['warnings']}")
    print(f"Saved frames to  : {out_dir}")
    print(f"Lane JSONL to    : {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--max_frames", type=int, default=200, help="最多处理帧数")
    parser.add_argument("--roi1", type=str, default=None, help="覆盖第一段ROI: x1,y1,x2,y2")
    parser.add_argument("--roi2", type=str, default=None, help="覆盖第二段ROI: x1,y1,x2,y2")
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

    roi1 = parse_roi(args.roi1) or CAMERA_CONFIG["roi1"]
    roi2 = parse_roi(args.roi2) or CAMERA_CONFIG["roi2"]
    switch_sec = CAMERA_CONFIG["switch_sec"]
    sprint_sec = CAMERA_CONFIG["sprint_sec"]
    sprint_int = CAMERA_CONFIG["sprint_interval"]
    normal_int = CAMERA_CONFIG["normal_interval"]
    roi1_lanes = CAMERA_CONFIG["roi1_lanes"]
    roi2_lanes = CAMERA_CONFIG["roi2_lanes"]

    print(f"[ROI] Zone1 (0~{switch_sec}s): {roi1 or '全画面'}")
    print(f"[ROI] Zone2 ({switch_sec}s+):  {roi2 or '全画面'}")
    print(f"[Lane] Zone1 lanes: {json.dumps(roi1_lanes, ensure_ascii=False)}")
    print(f"[Lane] Zone2 lanes: {json.dumps(roi2_lanes, ensure_ascii=False)}")

    run_pipeline(
        args.video,
        normal_int,
        args.max_frames,
        roi1=roi1,
        roi2=roi2,
        switch_sec=switch_sec,
        sprint_sec=sprint_sec,
        sprint_interval=sprint_int,
        roi1_lanes=roi1_lanes,
        roi2_lanes=roi2_lanes,
    )
