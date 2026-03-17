from __future__ import annotations

import math
from typing import Dict, List, Tuple

from ultralytics import YOLO


Box = Tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class YOLOPersonDetectorDynamic:
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        confidence_threshold: float = 0.35,
        max_missing: int = 20,
        match_distance_factor: float = 0.65,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.max_missing = max_missing
        self.match_distance_factor = match_distance_factor
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1

    def detect_people(self, img, frame_idx: int) -> List[dict]:
        results = self.model(img, classes=[0], verbose=False)
        detections: List[dict] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                if conf < self.confidence_threshold:
                    continue
                det_box = (int(x1), int(y1), int(x2), int(y2))
                cx = (det_box[0] + det_box[2]) / 2.0
                cy = (det_box[1] + det_box[3]) / 2.0
                detections.append({'box': det_box, 'confidence': conf, 'center': (cx, cy)})
        tracks = self._match(detections, frame_idx)
        tracks.sort(key=lambda t: t['id'])
        return tracks

    def _match(self, detections: List[dict], frame_idx: int) -> List[dict]:
        updated_ids = set()
        outputs = []

        for det in detections:
            det_box = det['box']
            cx, cy = det['center']
            best_id = None
            best_score = -1.0
            bw = max(1, det_box[2] - det_box[0])
            bh = max(1, det_box[3] - det_box[1])
            allowed_dist = self.match_distance_factor * math.hypot(bw, bh)

            for tid, track in self.tracks.items():
                if tid in updated_ids:
                    continue
                tcx, tcy = track['center']
                dist = math.hypot(cx - tcx, cy - tcy)
                ov = iou(det_box, track['box'])
                score = ov * 2.0 - dist / max(allowed_dist, 1.0)
                if dist <= allowed_dist and score > best_score:
                    best_score = score
                    best_id = tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            self.tracks[best_id] = {
                'id': best_id,
                'box': det_box,
                'confidence': det['confidence'],
                'center': det['center'],
                'last_seen': frame_idx,
            }
            updated_ids.add(best_id)
            outputs.append(self.tracks[best_id].copy())

        expired = [tid for tid, t in self.tracks.items() if frame_idx - t['last_seen'] > self.max_missing]
        for tid in expired:
            self.tracks.pop(tid, None)
        return outputs