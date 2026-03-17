from __future__ import annotations

import cv2
import mediapipe as mp
from typing import Dict, Optional, Tuple


POSE_KEYPOINTS_33 = {
    0: 'nose', 7: 'left_ear', 8: 'right_ear',
    11: 'left_shoulder', 12: 'right_shoulder',
    13: 'left_elbow', 14: 'right_elbow',
    15: 'left_wrist', 16: 'right_wrist',
    23: 'left_hip', 24: 'right_hip',
    25: 'left_knee', 26: 'right_knee',
    27: 'left_ankle', 28: 'right_ankle',
    29: 'left_heel', 30: 'right_heel',
    31: 'left_foot_index', 32: 'right_foot_index',
}

SELECTED = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


class MediaPipePoseAnalyzerDynamic:
    def __init__(
        self,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        detection_confidence: float = 0.4,
        tracking_confidence: float = 0.4,
        visibility_threshold: float = 0.5,
    ) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.visibility_threshold = visibility_threshold

    def analyze_person(self, img, roi_box: Tuple[int, int, int, int]) -> Optional[dict]:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = roi_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = img[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return None

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(roi_rgb)
        if not results or not results.pose_landmarks:
            return None

        roi_h, roi_w = roi.shape[:2]
        keypoints: Dict[str, Tuple[int, int]] = {}
        landmarks = results.pose_landmarks.landmark

        for idx in SELECTED:
            lm = landmarks[idx]
            if lm.visibility < self.visibility_threshold:
                continue
            name = POSE_KEYPOINTS_33[idx]
            keypoints[name] = (int(lm.x * roi_w) + x1, int(lm.y * roi_h) + y1)

        if 'left_ear' in keypoints and 'right_ear' in keypoints:
            lx, ly = keypoints['left_ear']
            rx, ry = keypoints['right_ear']
            keypoints['head_center'] = ((lx + rx) // 2, (ly + ry) // 2)
        elif 'nose' in keypoints:
            keypoints['head_center'] = keypoints['nose']

        if not keypoints:
            return None

        return {
            'keypoints': keypoints,
            'landmarks': results.pose_landmarks,
            'roi_box': (x1, y1, x2, y2),
            'roi': roi,
        }
