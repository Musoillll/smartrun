import cv2
import mediapipe as mp
import numpy as np

# 33 个主要关键点索引
POSE_KEYPOINTS_33 = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}

# 用于计算的关键点索引
SELECTED_INDICES_FOR_CALC = [
    7, 8,  # 耳朵用于计算头部中心
    11, 12, 13, 14, 15, 16,  # 左/右 肩, 肘, 腕
    23, 24, 25, 26, 27, 28  # 左/右 髋, 膝, 踝
]

# 最终输出的关键点名称列表
FINAL_KEYPOINT_NAMES = {
    'head_center': 'Head Center (Calculated)',
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle"
}


class MediaPipePoseAnalyzer:
    def __init__(self, model_complexity=2, smooth_landmarks=True,
                 detection_confidence=0.5, tracking_confidence=0.5):
        """
        初始化MediaPipe姿态分析器

        Args:
            model_complexity: 模型复杂度 (0, 1, 2)
            smooth_landmarks: 是否平滑关键点
            detection_confidence: 检测置信度阈值
            tracking_confidence: 跟踪置信度阈值
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.selected_indices = SELECTED_INDICES_FOR_CALC
        self.final_names = FINAL_KEYPOINT_NAMES

    def analyze_person(self, img, person_box):
        """
        分析单个人物的姿态

        Args:
            img: 完整图像
            person_box: 人物边界框 (x1, y1, x2, y2)

        Returns:
            dict: 包含关键点坐标的字典，如果检测失败返回None
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = person_box

        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 检查边界框有效性
        if x2 <= x1 or y2 <= y1:
            return None

        # 提取ROI
        roi = img[y1:y2, x1:x2].copy()

        if roi.size == 0:
            return None

        # MediaPipe处理
        roi_h, roi_w = roi.shape[:2]
        imgRGB = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        if not results or not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        keypoints_coords = {}

        # 计算头部中心
        l_ear = landmarks[7]
        r_ear = landmarks[8]

        if l_ear.visibility > 0.6 and r_ear.visibility > 0.6:
            head_x = int((l_ear.x + r_ear.x) / 2 * roi_w) + x1
            head_y = int((l_ear.y + r_ear.y) / 2 * roi_h) + y1
            keypoints_coords['head_center'] = (head_x, head_y)
        else:
            keypoints_coords['head_center'] = "Not Found (Low Ear Visibility)"

        # 提取躯干和四肢关键点
        for index in self.selected_indices:
            if index == 7 or index == 8:
                continue
            name = POSE_KEYPOINTS_33[index]
            if landmarks[index].visibility > 0.6:
                x = int(landmarks[index].x * roi_w) + x1
                y = int(landmarks[index].y * roi_h) + y1
                keypoints_coords[name] = (x, y)
            else:
                keypoints_coords[name] = "Not Found (Low Visibility)"

        return {
            'keypoints': keypoints_coords,
            'landmarks': results.pose_landmarks,
            'roi_box': (x1, y1, x2, y2),
            'roi': roi
        }

    def draw_pose(self, img, analysis_result):
        """
        在图像上绘制姿态关键点

        Args:
            img: 原始图像
            analysis_result: analyze_person返回的结果
        """
        if analysis_result is None:
            return img

        keypoints = analysis_result['keypoints']
        landmarks = analysis_result['landmarks']
        x1, y1, x2, y2 = analysis_result['roi_box']
        roi = analysis_result['roi']

        # 在ROI上绘制关键点
        roi_with_landmarks = roi.copy()
        self.mp_drawing.draw_landmarks(
            roi_with_landmarks,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
        )

        # 贴回原图
        h, w = img.shape[:2]
        roi_h, roi_w = roi_with_landmarks.shape[:2]
        end_y = min(y1 + roi_h, h)
        end_x = min(x1 + roi_w, w)
        actual_h = end_y - y1
        actual_w = end_x - x1

        img[y1:end_y, x1:end_x] = roi_with_landmarks[:actual_h, :actual_w]

        # 绘制头部中心点
        if isinstance(keypoints.get('head_center'), tuple):
            head_x, head_y = keypoints['head_center']
            cv2.circle(img, (head_x, head_y), 8, (0, 0, 255), -1)

        return img