import cv2
from ultralytics import YOLO
import numpy as np


class YOLOPersonDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        初始化YOLO检测器

        Args:
            model_path: YOLO模型路径
            confidence_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.previous_people = []  # 存储上一帧的人物信息
        self.position_threshold = 50  # 位置匹配阈值（像素）

    def detect_people(self, img):
        """
        检测画面中的所有人物

        Returns:
            list: 人物列表，每个元素包含 {'id', 'box', 'confidence', 'center_x'}
        """
        h, w = img.shape[:2]
        results = self.model(img, classes=[0], verbose=False)  # class 0 是 person

        current_detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                if conf > self.confidence_threshold:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    current_detections.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'center_x': center_x,
                        'center_y': center_y
                    })

        # 按从左到右排序（根据center_x）
        current_detections.sort(key=lambda x: x['center_x'])

        # 分配稳定的ID
        people_with_ids = self._assign_stable_ids(current_detections)

        return people_with_ids

    def _assign_stable_ids(self, current_detections):
        """
        为检测到的人物分配稳定的ID
        基于位置匹配，保持ID稳定性
        """
        people_with_ids = []

        if not self.previous_people:
            # 第一帧，直接按顺序分配ID
            for idx, detection in enumerate(current_detections):
                detection['id'] = idx + 1
                people_with_ids.append(detection)
        else:
            # 后续帧，尝试匹配
            used_ids = set()

            for detection in current_detections:
                matched_id = None
                min_distance = float('inf')

                # 尝试与上一帧的人物匹配
                for prev_person in self.previous_people:
                    if prev_person['id'] in used_ids:
                        continue

                    # 计算中心点距离
                    distance = abs(detection['center_x'] - prev_person['center_x'])

                    if distance < self.position_threshold and distance < min_distance:
                        min_distance = distance
                        matched_id = prev_person['id']

                # 如果匹配成功，使用原ID
                if matched_id is not None:
                    detection['id'] = matched_id
                    used_ids.add(matched_id)
                else:
                    # 没有匹配，分配新ID
                    existing_ids = {p['id'] for p in self.previous_people}
                    new_id = 1
                    while new_id in existing_ids or new_id in used_ids:
                        new_id += 1
                    detection['id'] = new_id
                    used_ids.add(new_id)

                people_with_ids.append(detection)

            # 重新按照从左到右排序
            people_with_ids.sort(key=lambda x: x['center_x'])

        self.previous_people = people_with_ids.copy()
        return people_with_ids

    def draw_detections(self, img, people_list):
        """
        在图像上绘制检测结果

        Args:
            img: 原始图像
            people_list: 人物列表
        """
        for person in people_list:
            box = person['box']
            person_id = person['id']
            conf = person['confidence']

            # 绘制边界框
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

            # 绘制ID标签
            label = f'Person {person_id} ({conf:.2f})'
            cv2.putText(img, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return img