# smartrun——基于YOLO与MediaPipe的跑姿分析
## 项目简介：
本项目实现了一套完整的目标检测 - 姿态估计 - 特征计算流水线，结合 YOLO 目标检测与 MediaPipe 姿态估计技术，可对图片 / 视频中的人体进行姿态关键点提取与特征分析
## 文件结构
```bash
├── Pipeline_test.py       # 主测试脚本，串联完整检测-分析-特征计算流程（需要注意的是pipeline_test仅限儿童端固定摄像头）
├── yolo_detector.py       # YOLO目标检测模块，用于框选人体目标
├── mediapipe_analyzer.py  # MediaPipe姿态关键点提取模块（平常测试用这个）
├── feature_calculator.py  # 姿态特征计算模块（角度、距离等指标）
└── README.md              # 项目说明文档
```
## 环境要求：
- python 3.8～3.11
- 核心依赖：
```bash
pip install opencv-python==4.8.0
pip install mediapipe==0.10.9
pip install ultralytics==8.0.200
pip install numpy==1.24.3
```
## 快速开始：
### 运行完整流水线
- 如果你是自己人，有我们提供的儿童端视频，则：
```bash
python Pipeline_test.py --input test_video.mp4 --output result_video.mp4
```
- else：
```bash
python feature_calculator.py --input test_video.mp4 --output result_video.mp4
```
### 模块单独调用：
- yolo：
```bash
from yolo_detector import YOLODetector
# 初始化YOLO检测器（使用yolov8n轻量化模型）
detector = YOLODetector(model_name="yolov8n.pt")
# 检测单张图片中的人体
detections = detector.detect_image("test_image.jpg")
# 输出检测到的人体边界框信息
print("检测到的人体边界框：", detections)
```
- mediapipe：
```bash
from mediapipe_analyzer import MediaPipeAnalyzer
# 初始化姿态分析器
analyzer = MediaPipeAnalyzer()
# 提取单张图片的姿态关键点
keypoints = analyzer.extract_keypoints("test_image.jpg")
print("姿态关键点坐标：", keypoints.shape)
```
