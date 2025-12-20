import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
from config.config import POSE_CONFIG



class PoseEstimator:
    """姿态估计器类"""

    # MediaPipe关键点索引
    KEYPOINT_NAMES = {
        0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
        4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
        7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
        11: 'left_shoulder', 12: 'right_shoulder',
        13: 'left_elbow', 14: 'right_elbow',
        15: 'left_wrist', 16: 'right_wrist',
        17: 'left_pinky', 18: 'right_pinky',
        19: 'left_index', 20: 'right_index',
        21: 'left_thumb', 22: 'right_thumb',
        23: 'left_hip', 24: 'right_hip',
        25: 'left_knee', 26: 'right_knee',
        27: 'left_ankle', 28: 'right_ankle',
        29: 'left_heel', 30: 'right_heel',
        31: 'left_foot_index', 32: 'right_foot_index'
    }

    # 跑步分析关键关节
    RUNNING_KEYPOINTS = {
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'nose': 0
    }

    def __init__(self):
        """初始化姿态估计器"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            model_complexity=POSE_CONFIG['model_complexity'],
            min_detection_confidence=POSE_CONFIG['min_detection_confidence'],
            min_tracking_confidence=POSE_CONFIG['min_tracking_confidence'],
            static_image_mode=POSE_CONFIG['static_image_mode']
        )

    def process_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        处理视频帧序列
        Args:
            frames: 视频帧列表
        Returns:
            keypoints_sequence: 关键点时间序列
        """
        keypoints_sequence = []

        for idx, frame in enumerate(frames):
            # BGR转RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe姿态估计
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                keypoints = self._extract_keypoints(results.pose_landmarks, frame.shape)
                keypoints['frame_idx'] = idx
                keypoints['detected'] = True
            else:
                keypoints = self._get_empty_keypoints(frame.shape)
                keypoints['frame_idx'] = idx
                keypoints['detected'] = False

            keypoints_sequence.append(keypoints)

        return keypoints_sequence

    def _extract_keypoints(self, landmarks, image_shape: Tuple) -> Dict:
        """
        提取关键点坐标
        Args:
            landmarks: MediaPipe landmarks对象
            image_shape: 图像尺寸(H, W, C)
        Returns:
            keypoints字典
        """
        h, w = image_shape[:2]
        keypoints = {'landmarks': [], 'visibility': []}

        for idx, landmark in enumerate(landmarks.landmark):
            # 归一化坐标
            x_norm = landmark.x
            y_norm = landmark.y
            z_norm = landmark.z  # 深度信息
            visibility = landmark.visibility

            # 像素坐标
            x_pixel = int(x_norm * w)
            y_pixel = int(y_norm * h)

            keypoints['landmarks'].append({
                'id': idx,
                'name': self.KEYPOINT_NAMES.get(idx, f'point_{idx}'),
                'x': x_pixel,
                'y': y_pixel,
                'z': z_norm,
                'x_norm': x_norm,
                'y_norm': y_norm,
                'visibility': visibility
            })
            keypoints['visibility'].append(visibility)

        return keypoints

    def _get_empty_keypoints(self, image_shape: Tuple) -> Dict:
        """获取空关键点（检测失败时）"""
        return {
            'landmarks': [{'id': i, 'name': self.KEYPOINT_NAMES.get(i, f'point_{i}'),
                           'x': 0, 'y': 0, 'z': 0,
                           'x_norm': 0, 'y_norm': 0, 'visibility': 0}
                          for i in range(33)],
            'visibility': [0] * 33
        }

    def visualize_pose(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        可视化姿态（火柴人）
        Args:
            frame: 原始帧
            keypoints: 关键点数据
        Returns:
            可视化结果图像
        """
        vis_frame = frame.copy()

        if not keypoints['detected']:
            return vis_frame

        # 绘制关键点
        for kp in keypoints['landmarks']:
            if kp['visibility'] > 0.5:
                cv2.circle(vis_frame, (kp['x'], kp['y']), 5, (0, 255, 0), -1)

        # 绘制骨架连接
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start_kp = keypoints['landmarks'][start_idx]
            end_kp = keypoints['landmarks'][end_idx]

            if start_kp['visibility'] > 0.5 and end_kp['visibility'] > 0.5:
                cv2.line(vis_frame,
                         (start_kp['x'], start_kp['y']),
                         (end_kp['x'], end_kp['y']),
                         (255, 0, 0), 2)

        return vis_frame

    def get_keypoint_by_name(self, keypoints: Dict, name: str) -> Optional[Dict]:
        """根据名称获取关键点"""
        for kp in keypoints['landmarks']:
            if kp['name'] == name:
                return kp
        return None

    def get_running_keypoints(self, keypoints: Dict) -> Dict:
        """提取跑步分析所需的关键关节"""
        running_kps = {}
        for name, idx in self.RUNNING_KEYPOINTS.items():
            kp = keypoints['landmarks'][idx]
            if kp['visibility'] > 0.5:
                running_kps[name] = kp
        return running_kps

    def close(self):
        if self.pose is not None:
            self.pose.close()
            self.pose = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# 模块测试代码
if __name__ == "__main__":
    import sys
    from modules.video_processor import VideoProcessor

    if len(sys.argv) < 2:
        print("用法: python pose_estimator.py ")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        # 加载视频
        print("加载视频...")
        processor = VideoProcessor(video_path)
        frames, fps = processor.extract_frames(target_fps=30, max_frames=30)
        print(f"提取了 {len(frames)} 帧")

        # 姿态估计
        print("\n进行姿态估计...")
        estimator = PoseEstimator()
        keypoints_seq = estimator.process_frames(frames)

        detected_count = sum(1 for kp in keypoints_seq if kp['detected'])
        print(f"检测成功: {detected_count}/{len(keypoints_seq)} 帧")

        # 可视化第一帧
        # 可视化第一帧“检测成功”的姿态
        saved = False
        for frame, kps in zip(frames, keypoints_seq):
            if kps['detected']:
                vis_frame = estimator.visualize_pose(frame, kps)
                cv2.imwrite('pose_test_output.jpg', vis_frame)
                print("\n可视化结果已保存: pose_test_output.jpg")
                saved = True
                break

        if not saved:
            print("\n未找到可视化姿态帧，未生成图片")

        estimator.close()
        processor.release()
        print("\n模块测试完成!")

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)