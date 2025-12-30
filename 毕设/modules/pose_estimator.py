# modules/pose_estimator.py
"""
姿态估计器模块 - 统一接口设计
支持多种后端：
- MediaPipe（2D后备方案）
- MMPose + MotionBERT（3D主方案）
"""
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from config.config import POSE_CONFIG

# 导入3D估计器
try:
    from modules.pose_estimator_3d import (
        create_pose_estimator_3d,
        BasePoseEstimator3D,
        MMPose3DEstimator,
        MediaPipeFallback
    )
    HAS_3D_ESTIMATOR = True
except ImportError:
    HAS_3D_ESTIMATOR = False


class BasePoseEstimator(ABC):
    """
    姿态估计器基类
    定义统一接口，方便后续替换不同的姿态估计后端
    """

    # 统一的关键点定义（基于COCO格式）
    KEYPOINT_NAMES = {
        0: 'nose',
        1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
        4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
        7: 'left_ear', 8: 'right_ear',
        9: 'mouth_left', 10: 'mouth_right',
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

    @abstractmethod
    def process_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """处理视频帧序列，返回关键点时间序列"""
        pass

    @abstractmethod
    def process_single_frame(self, frame: np.ndarray) -> Dict:
        """处理单帧图像"""
        pass

    @abstractmethod
    def visualize_pose(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """可视化姿态"""
        pass

    @abstractmethod
    def close(self):
        """释放资源"""
        pass

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


class MediaPipePoseEstimator(BasePoseEstimator):
    """
    MediaPipe姿态估计器
    当前主要使用的后端
    """

    def __init__(self, config: Dict = None):
        """初始化MediaPipe姿态估计器"""
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        config = config or POSE_CONFIG

        self.pose = self.mp_pose.Pose(
            model_complexity=config.get('model_complexity', 1),
            min_detection_confidence=config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=config.get('min_tracking_confidence', 0.5),
            static_image_mode=config.get('static_image_mode', False)
        )

        self.backend = 'mediapipe'
        self.num_keypoints = 33

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
            keypoints = self.process_single_frame(frame)
            keypoints['frame_idx'] = idx
            keypoints_sequence.append(keypoints)

        return keypoints_sequence

    def process_single_frame(self, frame: np.ndarray) -> Dict:
        """
        处理单帧图像
        Args:
            frame: BGR格式的图像
        Returns:
            关键点数据字典
        """
        # BGR转RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe姿态估计
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = self._extract_keypoints(results.pose_landmarks, frame.shape)
            keypoints['detected'] = True
        else:
            keypoints = self._get_empty_keypoints(frame.shape)
            keypoints['detected'] = False

        return keypoints

    def _extract_keypoints(self, landmarks, image_shape: Tuple) -> Dict:
        """
        提取关键点坐标
        """
        h, w = image_shape[:2]
        keypoints = {'landmarks': [], 'visibility': []}

        for idx, landmark in enumerate(landmarks.landmark):
            # 归一化坐标
            x_norm = landmark.x
            y_norm = landmark.y
            z_norm = landmark.z
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
            'landmarks': [
                {
                    'id': i,
                    'name': self.KEYPOINT_NAMES.get(i, f'point_{i}'),
                    'x': 0, 'y': 0, 'z': 0,
                    'x_norm': 0, 'y_norm': 0,
                    'visibility': 0
                }
                for i in range(33)
            ],
            'visibility': [0] * 33
        }

    def visualize_pose(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        可视化姿态（火柴人）
        """
        vis_frame = frame.copy()

        if not keypoints.get('detected', False):
            return vis_frame

        # 绘制关键点
        for kp in keypoints['landmarks']:
            if kp['visibility'] > 0.5:
                color = self._get_keypoint_color(kp['id'])
                cv2.circle(vis_frame, (kp['x'], kp['y']), 5, color, -1)

        # 绘制骨架连接
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start_kp = keypoints['landmarks'][start_idx]
            end_kp = keypoints['landmarks'][end_idx]

            if start_kp['visibility'] > 0.5 and end_kp['visibility'] > 0.5:
                color = self._get_connection_color(start_idx, end_idx)
                cv2.line(vis_frame,
                         (start_kp['x'], start_kp['y']),
                         (end_kp['x'], end_kp['y']),
                         color, 2)

        return vis_frame

    def _get_keypoint_color(self, kp_id: int) -> Tuple[int, int, int]:
        """根据关键点ID获取颜色"""
        # 躯干：蓝色
        if kp_id in [11, 12, 23, 24]:
            return (255, 128, 0)
        # 腿部：绿色
        elif kp_id in [25, 26, 27, 28, 29, 30, 31, 32]:
            return (0, 255, 0)
        # 手臂：红色
        elif kp_id in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            return (0, 0, 255)
        # 头部：黄色
        else:
            return (0, 255, 255)

    def _get_connection_color(self, start_id: int, end_id: int) -> Tuple[int, int, int]:
        """根据连接获取颜色"""
        # 腿部连接
        leg_ids = {23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
        if start_id in leg_ids and end_id in leg_ids:
            return (0, 200, 0)
        # 手臂连接
        arm_ids = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
        if start_id in arm_ids and end_id in arm_ids:
            return (200, 0, 0)
        # 躯干连接
        return (200, 128, 0)

    def close(self):
        """释放资源"""
        if self.pose is not None:
            self.pose.close()
            self.pose = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MMPosePoseEstimator(BasePoseEstimator):
    """
    MMPose姿态估计器（预留接口）
    需要安装mmpose和mmcv才能使用
    """

    def __init__(self, config: Dict = None):
        """
        初始化MMPose姿态估计器
        注意：需要安装mmpose和mmcv
        """
        self.backend = 'mmpose'
        self.model = None
        self.initialized = False

        try:
            # 尝试导入mmpose
            from mmpose.apis import init_model, inference_topdown
            self._init_model = init_model
            self._inference = inference_topdown

            # 默认配置
            config = config or {}
            model_config = config.get('config_file',
                'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py')
            checkpoint = config.get('checkpoint',
                'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth')

            self.model = self._init_model(model_config, checkpoint)
            self.initialized = True
            print("MMPose初始化成功")

        except ImportError:
            print("警告: MMPose未安装，该后端不可用")
            print("安装命令: pip install mmpose mmcv-full")
        except Exception as e:
            print(f"警告: MMPose初始化失败 - {e}")

    def process_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """处理视频帧序列"""
        if not self.initialized:
            raise RuntimeError("MMPose未初始化")

        keypoints_sequence = []
        for idx, frame in enumerate(frames):
            keypoints = self.process_single_frame(frame)
            keypoints['frame_idx'] = idx
            keypoints_sequence.append(keypoints)

        return keypoints_sequence

    def process_single_frame(self, frame: np.ndarray) -> Dict:
        """处理单帧图像"""
        if not self.initialized:
            return self._get_empty_keypoints(frame.shape)

        # MMPose推理逻辑（预留）
        # results = self._inference(self.model, frame, ...)
        # 转换为统一格式

        return self._get_empty_keypoints(frame.shape)

    def _get_empty_keypoints(self, image_shape: Tuple) -> Dict:
        """获取空关键点"""
        return {
            'landmarks': [
                {
                    'id': i,
                    'name': self.KEYPOINT_NAMES.get(i, f'point_{i}'),
                    'x': 0, 'y': 0, 'z': 0,
                    'x_norm': 0, 'y_norm': 0,
                    'visibility': 0
                }
                for i in range(33)
            ],
            'visibility': [0] * 33,
            'detected': False
        }

    def visualize_pose(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """可视化姿态"""
        return frame.copy()

    def close(self):
        """释放资源"""
        self.model = None
        self.initialized = False


# 工厂函数
def create_pose_estimator(backend: str = None, config: Dict = None) -> BasePoseEstimator:
    """
    创建姿态估计器的工厂函数

    Args:
        backend: 后端类型 ('mediapipe', 'mmpose', 'mmpose_3d')
                 如果为None，使用配置文件中的默认值
        config: 配置字典

    Returns:
        姿态估计器实例
    """
    if backend is None:
        backend = POSE_CONFIG.get('backend', 'mediapipe')

    # 3D后端（推荐）
    if backend == 'mmpose_3d':
        if HAS_3D_ESTIMATOR:
            try:
                estimator_3d = create_pose_estimator_3d(config)
                # 包装为兼容接口
                return PoseEstimator3DWrapper(estimator_3d)
            except Exception as e:
                print(f"3D估计器创建失败: {e}")
                print("回退到MediaPipe")
                return MediaPipePoseEstimator(config)
        else:
            print("3D估计器模块不可用，使用MediaPipe")
            return MediaPipePoseEstimator(config)

    # 2D后端
    if backend == 'mediapipe':
        return MediaPipePoseEstimator(config)
    elif backend == 'mmpose':
        return MMPosePoseEstimator(config)
    else:
        raise ValueError(f"不支持的后端: {backend}")


class PoseEstimator3DWrapper(BasePoseEstimator):
    """
    3D估计器的包装类

    将3D估计器适配到原有的2D接口，同时提供3D数据访问
    """

    def __init__(self, estimator_3d):
        self.estimator_3d = estimator_3d
        self.backend = 'mmpose_3d'
        self.num_keypoints = 17
        self._last_result = None

        # 关键点映射：COCO 17点 -> MediaPipe 33点风格的索引
        # 用于兼容原有代码
        self._coco_to_mp_indices = {
            0: 0,    # nose
            5: 11,   # left_shoulder
            6: 12,   # right_shoulder
            7: 13,   # left_elbow
            8: 14,   # right_elbow
            9: 15,   # left_wrist
            10: 16,  # right_wrist
            11: 23,  # left_hip
            12: 24,  # right_hip
            13: 25,  # left_knee
            14: 26,  # right_knee
            15: 27,  # left_ankle
            16: 28,  # right_ankle
        }

    def process_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """处理视频帧序列，返回兼容格式的关键点"""
        # 调用3D估计器
        result = self.estimator_3d.process_video(frames)
        self._last_result = result

        # 转换为兼容格式
        keypoints_sequence = []
        n_frames = len(frames)

        for i in range(n_frames):
            kp_2d = result['keypoints_2d'][i]
            kp_3d = result['keypoints_3d'][i]
            conf = result['confidences'][i]

            # 构建兼容MediaPipe格式的输出
            landmarks = []
            h, w = frames[i].shape[:2]

            for j in range(17):
                # 获取对应的MediaPipe风格索引
                mp_idx = self._coco_to_mp_indices.get(j, j)

                landmarks.append({
                    'id': mp_idx,
                    'name': self.KEYPOINT_NAMES.get(mp_idx, f'point_{mp_idx}'),
                    'x': int(kp_2d[j, 0]),
                    'y': int(kp_2d[j, 1]),
                    'z': float(kp_3d[j, 2]),
                    'x_norm': kp_2d[j, 0] / w if w > 0 else 0,
                    'y_norm': kp_2d[j, 1] / h if h > 0 else 0,
                    'z_norm': float(kp_3d[j, 2]),
                    'visibility': float(conf[j]),
                    # 3D坐标（新增）
                    'x_3d': float(kp_3d[j, 0]),
                    'y_3d': float(kp_3d[j, 1]),
                    'z_3d': float(kp_3d[j, 2]),
                })

            # 填充到33个点（兼容MediaPipe格式）
            while len(landmarks) < 33:
                landmarks.append({
                    'id': len(landmarks),
                    'name': f'point_{len(landmarks)}',
                    'x': 0, 'y': 0, 'z': 0,
                    'x_norm': 0, 'y_norm': 0, 'z_norm': 0,
                    'visibility': 0,
                    'x_3d': 0, 'y_3d': 0, 'z_3d': 0,
                })

            keypoints_sequence.append({
                'landmarks': landmarks,
                'visibility': [lm['visibility'] for lm in landmarks],
                'detected': i in result['detected_frames'],
                'frame_idx': i,
                # 3D数据直接访问
                'keypoints_3d': kp_3d,
                'keypoints_2d': kp_2d,
            })

        return keypoints_sequence

    def process_single_frame(self, frame: np.ndarray) -> Dict:
        """处理单帧"""
        result = self.process_frames([frame])
        return result[0] if result else self._get_empty_keypoints(frame.shape)

    def get_3d_keypoints(self) -> Optional[np.ndarray]:
        """获取最后一次处理的3D关键点"""
        if self._last_result is not None:
            return self._last_result['keypoints_3d']
        return None

    def get_2d_keypoints(self) -> Optional[np.ndarray]:
        """获取最后一次处理的2D关键点"""
        if self._last_result is not None:
            return self._last_result['keypoints_2d']
        return None

    def visualize_pose(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """可视化姿态"""
        if hasattr(self.estimator_3d, 'visualize_pose'):
            kp_2d = keypoints.get('keypoints_2d')
            conf = keypoints.get('visibility', [])
            if kp_2d is not None and len(conf) >= 17:
                return self.estimator_3d.visualize_pose(
                    frame, kp_2d[:17], np.array(conf[:17])
                )

        # 后备可视化
        return self._fallback_visualize(frame, keypoints)

    def _fallback_visualize(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """后备可视化方法"""
        vis_frame = frame.copy()
        if not keypoints.get('detected', False):
            return vis_frame

        for lm in keypoints['landmarks'][:17]:
            if lm['visibility'] > 0.5:
                cv2.circle(vis_frame, (lm['x'], lm['y']), 4, (0, 255, 0), -1)

        return vis_frame

    def _get_empty_keypoints(self, image_shape) -> Dict:
        """返回空关键点"""
        return {
            'landmarks': [
                {'id': i, 'name': f'point_{i}',
                 'x': 0, 'y': 0, 'z': 0,
                 'x_norm': 0, 'y_norm': 0, 'z_norm': 0,
                 'visibility': 0,
                 'x_3d': 0, 'y_3d': 0, 'z_3d': 0}
                for i in range(33)
            ],
            'visibility': [0] * 33,
            'detected': False,
            'keypoints_3d': np.zeros((17, 3)),
            'keypoints_2d': np.zeros((17, 2)),
        }

    def close(self):
        """释放资源"""
        if self.estimator_3d:
            self.estimator_3d.close()


# 兼容性别名
PoseEstimator = MediaPipePoseEstimator


# 模块测试
if __name__ == "__main__":
    import sys
    from modules.video_processor import VideoProcessor

    print("=" * 60)
    print("测试姿态估计模块")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("用法: python pose_estimator.py <video_path>")
        print("\n测试工厂函数...")

        # 测试MediaPipe
        estimator = create_pose_estimator('mediapipe')
        print(f"创建成功: {estimator.backend}")
        estimator.close()

        print("\n✅ 模块基本测试完成!")
    else:
        video_path = sys.argv[1]

        try:
            # 加载视频
            print("加载视频...")
            processor = VideoProcessor(video_path)
            frames, fps = processor.extract_frames(target_fps=30, max_frames=30)
            print(f"提取了 {len(frames)} 帧")

            # 姿态估计
            print("\n进行姿态估计...")
            estimator = create_pose_estimator('mediapipe')
            keypoints_seq = estimator.process_frames(frames)

            detected_count = sum(1 for kp in keypoints_seq if kp['detected'])
            print(f"检测成功: {detected_count}/{len(keypoints_seq)} 帧")

            # 可视化
            saved = False
            for frame, kps in zip(frames, keypoints_seq):
                if kps['detected']:
                    vis_frame = estimator.visualize_pose(frame, kps)
                    cv2.imwrite('pose_test_output.jpg', vis_frame)
                    print("\n可视化结果已保存: pose_test_output.jpg")
                    saved = True
                    break

            if not saved:
                print("\n未检测到姿态")

            estimator.close()
            processor.release()
            print("\n✅ 模块测试完成!")

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
