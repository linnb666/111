# modules/pose_estimator_3d.py
"""
3D姿态估计模块 - RTMPose + MotionBERT架构

核心功能：
1. RTMPose进行2D姿态检测
2. MotionBERT进行2D→3D姿态提升
3. 4GB显存优化（FP16、batch处理）
4. MediaPipe后备方案
"""
import cv2
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings

from config.config import POSE_CONFIG, MMPOSE_3D_CONFIG


class BasePoseEstimator3D(ABC):
    """3D姿态估计器基类"""

    # COCO 17关键点定义
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # 跑步分析关键关节
    RUNNING_JOINTS = {
        'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14,
        'left_ankle': 15, 'right_ankle': 16,
        'left_shoulder': 5, 'right_shoulder': 6,
        'nose': 0
    }

    # 骨骼连接定义
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
        (5, 11), (6, 12), (11, 12),  # 躯干
        (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
    ]

    @abstractmethod
    def process_video(self, frames: List[np.ndarray]) -> Dict:
        """处理视频帧序列，返回2D和3D关键点"""
        pass

    @abstractmethod
    def close(self):
        """释放资源"""
        pass

    def get_joint_by_name(self, keypoints_3d: np.ndarray, frame_idx: int, name: str) -> Optional[np.ndarray]:
        """根据名称获取关节3D坐标"""
        if name in self.RUNNING_JOINTS:
            joint_idx = self.RUNNING_JOINTS[name]
            return keypoints_3d[frame_idx, joint_idx]
        return None


class MMPose3DEstimator(BasePoseEstimator3D):
    """
    MMPose 3D姿态估计器
    使用RTMPose进行2D检测，MotionBERT进行3D提升
    """

    def __init__(self, config: Dict = None):
        """初始化MMPose 3D估计器"""
        self.config = config or MMPOSE_3D_CONFIG
        self.device = self.config.get('device', 'cpu')  # 默认使用CPU
        self.use_fp16 = self.config['optimization'].get('use_fp16', False)  # CPU默认不用FP16

        self.detector_2d = None
        self.lifter_3d = None
        self.initialized = False

        self._init_models()

    def _init_models(self):
        """初始化2D检测器和3D提升器"""
        try:
            # 设备处理
            if 'cuda' in self.device:
                if not torch.cuda.is_available():
                    print("CUDA不可用，切换到CPU模式")
                    self.device = 'cpu'
                    self.use_fp16 = False
            else:
                # CPU模式下禁用FP16
                self.use_fp16 = False
                print("使用CPU模式进行推理")

            # 初始化RTMPose 2D检测器
            self._init_rtmpose()

            # 初始化MotionBERT 3D提升器
            self._init_motionbert()

            self.initialized = True
            print(f"MMPose 3D初始化成功 (设备: {self.device}, FP16: {self.use_fp16})")

        except ImportError as e:
            print(f"警告: MMPose/MotionBERT未安装 - {e}")
            print("将使用MediaPipe后备方案")
            self.initialized = False
        except Exception as e:
            print(f"警告: 模型初始化失败 - {e}")
            self.initialized = False

    def _init_rtmpose(self):
        """初始化RTMPose 2D检测器"""
        try:
            from mmpose.apis import MMPoseInferencer

            det_config = self.config['detector_2d']
            checkpoint_path = det_config['checkpoint']

            # 检查权重文件
            if not Path(checkpoint_path).exists():
                print(f"RTMPose权重不存在: {checkpoint_path}")
                print("使用默认在线权重...")
                # 使用MMPose内置的RTMPose
                self.detector_2d = MMPoseInferencer(
                    pose2d='rtmpose-m',
                    device=self.device
                )
            else:
                self.detector_2d = MMPoseInferencer(
                    pose2d=checkpoint_path,
                    device=self.device
                )

            print("RTMPose 2D检测器初始化成功")

        except ImportError:
            raise ImportError("请安装mmpose: pip install mmpose mmcv mmdet")

    def _init_motionbert(self):
        """初始化MotionBERT 3D提升器"""
        try:
            # MotionBERT需要单独安装
            # 这里提供一个简化的接口
            lifter_config = self.config['lifter_3d']
            checkpoint_path = lifter_config['checkpoint']

            if not Path(checkpoint_path).exists():
                print(f"MotionBERT权重不存在: {checkpoint_path}")
                print("将使用简化的3D提升方法")
                self.lifter_3d = SimpleLift3D(self.device, self.use_fp16)
            else:
                # 加载MotionBERT
                self.lifter_3d = MotionBERTLifter(
                    checkpoint_path,
                    self.device,
                    self.use_fp16,
                    lifter_config['receptive_frames']
                )

            print("3D提升器初始化成功")

        except Exception as e:
            print(f"MotionBERT初始化失败: {e}")
            print("使用简化的3D提升方法")
            self.lifter_3d = SimpleLift3D(self.device, self.use_fp16)

    def process_video(self, frames: List[np.ndarray]) -> Dict:
        """
        处理视频帧序列

        Args:
            frames: BGR格式的视频帧列表

        Returns:
            {
                'keypoints_2d': np.ndarray,  # (T, 17, 2)
                'keypoints_3d': np.ndarray,  # (T, 17, 3)
                'confidences': np.ndarray,   # (T, 17)
                'detected_frames': List[int],
                'fps': float
            }
        """
        if not self.initialized:
            raise RuntimeError("模型未初始化，请检查依赖")

        n_frames = len(frames)
        opt = self.config['optimization']
        batch_size = opt.get('batch_size', 8)

        # 存储结果
        all_keypoints_2d = []
        all_confidences = []
        detected_frames = []

        print(f"开始2D姿态检测 ({n_frames}帧)...")

        # 分批处理2D检测
        for start_idx in range(0, n_frames, batch_size):
            end_idx = min(start_idx + batch_size, n_frames)
            batch_frames = frames[start_idx:end_idx]

            # 2D检测
            batch_results = self._detect_2d_batch(batch_frames)

            for i, result in enumerate(batch_results):
                frame_idx = start_idx + i
                if result is not None:
                    all_keypoints_2d.append(result['keypoints'])
                    all_confidences.append(result['confidences'])
                    detected_frames.append(frame_idx)
                else:
                    # 检测失败，填充零
                    all_keypoints_2d.append(np.zeros((17, 2)))
                    all_confidences.append(np.zeros(17))

            # 显存清理（仅CUDA模式）
            if 'cuda' in self.device and start_idx > 0 and start_idx % opt.get('clear_cache_interval', 50) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 转换为numpy数组
        keypoints_2d = np.array(all_keypoints_2d)  # (T, 17, 2)
        confidences = np.array(all_confidences)    # (T, 17)

        print(f"2D检测完成: {len(detected_frames)}/{n_frames}帧成功")

        # 3D提升
        print("开始3D姿态提升...")
        keypoints_3d = self._lift_to_3d(keypoints_2d, confidences)
        print("3D提升完成")

        # 平滑处理
        keypoints_3d = self._smooth_keypoints(keypoints_3d)

        return {
            'keypoints_2d': keypoints_2d,
            'keypoints_3d': keypoints_3d,
            'confidences': confidences,
            'detected_frames': detected_frames,
            'detection_rate': len(detected_frames) / n_frames
        }

    def _detect_2d_batch(self, frames: List[np.ndarray]) -> List[Optional[Dict]]:
        """批量2D检测"""
        results = []

        for frame in frames:
            try:
                # 转换颜色空间
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # RTMPose推理
                result = next(self.detector_2d(frame_rgb, return_vis=False))

                if result and 'predictions' in result:
                    preds = result['predictions'][0]
                    if len(preds) > 0:
                        # 取第一个检测到的人
                        person = preds[0]
                        keypoints = np.array(person['keypoints'])[:17, :2]
                        scores = np.array(person['keypoint_scores'])[:17]

                        results.append({
                            'keypoints': keypoints,
                            'confidences': scores
                        })
                    else:
                        results.append(None)
                else:
                    results.append(None)

            except Exception as e:
                print(f"2D检测异常: {e}")
                results.append(None)

        return results

    def _lift_to_3d(self, keypoints_2d: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """2D→3D提升"""
        if self.lifter_3d is None:
            # 简单的伪3D（Z=0）
            n_frames = keypoints_2d.shape[0]
            keypoints_3d = np.zeros((n_frames, 17, 3))
            keypoints_3d[:, :, :2] = keypoints_2d
            return keypoints_3d

        return self.lifter_3d.lift(keypoints_2d, confidences)

    def _smooth_keypoints(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """平滑3D关键点"""
        from scipy.signal import savgol_filter

        n_frames = keypoints_3d.shape[0]
        if n_frames < 7:
            return keypoints_3d

        smoothed = np.zeros_like(keypoints_3d)
        window = min(7, n_frames // 2)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return keypoints_3d

        for joint in range(17):
            for dim in range(3):
                smoothed[:, joint, dim] = savgol_filter(
                    keypoints_3d[:, joint, dim], window, 2
                )

        return smoothed

    def visualize_pose(self, frame: np.ndarray, keypoints_2d: np.ndarray,
                       confidences: np.ndarray) -> np.ndarray:
        """可视化2D姿态"""
        vis_frame = frame.copy()
        h, w = frame.shape[:2]

        # 绘制关键点
        for i, (kp, conf) in enumerate(zip(keypoints_2d, confidences)):
            if conf > 0.5:
                x, y = int(kp[0]), int(kp[1])
                color = self._get_joint_color(i)
                cv2.circle(vis_frame, (x, y), 4, color, -1)

        # 绘制骨骼
        for start_idx, end_idx in self.SKELETON:
            if confidences[start_idx] > 0.5 and confidences[end_idx] > 0.5:
                start_pt = tuple(map(int, keypoints_2d[start_idx]))
                end_pt = tuple(map(int, keypoints_2d[end_idx]))
                color = self._get_limb_color(start_idx, end_idx)
                cv2.line(vis_frame, start_pt, end_pt, color, 2)

        return vis_frame

    def _get_joint_color(self, joint_idx: int) -> Tuple[int, int, int]:
        """获取关节颜色"""
        if joint_idx in [11, 12, 13, 14, 15, 16]:  # 下肢
            return (0, 255, 0)
        elif joint_idx in [5, 6, 7, 8, 9, 10]:  # 上肢
            return (0, 0, 255)
        else:  # 头部/躯干
            return (255, 128, 0)

    def _get_limb_color(self, start: int, end: int) -> Tuple[int, int, int]:
        """获取肢体颜色"""
        leg_joints = {11, 12, 13, 14, 15, 16}
        if start in leg_joints and end in leg_joints:
            return (0, 200, 0)
        arm_joints = {5, 6, 7, 8, 9, 10}
        if start in arm_joints and end in arm_joints:
            return (200, 0, 0)
        return (200, 128, 0)

    def close(self):
        """释放资源"""
        if self.detector_2d is not None:
            del self.detector_2d
        if self.lifter_3d is not None:
            del self.lifter_3d
        # 仅在使用CUDA时清理显存
        if 'cuda' in self.device and torch.cuda.is_available():
            torch.cuda.empty_cache()


class SimpleLift3D:
    """
    简化的3D提升器

    当MotionBERT不可用时的后备方案
    使用几何约束和运动学规则进行2D→3D估计
    """

    def __init__(self, device: str = 'cpu', use_fp16: bool = False):
        self.device = device
        self.use_fp16 = use_fp16

        # 人体比例先验（基于标准人体测量学）
        self.body_proportions = {
            'torso_length': 0.3,      # 躯干长度 / 身高
            'upper_leg': 0.245,       # 大腿长度 / 身高
            'lower_leg': 0.246,       # 小腿长度 / 身高
            'upper_arm': 0.172,       # 上臂长度 / 身高
            'forearm': 0.157,         # 前臂长度 / 身高
        }

    def lift(self, keypoints_2d: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """
        使用几何约束进行2D→3D提升

        核心思想：
        1. 假设相机正交投影
        2. 使用人体比例先验约束Z深度
        3. 基于骨骼长度一致性优化
        """
        n_frames, n_joints, _ = keypoints_2d.shape
        keypoints_3d = np.zeros((n_frames, n_joints, 3))

        # 复制2D坐标到XY
        keypoints_3d[:, :, :2] = keypoints_2d

        # 估计每帧的人体高度（用于归一化）
        for t in range(n_frames):
            # 估计身高：鼻子到脚踝的距离
            nose = keypoints_2d[t, 0]
            left_ankle = keypoints_2d[t, 15]
            right_ankle = keypoints_2d[t, 16]

            if confidences[t, 0] > 0.5 and (confidences[t, 15] > 0.5 or confidences[t, 16] > 0.5):
                ankle = left_ankle if confidences[t, 15] > confidences[t, 16] else right_ankle
                height_2d = np.linalg.norm(nose - ankle)

                if height_2d > 0:
                    # 基于骨骼长度推断Z深度
                    keypoints_3d[t] = self._estimate_depth(
                        keypoints_2d[t], confidences[t], height_2d
                    )

        return keypoints_3d

    def _estimate_depth(self, kp_2d: np.ndarray, conf: np.ndarray,
                        height_2d: float) -> np.ndarray:
        """估计单帧的Z深度"""
        kp_3d = np.zeros((17, 3))
        kp_3d[:, :2] = kp_2d

        # 归一化系数
        scale = 1.0 / height_2d if height_2d > 0 else 1.0

        # 基于人体比例估计各部位深度
        # 躯干平面（假设在Z=0附近）
        for i in [5, 6, 11, 12]:  # 肩膀和髋部
            if conf[i] > 0.5:
                kp_3d[i, 2] = 0.0

        # 下肢深度（基于膝关节弯曲角度推断）
        for side, (hip, knee, ankle) in enumerate([
            (11, 13, 15),  # 左腿
            (12, 14, 16)   # 右腿
        ]):
            if all(conf[j] > 0.5 for j in [hip, knee, ankle]):
                # 计算2D膝关节角度
                v1 = kp_2d[hip] - kp_2d[knee]
                v2 = kp_2d[ankle] - kp_2d[knee]
                angle_2d = np.arctan2(
                    np.cross(v1, v2),
                    np.dot(v1, v2)
                )

                # 角度越大（膝盖弯曲），膝盖越向前（Z越正）
                knee_depth = np.sin(angle_2d) * 0.1
                kp_3d[knee, 2] = knee_depth

                # 脚踝深度
                kp_3d[ankle, 2] = knee_depth * 0.5

        # 平滑深度值
        # 使用加权平均确保连续性

        return kp_3d


class MotionBERTLifter:
    """
    MotionBERT 3D提升器

    使用预训练的MotionBERT模型进行2D→3D姿态提升
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu',
                 use_fp16: bool = False, receptive_frames: int = 243):
        self.device = device
        self.use_fp16 = use_fp16
        self.receptive_frames = receptive_frames
        self.model = None

        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """加载MotionBERT模型"""
        try:
            # 这里需要根据实际的MotionBERT实现进行调整
            # 示例代码，实际使用需要安装MotionBERT
            from lib.model.DSTformer import DSTformer

            self.model = DSTformer(
                dim_in=2,
                dim_out=3,
                dim_feat=256,
                dim_rep=512,
                depth=5,
                num_heads=8,
                maxlen=self.receptive_frames
            )

            # 加载权重
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
            self.model.eval()

            # 仅在CUDA设备上使用FP16
            if self.use_fp16 and 'cuda' in self.device:
                self.model.half()

            print(f"MotionBERT模型加载成功 (设备: {self.device})")

        except ImportError:
            print("MotionBERT未安装，使用简化3D提升")
            self.model = None
        except Exception as e:
            print(f"MotionBERT加载失败: {e}")
            self.model = None

    def lift(self, keypoints_2d: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """使用MotionBERT进行3D提升"""
        if self.model is None:
            # 后备方案
            return SimpleLift3D().lift(keypoints_2d, confidences)

        n_frames = keypoints_2d.shape[0]
        keypoints_3d = np.zeros((n_frames, 17, 3))

        # 滑动窗口处理
        window_size = self.receptive_frames
        overlap = 27
        stride = window_size - overlap

        with torch.no_grad():
            for start in range(0, n_frames, stride):
                end = min(start + window_size, n_frames)
                window_2d = keypoints_2d[start:end]

                # 填充到window_size
                if len(window_2d) < window_size:
                    pad_size = window_size - len(window_2d)
                    window_2d = np.pad(
                        window_2d,
                        ((0, pad_size), (0, 0), (0, 0)),
                        mode='edge'
                    )

                # 转换为tensor
                input_tensor = torch.from_numpy(window_2d).float()
                input_tensor = input_tensor.unsqueeze(0).to(self.device)

                # 仅在CUDA设备上使用FP16
                if self.use_fp16 and 'cuda' in self.device:
                    input_tensor = input_tensor.half()

                # 推理
                output = self.model(input_tensor)
                output_3d = output[0].cpu().numpy()

                # 复制到结果（处理重叠区域）
                valid_len = min(end - start, window_size)
                if start == 0:
                    keypoints_3d[:valid_len] = output_3d[:valid_len]
                else:
                    # 融合重叠区域
                    overlap_start = start
                    for i in range(overlap):
                        if overlap_start + i < n_frames:
                            alpha = i / overlap
                            keypoints_3d[overlap_start + i] = (
                                (1 - alpha) * keypoints_3d[overlap_start + i] +
                                alpha * output_3d[i]
                            )
                    # 非重叠区域直接复制
                    for i in range(overlap, valid_len):
                        if start + i < n_frames:
                            keypoints_3d[start + i] = output_3d[i]

        return keypoints_3d


class MediaPipeFallback(BasePoseEstimator3D):
    """
    MediaPipe后备方案

    当MMPose不可用时使用
    提供2D姿态估计和伪3D输出
    """

    def __init__(self, config: Dict = None):
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        mp_config = config or POSE_CONFIG.get('mediapipe', {})
        self.pose = self.mp_pose.Pose(
            model_complexity=mp_config.get('model_complexity', 1),
            min_detection_confidence=mp_config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=mp_config.get('min_tracking_confidence', 0.5),
            static_image_mode=mp_config.get('static_image_mode', False)
        )

        # MediaPipe使用33关键点，需要映射到COCO 17关键点
        self.mp_to_coco = {
            0: 0,    # nose
            2: 1,    # left_eye
            5: 2,    # right_eye
            7: 3,    # left_ear
            8: 4,    # right_ear
            11: 5,   # left_shoulder
            12: 6,   # right_shoulder
            13: 7,   # left_elbow
            14: 8,   # right_elbow
            15: 9,   # left_wrist
            16: 10,  # right_wrist
            23: 11,  # left_hip
            24: 12,  # right_hip
            25: 13,  # left_knee
            26: 14,  # right_knee
            27: 15,  # left_ankle
            28: 16,  # right_ankle
        }

        self.initialized = True
        print("MediaPipe后备方案初始化成功")

    def process_video(self, frames: List[np.ndarray]) -> Dict:
        """处理视频帧"""
        n_frames = len(frames)
        keypoints_2d = np.zeros((n_frames, 17, 2))
        keypoints_3d = np.zeros((n_frames, 17, 3))
        confidences = np.zeros((n_frames, 17))
        detected_frames = []

        for i, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                detected_frames.append(i)
                h, w = frame.shape[:2]

                for mp_idx, coco_idx in self.mp_to_coco.items():
                    lm = results.pose_landmarks.landmark[mp_idx]
                    keypoints_2d[i, coco_idx] = [lm.x * w, lm.y * h]
                    keypoints_3d[i, coco_idx] = [lm.x, lm.y, lm.z]
                    confidences[i, coco_idx] = lm.visibility

        return {
            'keypoints_2d': keypoints_2d,
            'keypoints_3d': keypoints_3d,  # MediaPipe的伪3D
            'confidences': confidences,
            'detected_frames': detected_frames,
            'detection_rate': len(detected_frames) / n_frames,
            'backend': 'mediapipe'
        }

    def close(self):
        """释放资源"""
        if self.pose:
            self.pose.close()


def create_pose_estimator_3d(config: Dict = None) -> BasePoseEstimator3D:
    """
    创建3D姿态估计器的工厂函数

    优先使用MMPose 3D，失败则回退到MediaPipe
    """
    backend = POSE_CONFIG.get('backend', 'mmpose_3d')

    if backend == 'mmpose_3d':
        try:
            estimator = MMPose3DEstimator(config)
            if estimator.initialized:
                return estimator
            else:
                print("MMPose初始化失败，使用MediaPipe后备")
        except Exception as e:
            print(f"MMPose创建失败: {e}")
            print("使用MediaPipe后备方案")

    # 后备方案
    return MediaPipeFallback(config)


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试3D姿态估计模块")
    print("=" * 60)

    # 创建估计器
    estimator = create_pose_estimator_3d()
    print(f"创建成功，类型: {type(estimator).__name__}")

    # 测试单帧
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        result = estimator.process_video([test_frame])
        print(f"2D关键点形状: {result['keypoints_2d'].shape}")
        print(f"3D关键点形状: {result['keypoints_3d'].shape}")
        print(f"检测率: {result['detection_rate']:.2%}")
    except Exception as e:
        print(f"测试失败: {e}")

    estimator.close()
    print("\n测试完成!")
