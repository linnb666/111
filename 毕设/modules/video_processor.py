import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from config.config import VIDEO_CONFIG


class VideoProcessor:
    """视频处理器类"""

    def __init__(self, video_path: str):
        """
        初始化视频处理器
        Args:
            video_path: 视频文件路径
        """
        self.video_path = Path(video_path)
        self.cap = None
        self.video_info = {}

        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        if self.video_path.suffix.lower() not in VIDEO_CONFIG['supported_formats']:
            raise ValueError(f"不支持的视频格式: {self.video_path.suffix}")

        self._load_video()

    def _load_video(self):
        """加载视频并解析基本信息"""
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        # 获取视频基本信息
        self.video_info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': 0.0,
            'filename': self.video_path.name
        }

        self.video_info['duration'] = self.video_info['frame_count'] / self.video_info['fps']

    def get_video_info(self) -> Dict:
        """获取视频基本信息"""
        return self.video_info.copy()

    def extract_frames(self,
                       target_fps: Optional[int] = None,
                       resize: bool = False,
                       max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
        """
        提取视频帧
        Args:
            target_fps: 目标帧率（None则使用原始帧率）
            resize: 是否调整分辨率
            max_frames: 最大帧数限制
        Returns:
            frames: 帧列表
            actual_fps: 实际帧率
        """
        return self.extract_frames_from_position(
            start_frame=0,
            target_fps=target_fps,
            resize=resize,
            max_frames=max_frames
        )

    def extract_frames_from_position(self,
                                      start_frame: int = 0,
                                      target_fps: Optional[int] = None,
                                      resize: bool = False,
                                      max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
        """
        从指定位置提取视频帧（用于提取中间段）

        Args:
            start_frame: 开始帧索引
            target_fps: 目标帧率（None则使用原始帧率）
            resize: 是否调整分辨率
            max_frames: 最大帧数限制
        Returns:
            frames: 帧列表
            actual_fps: 实际帧率
        """
        frames = []
        original_fps = self.video_info['fps']

        # 计算帧采样间隔
        if target_fps is None or target_fps >= original_fps:
            frame_interval = 1
            actual_fps = original_fps
        else:
            frame_interval = int(original_fps / target_fps)
            actual_fps = original_fps / frame_interval

        # 设置起始位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 按间隔采样
            if frame_idx % frame_interval == 0:
                # 调整分辨率
                if resize:
                    frame = self._resize_frame(frame)

                frames.append(frame)

                # 检查最大帧数限制
                if max_frames and len(frames) >= max_frames:
                    break

            frame_idx += 1

        return frames, actual_fps

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """调整帧分辨率"""
        target_width = VIDEO_CONFIG['target_width']
        target_height = VIDEO_CONFIG['target_height']
        return cv2.resize(frame, (target_width, target_height))

    def extract_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        提取指定时间的帧
        Args:
            timestamp: 时间戳(秒)
        Returns:
            frame: 图像帧
        """
        frame_number = int(timestamp * self.video_info['fps'])
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """释放视频资源"""
        if self.cap:
            self.cap.release()

    def __del__(self):
        """析构函数"""
        self.release()

    @staticmethod
    def validate_video(video_path: str) -> bool:
        """
        验证视频文件是否有效
        Args:
            video_path: 视频路径
        Returns:
            是否有效
        """
        try:
            cap = cv2.VideoCapture(video_path)
            is_valid = cap.isOpened()
            cap.release()
            return is_valid
        except:
            return False


# 模块测试代码
if __name__ == "__main__":
    # 测试示例
    import sys

    if len(sys.argv) < 2:
        print("用法: python video_processor.py ")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        processor = VideoProcessor(video_path)
        info = processor.get_video_info()

        print("=" * 50)
        print("视频信息:")
        print(f"文件名: {info['filename']}")
        print(f"分辨率: {info['width']}x{info['height']}")
        print(f"帧率: {info['fps']:.2f} FPS")
        print(f"总帧数: {info['frame_count']}")
        print(f"时长: {info['duration']:.2f} 秒")
        print("=" * 50)

        # 提取帧
        print("\n正在提取帧...")
        frames, fps = processor.extract_frames(target_fps=30, resize=True, max_frames=10)
        print(f"提取了 {len(frames)} 帧, 实际FPS: {fps:.2f}")
        print(f"帧大小: {frames[0].shape}")

        processor.release()
        print("\n模块测试完成!")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)