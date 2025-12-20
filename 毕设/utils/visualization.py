import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def create_comparison_video(original_frames: List[np.ndarray],
                            pose_frames: List[np.ndarray],
                            output_path: str,
                            fps: float = 30):
    """创建对比视频"""
    if not original_frames or not pose_frames:
        return

    h, w = original_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    for orig, pose in zip(original_frames, pose_frames):
        combined = np.hstack([orig, pose])
        out.write(combined)

    out.release()


def plot_angle_curves(angles: Dict, output_path: str):
    """绘制角度变化曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(angles['knee_left'])
    axes[0, 0].set_title('Left Knee Angle')
    axes[0, 0].set_ylabel('Degrees')

    axes[0, 1].plot(angles['knee_right'])
    axes[0, 1].set_title('Right Knee Angle')

    axes[1, 0].plot(angles['hip_left'])
    axes[1, 0].set_title('Left Hip Angle')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Degrees')

    axes[1, 1].plot(angles['hip_right'])
    axes[1, 1].set_title('Right Hip Angle')
    axes[1, 1].set_xlabel('Frame')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()