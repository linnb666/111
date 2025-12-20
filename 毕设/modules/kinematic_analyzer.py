# modules/kinematic_analyzer.py
"""
改进版运动学特征解析模块
参考OpenPose和DeepGait的算法思路，提高准确性
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple, Optional
from config.config import KINEMATIC_CONFIG


class KinematicAnalyzer:
    """改进版运动学分析器"""

    def __init__(self):
        """初始化分析器"""
        self.smooth_window = KINEMATIC_CONFIG['smooth_window']

    def analyze_sequence(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """
        分析完整关键点序列
        Args:
            keypoints_sequence: 关键点时间序列
            fps: 帧率
        Returns:
            分析结果字典
        """
        # 提取有效帧
        valid_frames = [kp for kp in keypoints_sequence if kp['detected']]

        if len(valid_frames) < 10:
            return self._get_empty_analysis()

        print(f"有效帧数: {len(valid_frames)}/{len(keypoints_sequence)}")

        # 计算各项运动学指标（改进版）
        results = {
            'fps': fps,
            'total_frames': len(keypoints_sequence),
            'valid_frames': len(valid_frames),
            'angles': self._calculate_angles_improved(valid_frames),
            'vertical_motion': self._calculate_vertical_motion_improved(valid_frames, fps),
            'cadence': self._calculate_cadence_improved(valid_frames, fps),
            'stride_info': self._calculate_stride_info_improved(valid_frames, fps),
            'stability': self._calculate_stability_improved(valid_frames),
            'body_lean': self._calculate_body_lean(valid_frames),
            'arm_swing': self._calculate_arm_swing(valid_frames)
        }

        return results

    def _calculate_angles_improved(self, keypoints_sequence: List[Dict]) -> Dict:
        """
        改进的关节角度计算
        使用平滑滤波和异常值处理
        """
        knee_angles_left = []
        knee_angles_right = []
        hip_angles_left = []
        hip_angles_right = []
        ankle_angles_left = []
        ankle_angles_right = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 左膝角度（髋-膝-踝）
            left_knee_angle = self._calculate_joint_angle_safe(
                landmarks[23],  # left_hip
                landmarks[25],  # left_knee
                landmarks[27]  # left_ankle
            )
            knee_angles_left.append(left_knee_angle)

            # 右膝角度
            right_knee_angle = self._calculate_joint_angle_safe(
                landmarks[24],  # right_hip
                landmarks[26],  # right_knee
                landmarks[28]  # right_ankle
            )
            knee_angles_right.append(right_knee_angle)

            # 左髋角度（肩-髋-膝）
            left_hip_angle = self._calculate_joint_angle_safe(
                landmarks[11],  # left_shoulder
                landmarks[23],  # left_hip
                landmarks[25]  # left_knee
            )
            hip_angles_left.append(left_hip_angle)

            # 右髋角度
            right_hip_angle = self._calculate_joint_angle_safe(
                landmarks[12],  # right_shoulder
                landmarks[24],  # right_hip
                landmarks[26]  # right_knee
            )
            hip_angles_right.append(right_hip_angle)

            # 左踝角度（膝-踝-脚尖）
            left_ankle_angle = self._calculate_joint_angle_safe(
                landmarks[25],  # left_knee
                landmarks[27],  # left_ankle
                landmarks[31]  # left_foot_index
            )
            ankle_angles_left.append(left_ankle_angle)

            # 右踝角度
            right_ankle_angle = self._calculate_joint_angle_safe(
                landmarks[26],  # right_knee
                landmarks[28],  # right_ankle
                landmarks[32]  # right_foot_index
            )
            ankle_angles_right.append(right_ankle_angle)

        # 异常值处理和平滑
        knee_angles_left = self._smooth_and_filter_angles(knee_angles_left)
        knee_angles_right = self._smooth_and_filter_angles(knee_angles_right)
        hip_angles_left = self._smooth_and_filter_angles(hip_angles_left)
        hip_angles_right = self._smooth_and_filter_angles(hip_angles_right)
        ankle_angles_left = self._smooth_and_filter_angles(ankle_angles_left)
        ankle_angles_right = self._smooth_and_filter_angles(ankle_angles_right)

        return {
            'knee_left': knee_angles_left,
            'knee_right': knee_angles_right,
            'hip_left': hip_angles_left,
            'hip_right': hip_angles_right,
            'ankle_left': ankle_angles_left,
            'ankle_right': ankle_angles_right,
            # 统计量
            'knee_left_mean': np.mean(knee_angles_left),
            'knee_right_mean': np.mean(knee_angles_right),
            'knee_left_std': np.std(knee_angles_left),
            'knee_right_std': np.std(knee_angles_right),
            'knee_left_max': np.max(knee_angles_left),
            'knee_left_min': np.min(knee_angles_left),
            'knee_rom': np.max(knee_angles_left) - np.min(knee_angles_left)  # 关节活动度
        }

    def _calculate_joint_angle_safe(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """
        安全的关节角度计算（带异常处理）
        p2为关节点
        """
        if p1['visibility'] < 0.5 or p2['visibility'] < 0.5 or p3['visibility'] < 0.5:
            return np.nan

        try:
            # 使用归一化坐标（更稳定）
            v1 = np.array([p1['x_norm'] - p2['x_norm'], p1['y_norm'] - p2['y_norm']])
            v2 = np.array([p3['x_norm'] - p2['x_norm'], p3['y_norm'] - p2['y_norm']])

            # 计算角度
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 1e-6 or norm2 < 1e-6:
                return np.nan

            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            return np.degrees(angle)
        except:
            return np.nan

    def _smooth_and_filter_angles(self, angles: List[float]) -> List[float]:
        """
        平滑和滤波角度序列
        1. 处理NaN值
        2. 异常值检测和替换
        3. 低通滤波
        4. Savitzky-Golay平滑
        """
        angles = np.array(angles)

        # 1. 插值处理NaN
        if np.any(np.isnan(angles)):
            valid_idx = ~np.isnan(angles)
            if np.sum(valid_idx) < 2:
                return [0.0] * len(angles)

            interp_func = interp1d(
                np.where(valid_idx)[0],
                angles[valid_idx],
                kind='linear',
                fill_value='extrapolate'
            )
            angles = interp_func(np.arange(len(angles)))

        # 2. 异常值检测（3-sigma原则）
        mean = np.mean(angles)
        std = np.std(angles)
        outliers = np.abs(angles - mean) > 3 * std
        if np.any(outliers):
            angles[outliers] = mean

        # 3. 低通滤波（去除高频噪声）
        if len(angles) > 6:
            b, a = butter(2, 0.2, btype='low')
            angles = filtfilt(b, a, angles)

        # 4. Savitzky-Golay平滑
        if len(angles) > self.smooth_window:
            window = min(self.smooth_window, len(angles))
            if window % 2 == 0:
                window -= 1
            angles = savgol_filter(angles, window_length=window, polyorder=2)

        return angles.tolist()

    def _calculate_vertical_motion_improved(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """
        改进的垂直运动分析
        使用髋部中点作为质心，分析垂直位移模式
        """
        # 提取髋部Y坐标（归一化）
        hip_y_positions = []
        valid_indices = []

        for i, kp in enumerate(keypoints_sequence):
            left_hip = kp['landmarks'][23]
            right_hip = kp['landmarks'][24]

            if left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
                # 使用归一化坐标
                center_y = (left_hip['y_norm'] + right_hip['y_norm']) / 2
                hip_y_positions.append(center_y)
                valid_indices.append(i)

        if len(hip_y_positions) < 10:
            return {'amplitude': 0.0, 'frequency': 0.0, 'positions': []}

        hip_y_positions = np.array(hip_y_positions)

        # 平滑处理
        hip_y_smooth = self._smooth_signal_advanced(hip_y_positions)

        # 分析垂直振荡
        peaks, _ = find_peaks(hip_y_smooth, distance=int(fps * 0.2))
        troughs, _ = find_peaks(-hip_y_smooth, distance=int(fps * 0.2))

        # 计算振幅（使用峰谷差的平均值）
        if len(peaks) > 0 and len(troughs) > 0:
            peak_values = hip_y_smooth[peaks]
            trough_values = hip_y_smooth[troughs]
            amplitude = np.mean(peak_values) - np.mean(trough_values)
        else:
            amplitude = np.max(hip_y_smooth) - np.min(hip_y_smooth)

        # 计算频率（Hz）
        if len(peaks) > 1:
            time_span = len(hip_y_positions) / fps
            frequency = len(peaks) / time_span
        else:
            frequency = 0.0

        return {
            'amplitude': float(amplitude),
            'frequency': float(frequency),
            'positions': hip_y_smooth.tolist(),
            'mean_position': float(np.mean(hip_y_smooth)),
            'std_position': float(np.std(hip_y_smooth)),
            'peak_count': len(peaks),
            'trough_count': len(troughs)
        }

    def _calculate_cadence_improved(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """
        改进的步频计算
        综合使用膝关节角度、脚踝Y坐标、髋部运动三种方法
        """
        # 方法1: 基于膝关节角度
        cadence1, step_count1 = self._cadence_from_knee_angle(keypoints_sequence, fps)

        # 方法2: 基于脚踝Y坐标（触地检测）
        cadence2, step_count2 = self._cadence_from_ankle_position(keypoints_sequence, fps)

        # 方法3: 基于髋部垂直运动
        cadence3, step_count3 = self._cadence_from_hip_motion(keypoints_sequence, fps)

        # 加权平均（脚踝方法权重最高）
        cadences = [cadence1, cadence2, cadence3]
        weights = [0, 1, 0]  # 脚踝触地检测最准确

        valid_cadences = [(c, w) for c, w in zip(cadences, weights) if c > 0]
        if valid_cadences:
            weighted_cadence = sum(c * w for c, w in valid_cadences) / sum(w for _, w in valid_cadences)
            avg_step_count = int(np.mean([step_count1, step_count2, step_count3]))
        else:
            weighted_cadence = 0
            avg_step_count = 0

        duration = len(keypoints_sequence) / fps

        return {
            'cadence': float(weighted_cadence),
            'step_count': avg_step_count,
            'duration': float(duration),
            'cadence_knee': float(cadence1),
            'cadence_ankle': float(cadence2),
            'cadence_hip': float(cadence3),
            'confidence': self._calculate_cadence_confidence([cadence1, cadence2, cadence3])
        }

    def _cadence_from_knee_angle(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int]:
        """从膝关节角度计算步频"""
        knee_angles = []

        for kp in keypoints_sequence:
            # 使用右膝（可以换成左膝或取平均）
            angle = self._calculate_joint_angle_safe(
                kp['landmarks'][24],
                kp['landmarks'][26],
                kp['landmarks'][28]
            )
            knee_angles.append(angle if not np.isnan(angle) else 0)

        if len(knee_angles) < 10:
            return 0.0, 0

        # 平滑
        knee_angles = self._smooth_signal_advanced(np.array(knee_angles))

        # 检测峰值（膝盖弯曲最大时）
        peaks, properties = find_peaks(
            -knee_angles,  # 负号因为角度越小表示弯曲越大
            distance=int(fps * 0.3),  # 最小步态周期0.3秒
            prominence=5  # 最小显著性
        )

        step_count = len(peaks)
        duration = len(keypoints_sequence) / fps
        cadence = (step_count / duration) * 60 if duration > 0 else 0

        return cadence, step_count

    def _cadence_from_ankle_position(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int]:
        """从脚踝位置计算步频（触地检测）"""
        # 收集左右脚踝Y坐标
        left_ankle_y = []
        right_ankle_y = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]

            if left['visibility'] > 0.5:
                left_ankle_y.append(left['y_norm'])
            else:
                left_ankle_y.append(np.nan)

            if right['visibility'] > 0.5:
                right_ankle_y.append(right['y_norm'])
            else:
                right_ankle_y.append(np.nan)

        # 插值处理
        left_ankle_y = self._interpolate_nans(np.array(left_ankle_y))
        right_ankle_y = self._interpolate_nans(np.array(right_ankle_y))

        if len(left_ankle_y) < 10:
            return 0.0, 0

        # 平滑
        left_ankle_y = self._smooth_signal_advanced(left_ankle_y)
        right_ankle_y = self._smooth_signal_advanced(right_ankle_y)

        # 检测触地（Y坐标最大时，因为Y轴向下）
        left_peaks, _ = find_peaks(left_ankle_y, distance=int(fps * 0.3), prominence=0.01)
        right_peaks, _ = find_peaks(right_ankle_y, distance=int(fps * 0.3), prominence=0.01)

        step_count = len(left_peaks) + len(right_peaks)
        duration = len(keypoints_sequence) / fps
        cadence = (step_count / duration) * 60 if duration > 0 else 0

        return cadence, step_count

    def _cadence_from_hip_motion(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int]:
        """从髋部运动计算步频"""
        hip_y = []

        for kp in keypoints_sequence:
            left_hip = kp['landmarks'][23]
            right_hip = kp['landmarks'][24]

            if left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
                center_y = (left_hip['y_norm'] + right_hip['y_norm']) / 2
                hip_y.append(center_y)

        if len(hip_y) < 10:
            return 0.0, 0

        hip_y = self._smooth_signal_advanced(np.array(hip_y))

        # 髋部有两个峰值周期对应一步
        peaks, _ = find_peaks(hip_y, distance=int(fps * 0.15))

        step_count = len(peaks)
        duration = len(hip_y) / fps
        cadence = (step_count / duration) * 60 if duration > 0 else 0

        return cadence, step_count

    def _calculate_cadence_confidence(self, cadences: List[float]) -> float:
        """计算步频置信度（基于三种方法的一致性）"""
        valid_cadences = [c for c in cadences if c > 0]
        if len(valid_cadences) < 2:
            return 0.5

        # 计算变异系数
        mean = np.mean(valid_cadences)
        std = np.std(valid_cadences)
        cv = std / mean if mean > 0 else 1.0

        # 变异系数越小，置信度越高
        confidence = max(0, 1.0 - cv)
        return float(confidence)

    def _calculate_stride_info_improved(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """改进的步态信息计算"""
        # 计算步长（使用脚踝X坐标的最大跨度）
        left_ankle_x = []
        right_ankle_x = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]

            if left['visibility'] > 0.5:
                left_ankle_x.append(left['x_norm'])
            if right['visibility'] > 0.5:
                right_ankle_x.append(right['x_norm'])

        stride_length = 0.0
        if left_ankle_x and right_ankle_x:
            # 步长 = 左右脚最大横向距离
            left_range = np.max(left_ankle_x) - np.min(left_ankle_x)
            right_range = np.max(right_ankle_x) - np.min(right_ankle_x)
            stride_length = (left_range + right_range) / 2

        # 计算触地时间比例
        ground_contact_ratio = self._estimate_ground_contact_ratio(keypoints_sequence, fps)

        return {
            'stride_length_norm': float(stride_length),  # 归一化步长
            'ground_contact_ratio': float(ground_contact_ratio),
            'flight_time_ratio': float(1.0 - ground_contact_ratio)
        }

    def _estimate_ground_contact_ratio(self, keypoints_sequence: List[Dict], fps: float) -> float:
        """估算触地时间比例"""
        ankle_velocities = []

        for i in range(1, len(keypoints_sequence)):
            curr = keypoints_sequence[i]['landmarks'][27]  # 左脚踝
            prev = keypoints_sequence[i - 1]['landmarks'][27]

            if curr['visibility'] > 0.5 and prev['visibility'] > 0.5:
                # 计算垂直速度
                vy = (curr['y_norm'] - prev['y_norm']) * fps
                ankle_velocities.append(abs(vy))

        if not ankle_velocities:
            return 0.6  # 默认值

        # 速度小的帧认为是触地
        threshold = np.percentile(ankle_velocities, 50)
        ground_frames = sum(1 for v in ankle_velocities if v < threshold)

        return ground_frames / len(ankle_velocities)

    def _calculate_stability_improved(self, keypoints_sequence: List[Dict]) -> Dict:
        """改进的稳定性计算"""
        # 1. 躯干稳定性（肩部和髋部的横向和纵向稳定性）
        trunk_stability = self._calculate_trunk_stability(keypoints_sequence)

        # 2. 头部稳定性
        head_stability = self._calculate_head_stability(keypoints_sequence)

        # 3. 步态对称性
        gait_symmetry = self._calculate_gait_symmetry(keypoints_sequence)

        # 综合稳定性
        overall = (trunk_stability * 0.5 + head_stability * 0.2 + gait_symmetry * 0.3)

        return {
            'overall': float(overall),
            'trunk': float(trunk_stability),
            'head': float(head_stability),
            'symmetry': float(gait_symmetry)
        }

    def _calculate_trunk_stability(self, keypoints_sequence: List[Dict]) -> float:
        """计算躯干稳定性"""
        shoulder_x = []
        hip_x = []

        for kp in keypoints_sequence:
            # 肩部中点
            ls, rs = kp['landmarks'][11], kp['landmarks'][12]
            if ls['visibility'] > 0.5 and rs['visibility'] > 0.5:
                shoulder_x.append((ls['x_norm'] + rs['x_norm']) / 2)

            # 髋部中点
            lh, rh = kp['landmarks'][23], kp['landmarks'][24]
            if lh['visibility'] > 0.5 and rh['visibility'] > 0.5:
                hip_x.append((lh['x_norm'] + rh['x_norm']) / 2)

        if not shoulder_x or not hip_x:
            return 50.0

        # 稳定性 = 100 - 横向摆动幅度
        shoulder_std = np.std(shoulder_x) * 100
        hip_std = np.std(hip_x) * 100

        stability = 100 - min((shoulder_std + hip_std) / 2, 100)
        return max(0, stability)

    def _calculate_head_stability(self, keypoints_sequence: List[Dict]) -> float:
        """计算头部稳定性"""
        nose_y = []

        for kp in keypoints_sequence:
            nose = kp['landmarks'][0]
            if nose['visibility'] > 0.5:
                nose_y.append(nose['y_norm'])

        if len(nose_y) < 10:
            return 50.0

        # 头部垂直稳定性
        nose_y = self._smooth_signal_advanced(np.array(nose_y))
        stability = 100 - min(np.std(nose_y) * 100, 100)

        return max(0, stability)

    def _calculate_gait_symmetry(self, keypoints_sequence: List[Dict]) -> float:
        """计算步态对称性"""
        left_knee_angles = []
        right_knee_angles = []

        for kp in keypoints_sequence:
            left_angle = self._calculate_joint_angle_safe(
                kp['landmarks'][23], kp['landmarks'][25], kp['landmarks'][27]
            )
            right_angle = self._calculate_joint_angle_safe(
                kp['landmarks'][24], kp['landmarks'][26], kp['landmarks'][28]
            )

            if not np.isnan(left_angle):
                left_knee_angles.append(left_angle)
            if not np.isnan(right_angle):
                right_knee_angles.append(right_angle)

        if len(left_knee_angles) < 10 or len(right_knee_angles) < 10:
            return 50.0

        # 左右膝盖角度的相关性
        min_len = min(len(left_knee_angles), len(right_knee_angles))
        correlation = np.corrcoef(
            left_knee_angles[:min_len],
            right_knee_angles[:min_len]
        )[0, 1]

        symmetry = abs(correlation) * 100
        return max(0, min(symmetry, 100))

    def _calculate_body_lean(self, keypoints_sequence: List[Dict]) -> Dict:
        """计算身体前倾角度"""
        lean_angles = []

        for kp in keypoints_sequence:
            # 使用髋部和肩部中点计算前倾
            shoulder = kp['landmarks'][11]  # 左肩（也可以用中点）
            hip = kp['landmarks'][23]  # 左髋

            if shoulder['visibility'] > 0.5 and hip['visibility'] > 0.5:
                # 计算与垂直线的夹角
                dx = shoulder['x_norm'] - hip['x_norm']
                dy = shoulder['y_norm'] - hip['y_norm']
                lean_angle = np.degrees(np.arctan2(dx, -dy))  # 负号因为Y轴向下
                lean_angles.append(lean_angle)

        if not lean_angles:
            return {'mean_lean': 0.0, 'std_lean': 0.0}

        lean_angles = self._smooth_signal_advanced(np.array(lean_angles))

        return {
            'mean_lean': float(np.mean(lean_angles)),
            'std_lean': float(np.std(lean_angles)),
            'forward_lean': float(np.mean([a for a in lean_angles if a > 0]))
        }

    def _calculate_arm_swing(self, keypoints_sequence: List[Dict]) -> Dict:
        """计算手臂摆动幅度"""
        left_elbow_angles = []
        right_elbow_angles = []

        for kp in keypoints_sequence:
            # 左手肘角度（肩-肘-腕）
            left_angle = self._calculate_joint_angle_safe(
                kp['landmarks'][11],  # left_shoulder
                kp['landmarks'][13],  # left_elbow
                kp['landmarks'][15]  # left_wrist
            )
            if not np.isnan(left_angle):
                left_elbow_angles.append(left_angle)

            # 右手肘角度
            right_angle = self._calculate_joint_angle_safe(
                kp['landmarks'][12],  # right_shoulder
                kp['landmarks'][14],  # right_elbow
                kp['landmarks'][16]  # right_wrist
            )
            if not np.isnan(right_angle):
                right_elbow_angles.append(right_angle)

        if not left_elbow_angles or not right_elbow_angles:
            return {'arm_swing_amplitude': 0.0}

        left_range = np.max(left_elbow_angles) - np.min(left_elbow_angles)
        right_range = np.max(right_elbow_angles) - np.min(right_elbow_angles)

        return {
            'arm_swing_amplitude': float((left_range + right_range) / 2),
            'left_arm_range': float(left_range),
            'right_arm_range': float(right_range)
        }

    def _smooth_signal_advanced(self, signal: np.ndarray) -> np.ndarray:
        """
        高级信号平滑
        结合多种方法
        """
        if len(signal) < 5:
            return signal

        # 1. 中值滤波去除尖峰
        from scipy.ndimage import median_filter
        signal = median_filter(signal, size=3)

        # 2. 低通滤波
        if len(signal) > 6:
            b, a = butter(2, 0.2, btype='low')
            signal = filtfilt(b, a, signal)

        # 3. Savitzky-Golay平滑
        if len(signal) > self.smooth_window:
            window = min(self.smooth_window, len(signal))
            if window % 2 == 0:
                window -= 1
            signal = savgol_filter(signal, window_length=window, polyorder=2)

        return signal

    def _interpolate_nans(self, arr: np.ndarray) -> np.ndarray:
        """插值处理NaN值"""
        if not np.any(np.isnan(arr)):
            return arr

        valid_idx = ~np.isnan(arr)
        if np.sum(valid_idx) < 2:
            return np.zeros_like(arr)

        interp_func = interp1d(
            np.where(valid_idx)[0],
            arr[valid_idx],
            kind='linear',
            fill_value='extrapolate'
        )
        return interp_func(np.arange(len(arr)))

    def _get_empty_analysis(self) -> Dict:
        """返回空分析结果"""
        return {
            'fps': 0,
            'total_frames': 0,
            'valid_frames': 0,
            'angles': {},
            'vertical_motion': {},
            'cadence': {},
            'stride_info': {},
            'stability': {},
            'body_lean': {},
            'arm_swing': {}
        }


# 模块测试代码
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

    print("=" * 60)
    print("测试改进版运动学分析模块")
    print("=" * 60)

    # 生成模拟数据
    print("\n生成模拟关键点数据...")
    mock_keypoints = []
    fps = 30
    duration = 3  # 3秒
    num_frames = fps * duration

    for i in range(num_frames):
        kp = {
            'detected': True,
            'landmarks': []
        }

        # 模拟跑步周期（2步/秒）
        t = i / fps
        phase = t * 2 * np.pi * 2  # 2 Hz

        for j in range(33):
            # 膝盖和脚踝有明显周期性
            if j in [25, 26, 27, 28]:
                y_offset = np.sin(phase + (j % 2) * np.pi) * 0.1
            else:
                y_offset = 0

            kp['landmarks'].append({
                'id': j,
                'name': f'point_{j}',
                'x': 320,
                'y': 240 + y_offset * 100,
                'x_norm': 0.5,
                'y_norm': 0.5 + y_offset,
                'visibility': 0.9
            })

        mock_keypoints.append(kp)

    print(f"生成了 {len(mock_keypoints)} 帧数据")

    # 执行分析
    print("\n执行运动学分析...")
    analyzer = KinematicAnalyzer()
    results = analyzer.analyze_sequence(mock_keypoints, fps)

    # 打印结果
    print("\n" + "=" * 60)
    print("分析结果:")
    print("=" * 60)
    print(f"步频: {results['cadence']['cadence']:.1f} 步/分")
    print(f"  - 膝盖方法: {results['cadence']['cadence_knee']:.1f}")
    print(f"  - 脚踝方法: {results['cadence']['cadence_ankle']:.1f}")
    print(f"  - 髋部方法: {results['cadence']['cadence_hip']:.1f}")
    print(f"  - 置信度: {results['cadence']['confidence']:.2f}")
    print(f"\n步数: {results['cadence']['step_count']}")
    print(f"垂直振幅: {results['vertical_motion']['amplitude']:.4f}")
    print(f"稳定性: {results['stability']['overall']:.2f}")
    print(f"  - 躯干: {results['stability']['trunk']:.2f}")
    print(f"  - 头部: {results['stability']['head']:.2f}")
    print(f"  - 对称性: {results['stability']['symmetry']:.2f}")

    print("\n✅ 模块测试完成!")