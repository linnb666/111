# modules/kinematic_analyzer.py
"""
重构版运动学特征解析模块
核心改进：
1. 垂直振幅使用躯干长度归一化（解决单位不匹配问题）
2. 膝关节角度分阶段分析（触地期、摆动期、蹬离期）
3. 支持不同视角的分析策略
4. 更精确的步态周期检测
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple, Optional
from config.config import KINEMATIC_CONFIG


class KinematicAnalyzer:
    """重构版运动学分析器"""

    # 跑步阶段定义
    PHASE_GROUND_CONTACT = 0  # 触地期
    PHASE_FLIGHT = 1          # 腾空期
    PHASE_TRANSITION = 2      # 过渡期

    def __init__(self):
        """初始化分析器"""
        self.smooth_window = KINEMATIC_CONFIG['smooth_window']

    def analyze_sequence(self, keypoints_sequence: List[Dict], fps: float,
                         view_angle: str = 'side') -> Dict:
        """
        分析完整关键点序列
        Args:
            keypoints_sequence: 关键点时间序列
            fps: 帧率
            view_angle: 视频视角 ('side', 'front', 'back')
        Returns:
            分析结果字典
        """
        # 提取有效帧
        valid_frames = [kp for kp in keypoints_sequence if kp['detected']]

        if len(valid_frames) < 10:
            return self._get_empty_analysis()

        print(f"有效帧数: {len(valid_frames)}/{len(keypoints_sequence)}")

        # 计算躯干参考长度（用于归一化）
        trunk_length = self._calculate_trunk_reference(valid_frames)
        print(f"躯干参考长度: {trunk_length:.4f} (归一化坐标)")

        # 基础运动学指标（所有视角通用）
        results = {
            'fps': fps,
            'total_frames': len(keypoints_sequence),
            'valid_frames': len(valid_frames),
            'view_angle': view_angle,
            'trunk_reference': trunk_length,
        }

        # 根据视角选择分析策略
        if view_angle == 'side':
            results.update(self._analyze_side_view(valid_frames, fps, trunk_length))
        elif view_angle in ['front', 'back']:
            results.update(self._analyze_frontal_view(valid_frames, fps, trunk_length))
        else:
            # 混合分析
            results.update(self._analyze_side_view(valid_frames, fps, trunk_length))

        return results

    def _calculate_trunk_reference(self, keypoints_sequence: List[Dict]) -> float:
        """
        计算躯干参考长度（肩到髋的平均距离）
        用于将垂直振幅归一化为相对身体尺度的比例
        """
        trunk_lengths = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 左侧躯干长度
            left_shoulder = landmarks[11]
            left_hip = landmarks[23]
            if left_shoulder['visibility'] > 0.5 and left_hip['visibility'] > 0.5:
                left_trunk = np.sqrt(
                    (left_shoulder['y_norm'] - left_hip['y_norm'])**2 +
                    (left_shoulder['x_norm'] - left_hip['x_norm'])**2
                )
                trunk_lengths.append(left_trunk)

            # 右侧躯干长度
            right_shoulder = landmarks[12]
            right_hip = landmarks[24]
            if right_shoulder['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
                right_trunk = np.sqrt(
                    (right_shoulder['y_norm'] - right_hip['y_norm'])**2 +
                    (right_shoulder['x_norm'] - right_hip['x_norm'])**2
                )
                trunk_lengths.append(right_trunk)

        if trunk_lengths:
            return np.median(trunk_lengths)  # 使用中位数更稳健
        return 0.25  # 默认值（约占画面1/4）

    # ==================== 侧面视角分析 ====================

    def _analyze_side_view(self, valid_frames: List[Dict], fps: float,
                           trunk_length: float) -> Dict:
        """侧面视角分析 - 主要分析策略"""
        results = {
            # 核心指标
            'angles': self._calculate_angles_by_phase(valid_frames, fps),
            'vertical_motion': self._calculate_vertical_motion_normalized(
                valid_frames, fps, trunk_length
            ),
            'cadence': self._calculate_cadence_improved(valid_frames, fps),
            'stride_info': self._calculate_stride_info(valid_frames, fps),

            # 稳定性指标
            'stability': self._calculate_stability_improved(valid_frames),
            'body_lean': self._calculate_body_lean(valid_frames),
            'arm_swing': self._calculate_arm_swing(valid_frames),

            # 步态周期分析
            'gait_cycle': self._analyze_gait_cycle(valid_frames, fps),
        }
        return results

    def _calculate_angles_by_phase(self, keypoints_sequence: List[Dict],
                                    fps: float) -> Dict:
        """
        分阶段计算关节角度
        核心改进：区分触地期、摆动期、蹬离期的角度
        """
        # 收集原始角度数据
        knee_angles_left = []
        knee_angles_right = []
        hip_angles_left = []
        hip_angles_right = []
        ankle_angles_left = []
        ankle_angles_right = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 膝关节角度（髋-膝-踝）
            knee_angles_left.append(self._calculate_joint_angle_safe(
                landmarks[23], landmarks[25], landmarks[27]
            ))
            knee_angles_right.append(self._calculate_joint_angle_safe(
                landmarks[24], landmarks[26], landmarks[28]
            ))

            # 髋关节角度（肩-髋-膝）
            hip_angles_left.append(self._calculate_joint_angle_safe(
                landmarks[11], landmarks[23], landmarks[25]
            ))
            hip_angles_right.append(self._calculate_joint_angle_safe(
                landmarks[12], landmarks[24], landmarks[26]
            ))

            # 踝关节角度（膝-踝-脚尖）
            ankle_angles_left.append(self._calculate_joint_angle_safe(
                landmarks[25], landmarks[27], landmarks[31]
            ))
            ankle_angles_right.append(self._calculate_joint_angle_safe(
                landmarks[26], landmarks[28], landmarks[32]
            ))

        # 平滑处理
        knee_left_smooth = self._smooth_and_filter_angles(knee_angles_left)
        knee_right_smooth = self._smooth_and_filter_angles(knee_angles_right)
        hip_left_smooth = self._smooth_and_filter_angles(hip_angles_left)
        hip_right_smooth = self._smooth_and_filter_angles(hip_angles_right)
        ankle_left_smooth = self._smooth_and_filter_angles(ankle_angles_left)
        ankle_right_smooth = self._smooth_and_filter_angles(ankle_angles_right)

        # 检测步态阶段
        phases = self._detect_gait_phases(keypoints_sequence, fps)

        # 分阶段统计膝关节角度
        phase_angles = self._analyze_angles_by_phase(
            knee_left_smooth, knee_right_smooth, phases
        )

        return {
            # 原始时间序列
            'knee_left': knee_left_smooth,
            'knee_right': knee_right_smooth,
            'hip_left': hip_left_smooth,
            'hip_right': hip_right_smooth,
            'ankle_left': ankle_left_smooth,
            'ankle_right': ankle_right_smooth,

            # 基础统计量
            'knee_left_mean': float(np.nanmean(knee_left_smooth)),
            'knee_right_mean': float(np.nanmean(knee_right_smooth)),
            'knee_left_std': float(np.nanstd(knee_left_smooth)),
            'knee_right_std': float(np.nanstd(knee_right_smooth)),
            'knee_rom': float(np.nanmax(knee_left_smooth) - np.nanmin(knee_left_smooth)),

            # ⭐ 分阶段角度分析（核心改进）
            'phase_analysis': phase_angles,
        }

    def _detect_gait_phases(self, keypoints_sequence: List[Dict],
                            fps: float) -> List[int]:
        """
        检测每一帧的步态阶段
        基于脚踝Y坐标和膝关节角度判断
        """
        phases = []

        # 提取脚踝Y坐标
        left_ankle_y = []
        right_ankle_y = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]

            left_y = left['y_norm'] if left['visibility'] > 0.5 else np.nan
            right_y = right['y_norm'] if right['visibility'] > 0.5 else np.nan

            left_ankle_y.append(left_y)
            right_ankle_y.append(right_y)

        # 插值和平滑
        left_ankle_y = self._interpolate_nans(np.array(left_ankle_y))
        right_ankle_y = self._interpolate_nans(np.array(right_ankle_y))

        if len(left_ankle_y) > 5:
            left_ankle_y = self._smooth_signal_advanced(left_ankle_y)
            right_ankle_y = self._smooth_signal_advanced(right_ankle_y)

        # 检测触地点（Y坐标最大值，因为Y轴向下）
        left_peaks, _ = find_peaks(left_ankle_y, distance=int(fps * 0.25))
        right_peaks, _ = find_peaks(right_ankle_y, distance=int(fps * 0.25))

        # 合并所有触地点
        all_ground_contacts = sorted(set(list(left_peaks) + list(right_peaks)))

        # 为每一帧标注阶段
        for i in range(len(keypoints_sequence)):
            # 找到最近的触地点
            distances = [abs(i - gc) for gc in all_ground_contacts]
            if distances:
                min_dist = min(distances)
                frame_window = fps * 0.1  # 100ms窗口

                if min_dist < frame_window:
                    phases.append(self.PHASE_GROUND_CONTACT)
                elif min_dist < frame_window * 2:
                    phases.append(self.PHASE_TRANSITION)
                else:
                    phases.append(self.PHASE_FLIGHT)
            else:
                phases.append(self.PHASE_TRANSITION)

        return phases

    def _analyze_angles_by_phase(self, knee_left: List[float],
                                  knee_right: List[float],
                                  phases: List[int]) -> Dict:
        """
        按阶段统计膝关节角度
        """
        # 分组收集各阶段的角度
        ground_contact_angles = []
        flight_angles = []
        transition_angles = []

        for i, phase in enumerate(phases):
            if i < len(knee_left) and i < len(knee_right):
                avg_knee = (knee_left[i] + knee_right[i]) / 2
                if not np.isnan(avg_knee):
                    if phase == self.PHASE_GROUND_CONTACT:
                        ground_contact_angles.append(avg_knee)
                    elif phase == self.PHASE_FLIGHT:
                        flight_angles.append(avg_knee)
                    else:
                        transition_angles.append(avg_knee)

        # 计算各阶段统计量
        def safe_stats(angles):
            if len(angles) > 0:
                return {
                    'mean': float(np.mean(angles)),
                    'std': float(np.std(angles)),
                    'min': float(np.min(angles)),
                    'max': float(np.max(angles)),
                    'count': len(angles)
                }
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}

        # 计算关键时刻角度
        knee_all = [(knee_left[i] + knee_right[i]) / 2 for i in range(min(len(knee_left), len(knee_right)))]
        knee_all = [x for x in knee_all if not np.isnan(x)]

        return {
            # 触地期角度（理想范围：155-170°）
            'ground_contact': safe_stats(ground_contact_angles),
            # 腾空期/摆动期角度（理想范围：90-130°，弯曲较大）
            'flight': safe_stats(flight_angles),
            # 过渡期角度
            'transition': safe_stats(transition_angles),
            # 关键指标
            'max_flexion': float(np.min(knee_all)) if knee_all else 0,  # 最大弯曲（最小角度）
            'max_extension': float(np.max(knee_all)) if knee_all else 0,  # 最大伸展
            'range_of_motion': float(np.max(knee_all) - np.min(knee_all)) if knee_all else 0,
        }

    def _calculate_vertical_motion_normalized(self, keypoints_sequence: List[Dict],
                                               fps: float, trunk_length: float) -> Dict:
        """
        归一化垂直运动分析
        核心改进：使用躯干长度作为参考尺度，输出相对振幅
        """
        # 提取髋部Y坐标
        hip_y_positions = []
        valid_indices = []

        for i, kp in enumerate(keypoints_sequence):
            left_hip = kp['landmarks'][23]
            right_hip = kp['landmarks'][24]

            if left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
                center_y = (left_hip['y_norm'] + right_hip['y_norm']) / 2
                hip_y_positions.append(center_y)
                valid_indices.append(i)

        if len(hip_y_positions) < 10:
            return self._get_empty_vertical_motion()

        hip_y_positions = np.array(hip_y_positions)
        hip_y_smooth = self._smooth_signal_advanced(hip_y_positions)

        # 分析垂直振荡
        peaks, _ = find_peaks(hip_y_smooth, distance=int(fps * 0.2))
        troughs, _ = find_peaks(-hip_y_smooth, distance=int(fps * 0.2))

        # 计算原始振幅（归一化坐标）
        if len(peaks) > 0 and len(troughs) > 0:
            peak_values = hip_y_smooth[peaks]
            trough_values = hip_y_smooth[troughs]
            raw_amplitude = np.mean(peak_values) - np.mean(trough_values)
        else:
            raw_amplitude = np.max(hip_y_smooth) - np.min(hip_y_smooth)

        # ⭐ 核心改进：归一化为躯干长度的百分比
        # 这样振幅变成一个与身高无关的相对值
        normalized_amplitude = (raw_amplitude / trunk_length) * 100 if trunk_length > 0 else 0

        # 计算频率（Hz）
        if len(peaks) > 1:
            time_span = len(hip_y_positions) / fps
            frequency = len(peaks) / time_span
        else:
            frequency = 0.0

        return {
            'amplitude': float(raw_amplitude),  # 原始归一化振幅
            'amplitude_normalized': float(normalized_amplitude),  # ⭐ 相对躯干长度的百分比
            'frequency': float(frequency),
            'positions': hip_y_smooth.tolist(),
            'mean_position': float(np.mean(hip_y_smooth)),
            'std_position': float(np.std(hip_y_smooth)),
            'peak_count': len(peaks),
            'trough_count': len(troughs),
            # 评估等级
            'amplitude_rating': self._rate_vertical_amplitude(normalized_amplitude),
        }

    def _rate_vertical_amplitude(self, normalized_amplitude: float) -> Dict:
        """
        评估垂直振幅等级
        基于躯干百分比的专业标准：
        - 优秀：3-6%（垂直振幅约为躯干长度的3-6%）
        - 良好：6-10%
        - 一般：10-15%
        - 待改进：>15% 或 <3%
        """
        if 3 <= normalized_amplitude <= 6:
            return {'level': 'excellent', 'score': 100, 'description': '垂直振幅非常理想'}
        elif 6 < normalized_amplitude <= 10:
            return {'level': 'good', 'score': 80, 'description': '垂直振幅良好'}
        elif 10 < normalized_amplitude <= 15:
            return {'level': 'fair', 'score': 60, 'description': '垂直振幅偏大，能量损耗较多'}
        elif normalized_amplitude < 3:
            return {'level': 'low', 'score': 70, 'description': '垂直振幅偏小，步态可能过于保守'}
        else:
            return {'level': 'poor', 'score': 40, 'description': '垂直振幅过大，建议改善跑姿'}

    def _get_empty_vertical_motion(self) -> Dict:
        """返回空的垂直运动结果"""
        return {
            'amplitude': 0.0,
            'amplitude_normalized': 0.0,
            'frequency': 0.0,
            'positions': [],
            'mean_position': 0.0,
            'std_position': 0.0,
            'peak_count': 0,
            'trough_count': 0,
            'amplitude_rating': {'level': 'unknown', 'score': 0, 'description': '数据不足'}
        }

    def _analyze_gait_cycle(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """
        分析完整步态周期
        """
        phases = self._detect_gait_phases(keypoints_sequence, fps)

        # 统计各阶段占比
        total = len(phases)
        ground_contact_ratio = phases.count(self.PHASE_GROUND_CONTACT) / total if total > 0 else 0
        flight_ratio = phases.count(self.PHASE_FLIGHT) / total if total > 0 else 0
        transition_ratio = phases.count(self.PHASE_TRANSITION) / total if total > 0 else 0

        # 计算步态周期时间
        # 寻找连续的触地-腾空-触地周期
        cycle_durations = []
        i = 0
        while i < len(phases) - 1:
            if phases[i] == self.PHASE_GROUND_CONTACT:
                # 找到下一个触地期的开始
                start = i
                j = i + 1
                while j < len(phases) and phases[j] != self.PHASE_GROUND_CONTACT:
                    j += 1
                if j < len(phases):
                    # 继续找当前触地期结束
                    k = j + 1
                    while k < len(phases) and phases[k] == self.PHASE_GROUND_CONTACT:
                        k += 1
                    if k > j:
                        cycle_duration = (j - start) / fps
                        if 0.3 < cycle_duration < 1.5:  # 合理的步态周期范围
                            cycle_durations.append(cycle_duration)
                i = j
            else:
                i += 1

        avg_cycle_duration = np.mean(cycle_durations) if cycle_durations else 0

        return {
            'phase_distribution': {
                'ground_contact': float(ground_contact_ratio),
                'flight': float(flight_ratio),
                'transition': float(transition_ratio),
            },
            'avg_cycle_duration': float(avg_cycle_duration),
            'cycle_count': len(cycle_durations),
            'phases': phases,
            # 评估
            'gait_rating': self._rate_gait_distribution(ground_contact_ratio, flight_ratio),
        }

    def _rate_gait_distribution(self, ground_ratio: float, flight_ratio: float) -> Dict:
        """
        评估步态分布
        理想的跑步步态：触地期约40-50%，腾空期约30-40%
        """
        if 0.35 <= ground_ratio <= 0.55 and 0.25 <= flight_ratio <= 0.45:
            return {'level': 'excellent', 'score': 100, 'description': '步态节奏优秀'}
        elif 0.30 <= ground_ratio <= 0.60 and 0.20 <= flight_ratio <= 0.50:
            return {'level': 'good', 'score': 80, 'description': '步态节奏良好'}
        elif ground_ratio > 0.60:
            return {'level': 'heavy', 'score': 60, 'description': '触地时间偏长，可能步态较重'}
        elif flight_ratio < 0.15:
            return {'level': 'shuffling', 'score': 50, 'description': '腾空不足，可能为拖步跑'}
        else:
            return {'level': 'fair', 'score': 70, 'description': '步态节奏一般'}

    # ==================== 正面/后方视角分析 ====================

    def _analyze_frontal_view(self, valid_frames: List[Dict], fps: float,
                               trunk_length: float) -> Dict:
        """正面/后方视角分析 - 侧重力线和对称性"""
        results = {
            'lower_limb_alignment': self._calculate_lower_limb_alignment(valid_frames),
            'gait_symmetry': self._calculate_gait_symmetry_detailed(valid_frames),
            'lateral_stability': self._calculate_lateral_stability(valid_frames),
            'vertical_motion': self._calculate_vertical_motion_normalized(
                valid_frames, fps, trunk_length
            ),
            'cadence': self._calculate_cadence_improved(valid_frames, fps),
            'stability': self._calculate_stability_improved(valid_frames),
        }
        return results

    def _calculate_lower_limb_alignment(self, keypoints_sequence: List[Dict]) -> Dict:
        """
        计算下肢力线（膝内扣/外翻趋势）
        通过分析髋-膝-踝的横向对齐情况
        """
        left_valgus_angles = []  # 左腿外翻角
        right_valgus_angles = []  # 右腿外翻角

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 左腿力线
            left_hip = landmarks[23]
            left_knee = landmarks[25]
            left_ankle = landmarks[27]

            if all(p['visibility'] > 0.5 for p in [left_hip, left_knee, left_ankle]):
                # 计算膝关节相对于髋-踝连线的横向偏移
                # 正值表示膝外翻（X向外），负值表示膝内扣
                hip_ankle_x = (left_hip['x_norm'] + left_ankle['x_norm']) / 2
                knee_offset = left_knee['x_norm'] - hip_ankle_x
                # 转换为角度（简化计算）
                hip_ankle_dist = abs(left_hip['y_norm'] - left_ankle['y_norm'])
                if hip_ankle_dist > 0.01:
                    valgus_angle = np.degrees(np.arctan(knee_offset / hip_ankle_dist))
                    left_valgus_angles.append(valgus_angle)

            # 右腿力线
            right_hip = landmarks[24]
            right_knee = landmarks[26]
            right_ankle = landmarks[28]

            if all(p['visibility'] > 0.5 for p in [right_hip, right_knee, right_ankle]):
                hip_ankle_x = (right_hip['x_norm'] + right_ankle['x_norm']) / 2
                knee_offset = right_knee['x_norm'] - hip_ankle_x
                hip_ankle_dist = abs(right_hip['y_norm'] - right_ankle['y_norm'])
                if hip_ankle_dist > 0.01:
                    valgus_angle = np.degrees(np.arctan(knee_offset / hip_ankle_dist))
                    right_valgus_angles.append(valgus_angle)

        # 统计分析
        def analyze_alignment(angles, side):
            if not angles:
                return {'mean': 0, 'max': 0, 'issue': 'unknown', 'severity': 'unknown'}

            mean_angle = np.mean(angles)
            max_angle = np.max(np.abs(angles))

            # 判断问题类型
            if mean_angle > 5:
                issue = 'valgus'  # 膝外翻
                severity = 'mild' if mean_angle < 10 else 'moderate' if mean_angle < 15 else 'severe'
            elif mean_angle < -5:
                issue = 'varus'  # 膝内扣
                severity = 'mild' if mean_angle > -10 else 'moderate' if mean_angle > -15 else 'severe'
            else:
                issue = 'normal'
                severity = 'none'

            return {
                'mean': float(mean_angle),
                'max': float(max_angle),
                'issue': issue,
                'severity': severity
            }

        return {
            'left_leg': analyze_alignment(left_valgus_angles, 'left'),
            'right_leg': analyze_alignment(right_valgus_angles, 'right'),
            'overall_rating': self._rate_lower_limb_alignment(
                left_valgus_angles, right_valgus_angles
            )
        }

    def _rate_lower_limb_alignment(self, left_angles: List, right_angles: List) -> Dict:
        """评估下肢力线"""
        if not left_angles or not right_angles:
            return {'level': 'unknown', 'score': 0, 'description': '数据不足'}

        max_deviation = max(
            np.max(np.abs(left_angles)) if left_angles else 0,
            np.max(np.abs(right_angles)) if right_angles else 0
        )

        if max_deviation < 5:
            return {'level': 'excellent', 'score': 100, 'description': '下肢力线良好'}
        elif max_deviation < 10:
            return {'level': 'good', 'score': 80, 'description': '下肢力线基本正常'}
        elif max_deviation < 15:
            return {'level': 'fair', 'score': 60, 'description': '存在轻度膝关节偏移'}
        else:
            return {'level': 'poor', 'score': 40, 'description': '膝关节偏移明显，建议关注'}

    def _calculate_gait_symmetry_detailed(self, keypoints_sequence: List[Dict]) -> Dict:
        """
        详细的步态对称性分析
        """
        # 收集左右侧关键点运动数据
        left_ankle_y = []
        right_ankle_y = []
        left_knee_angles = []
        right_knee_angles = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 脚踝Y坐标
            if landmarks[27]['visibility'] > 0.5:
                left_ankle_y.append(landmarks[27]['y_norm'])
            if landmarks[28]['visibility'] > 0.5:
                right_ankle_y.append(landmarks[28]['y_norm'])

            # 膝关节角度
            left_angle = self._calculate_joint_angle_safe(
                landmarks[23], landmarks[25], landmarks[27]
            )
            right_angle = self._calculate_joint_angle_safe(
                landmarks[24], landmarks[26], landmarks[28]
            )

            if not np.isnan(left_angle):
                left_knee_angles.append(left_angle)
            if not np.isnan(right_angle):
                right_knee_angles.append(right_angle)

        # 计算对称性指标
        symmetry_scores = []

        # 脚踝运动对称性
        if len(left_ankle_y) > 10 and len(right_ankle_y) > 10:
            min_len = min(len(left_ankle_y), len(right_ankle_y))
            ankle_corr = np.corrcoef(left_ankle_y[:min_len], right_ankle_y[:min_len])[0, 1]
            if not np.isnan(ankle_corr):
                symmetry_scores.append(abs(ankle_corr) * 100)

        # 膝关节角度对称性
        if len(left_knee_angles) > 10 and len(right_knee_angles) > 10:
            min_len = min(len(left_knee_angles), len(right_knee_angles))
            knee_corr = np.corrcoef(left_knee_angles[:min_len], right_knee_angles[:min_len])[0, 1]
            if not np.isnan(knee_corr):
                symmetry_scores.append(abs(knee_corr) * 100)

            # 左右差异
            left_mean = np.mean(left_knee_angles)
            right_mean = np.mean(right_knee_angles)
            angle_diff = abs(left_mean - right_mean)
        else:
            angle_diff = 0

        overall_symmetry = np.mean(symmetry_scores) if symmetry_scores else 50

        return {
            'overall_score': float(overall_symmetry),
            'knee_angle_difference': float(angle_diff),
            'rating': self._rate_symmetry(overall_symmetry, angle_diff)
        }

    def _rate_symmetry(self, symmetry_score: float, angle_diff: float) -> Dict:
        """评估对称性"""
        if symmetry_score >= 90 and angle_diff < 5:
            return {'level': 'excellent', 'score': 100, 'description': '步态高度对称'}
        elif symmetry_score >= 80 and angle_diff < 10:
            return {'level': 'good', 'score': 80, 'description': '步态对称性良好'}
        elif symmetry_score >= 70:
            return {'level': 'fair', 'score': 60, 'description': '存在轻度不对称'}
        else:
            return {'level': 'poor', 'score': 40, 'description': '步态不对称明显'}

    def _calculate_lateral_stability(self, keypoints_sequence: List[Dict]) -> Dict:
        """计算横向稳定性（正面视角）"""
        hip_x_positions = []
        shoulder_x_positions = []

        for kp in keypoints_sequence:
            landmarks = kp['landmarks']

            # 髋部中点横向位置
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            if left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
                hip_x = (left_hip['x_norm'] + right_hip['x_norm']) / 2
                hip_x_positions.append(hip_x)

            # 肩部中点横向位置
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            if left_shoulder['visibility'] > 0.5 and right_shoulder['visibility'] > 0.5:
                shoulder_x = (left_shoulder['x_norm'] + right_shoulder['x_norm']) / 2
                shoulder_x_positions.append(shoulder_x)

        # 计算横向摆动幅度
        hip_sway = np.std(hip_x_positions) * 100 if hip_x_positions else 0
        shoulder_sway = np.std(shoulder_x_positions) * 100 if shoulder_x_positions else 0

        # 综合评分
        total_sway = (hip_sway + shoulder_sway) / 2
        stability_score = max(0, 100 - total_sway * 10)

        return {
            'hip_sway': float(hip_sway),
            'shoulder_sway': float(shoulder_sway),
            'stability_score': float(stability_score),
            'rating': self._rate_lateral_stability(stability_score)
        }

    def _rate_lateral_stability(self, score: float) -> Dict:
        """评估横向稳定性"""
        if score >= 90:
            return {'level': 'excellent', 'score': 100, 'description': '横向稳定性优秀'}
        elif score >= 75:
            return {'level': 'good', 'score': 80, 'description': '横向稳定性良好'}
        elif score >= 60:
            return {'level': 'fair', 'score': 60, 'description': '存在横向摆动'}
        else:
            return {'level': 'poor', 'score': 40, 'description': '横向摆动明显，影响效率'}

    # ==================== 通用计算方法 ====================

    def _calculate_joint_angle_safe(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """安全的关节角度计算（p2为关节点）"""
        if p1['visibility'] < 0.5 or p2['visibility'] < 0.5 or p3['visibility'] < 0.5:
            return np.nan

        try:
            v1 = np.array([p1['x_norm'] - p2['x_norm'], p1['y_norm'] - p2['y_norm']])
            v2 = np.array([p3['x_norm'] - p2['x_norm'], p3['y_norm'] - p2['y_norm']])

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
        """平滑和滤波角度序列"""
        angles = np.array(angles)

        # 插值处理NaN
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

        # 异常值检测（3-sigma原则）
        mean = np.mean(angles)
        std = np.std(angles)
        if std > 0:
            outliers = np.abs(angles - mean) > 3 * std
            if np.any(outliers):
                angles[outliers] = mean

        # 低通滤波
        if len(angles) > 6:
            b, a = butter(2, 0.2, btype='low')
            angles = filtfilt(b, a, angles)

        # Savitzky-Golay平滑
        if len(angles) > self.smooth_window:
            window = min(self.smooth_window, len(angles))
            if window % 2 == 0:
                window -= 1
            angles = savgol_filter(angles, window_length=window, polyorder=2)

        return angles.tolist()

    def _calculate_cadence_improved(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """改进的步频计算"""
        # 方法1: 基于膝关节角度
        cadence1, step_count1 = self._cadence_from_knee_angle(keypoints_sequence, fps)

        # 方法2: 基于脚踝Y坐标
        cadence2, step_count2 = self._cadence_from_ankle_position(keypoints_sequence, fps)

        # 方法3: 基于髋部运动
        cadence3, step_count3 = self._cadence_from_hip_motion(keypoints_sequence, fps)

        # 加权平均（脚踝方法权重最高）
        cadences = [cadence1, cadence2, cadence3]
        weights = [0.2, 0.6, 0.2]

        valid_cadences = [(c, w) for c, w in zip(cadences, weights) if c > 0]
        if valid_cadences:
            weighted_cadence = sum(c * w for c, w in valid_cadences) / sum(w for _, w in valid_cadences)
            avg_step_count = int(np.mean([s for s in [step_count1, step_count2, step_count3] if s > 0]))
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
            'confidence': self._calculate_cadence_confidence([cadence1, cadence2, cadence3]),
            'rating': self._rate_cadence(weighted_cadence),
        }

    def _rate_cadence(self, cadence: float) -> Dict:
        """评估步频（保留用户原有阈值逻辑）"""
        if 180 <= cadence <= 200:
            return {'level': 'optimal', 'score': 100, 'description': '步频处于最佳范围'}
        elif 170 <= cadence < 180 or 200 < cadence <= 210:
            return {'level': 'good', 'score': 85, 'description': '步频良好'}
        elif 160 <= cadence < 170 or 210 < cadence <= 220:
            return {'level': 'fair', 'score': 70, 'description': '步频可以优化'}
        elif cadence < 160:
            return {'level': 'low', 'score': 55, 'description': '步频偏低，建议提高'}
        else:
            return {'level': 'high', 'score': 60, 'description': '步频偏高，注意控制'}

    def _cadence_from_knee_angle(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int]:
        """从膝关节角度计算步频"""
        knee_angles = []
        for kp in keypoints_sequence:
            angle = self._calculate_joint_angle_safe(
                kp['landmarks'][24], kp['landmarks'][26], kp['landmarks'][28]
            )
            knee_angles.append(angle if not np.isnan(angle) else 0)

        if len(knee_angles) < 10:
            return 0.0, 0

        knee_angles = self._smooth_signal_advanced(np.array(knee_angles))
        peaks, _ = find_peaks(-knee_angles, distance=int(fps * 0.3), prominence=5)

        step_count = len(peaks)
        duration = len(keypoints_sequence) / fps
        cadence = (step_count / duration) * 60 if duration > 0 else 0

        return cadence, step_count

    def _cadence_from_ankle_position(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[float, int]:
        """从脚踝位置计算步频"""
        left_ankle_y = []
        right_ankle_y = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]
            left_ankle_y.append(left['y_norm'] if left['visibility'] > 0.5 else np.nan)
            right_ankle_y.append(right['y_norm'] if right['visibility'] > 0.5 else np.nan)

        left_ankle_y = self._interpolate_nans(np.array(left_ankle_y))
        right_ankle_y = self._interpolate_nans(np.array(right_ankle_y))

        if len(left_ankle_y) < 10:
            return 0.0, 0

        left_ankle_y = self._smooth_signal_advanced(left_ankle_y)
        right_ankle_y = self._smooth_signal_advanced(right_ankle_y)

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
                hip_y.append((left_hip['y_norm'] + right_hip['y_norm']) / 2)

        if len(hip_y) < 10:
            return 0.0, 0

        hip_y = self._smooth_signal_advanced(np.array(hip_y))
        peaks, _ = find_peaks(hip_y, distance=int(fps * 0.15))

        step_count = len(peaks)
        duration = len(hip_y) / fps
        cadence = (step_count / duration) * 60 if duration > 0 else 0

        return cadence, step_count

    def _calculate_cadence_confidence(self, cadences: List[float]) -> float:
        """计算步频置信度"""
        valid_cadences = [c for c in cadences if c > 0]
        if len(valid_cadences) < 2:
            return 0.5

        mean = np.mean(valid_cadences)
        std = np.std(valid_cadences)
        cv = std / mean if mean > 0 else 1.0

        confidence = max(0, 1.0 - cv)
        return float(confidence)

    def _calculate_stride_info(self, keypoints_sequence: List[Dict], fps: float) -> Dict:
        """步态信息计算"""
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
            left_range = np.max(left_ankle_x) - np.min(left_ankle_x)
            right_range = np.max(right_ankle_x) - np.min(right_ankle_x)
            stride_length = (left_range + right_range) / 2

        ground_contact_ratio = self._estimate_ground_contact_ratio(keypoints_sequence, fps)

        return {
            'stride_length_norm': float(stride_length),
            'ground_contact_ratio': float(ground_contact_ratio),
            'flight_time_ratio': float(1.0 - ground_contact_ratio)
        }

    def _estimate_ground_contact_ratio(self, keypoints_sequence: List[Dict], fps: float) -> float:
        """估算触地时间比例"""
        ankle_velocities = []
        for i in range(1, len(keypoints_sequence)):
            curr = keypoints_sequence[i]['landmarks'][27]
            prev = keypoints_sequence[i - 1]['landmarks'][27]
            if curr['visibility'] > 0.5 and prev['visibility'] > 0.5:
                vy = (curr['y_norm'] - prev['y_norm']) * fps
                ankle_velocities.append(abs(vy))

        if not ankle_velocities:
            return 0.5

        threshold = np.percentile(ankle_velocities, 50)
        ground_frames = sum(1 for v in ankle_velocities if v < threshold)
        return ground_frames / len(ankle_velocities)

    def _calculate_stability_improved(self, keypoints_sequence: List[Dict]) -> Dict:
        """改进的稳定性计算"""
        trunk_stability = self._calculate_trunk_stability(keypoints_sequence)
        head_stability = self._calculate_head_stability(keypoints_sequence)
        gait_symmetry = self._calculate_gait_symmetry(keypoints_sequence)

        overall = (trunk_stability * 0.5 + head_stability * 0.2 + gait_symmetry * 0.3)

        return {
            'overall': float(overall),
            'trunk': float(trunk_stability),
            'head': float(head_stability),
            'symmetry': float(gait_symmetry),
            'rating': self._rate_stability(overall)
        }

    def _rate_stability(self, score: float) -> Dict:
        """评估稳定性"""
        if score >= 85:
            return {'level': 'excellent', 'score': 100, 'description': '动作非常稳定'}
        elif score >= 70:
            return {'level': 'good', 'score': 80, 'description': '动作稳定性良好'}
        elif score >= 55:
            return {'level': 'fair', 'score': 60, 'description': '稳定性一般'}
        else:
            return {'level': 'poor', 'score': 40, 'description': '动作稳定性需要改善'}

    def _calculate_trunk_stability(self, keypoints_sequence: List[Dict]) -> float:
        """计算躯干稳定性"""
        shoulder_x = []
        hip_x = []

        for kp in keypoints_sequence:
            ls, rs = kp['landmarks'][11], kp['landmarks'][12]
            if ls['visibility'] > 0.5 and rs['visibility'] > 0.5:
                shoulder_x.append((ls['x_norm'] + rs['x_norm']) / 2)

            lh, rh = kp['landmarks'][23], kp['landmarks'][24]
            if lh['visibility'] > 0.5 and rh['visibility'] > 0.5:
                hip_x.append((lh['x_norm'] + rh['x_norm']) / 2)

        if not shoulder_x or not hip_x:
            return 50.0

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

        min_len = min(len(left_knee_angles), len(right_knee_angles))
        correlation = np.corrcoef(left_knee_angles[:min_len], right_knee_angles[:min_len])[0, 1]

        if np.isnan(correlation):
            return 50.0

        symmetry = abs(correlation) * 100
        return max(0, min(symmetry, 100))

    def _calculate_body_lean(self, keypoints_sequence: List[Dict]) -> Dict:
        """计算身体前倾角度"""
        lean_angles = []

        for kp in keypoints_sequence:
            shoulder = kp['landmarks'][11]
            hip = kp['landmarks'][23]

            if shoulder['visibility'] > 0.5 and hip['visibility'] > 0.5:
                dx = shoulder['x_norm'] - hip['x_norm']
                dy = shoulder['y_norm'] - hip['y_norm']
                lean_angle = np.degrees(np.arctan2(dx, -dy))
                lean_angles.append(lean_angle)

        if not lean_angles:
            return {'mean_lean': 0.0, 'std_lean': 0.0, 'forward_lean': 0.0, 'rating': {}}

        lean_angles = self._smooth_signal_advanced(np.array(lean_angles))

        forward_leans = [a for a in lean_angles if a > 0]
        mean_forward = np.mean(forward_leans) if forward_leans else 0

        return {
            'mean_lean': float(np.mean(lean_angles)),
            'std_lean': float(np.std(lean_angles)),
            'forward_lean': float(mean_forward),
            'rating': self._rate_body_lean(mean_forward)
        }

    def _rate_body_lean(self, forward_lean: float) -> Dict:
        """评估身体前倾"""
        if 5 <= forward_lean <= 15:
            return {'level': 'optimal', 'score': 100, 'description': '前倾角度适中'}
        elif 3 <= forward_lean < 5 or 15 < forward_lean <= 20:
            return {'level': 'good', 'score': 80, 'description': '前倾角度可接受'}
        elif forward_lean < 3:
            return {'level': 'upright', 'score': 60, 'description': '身体过于直立'}
        else:
            return {'level': 'excessive', 'score': 50, 'description': '前倾过大'}

    def _calculate_arm_swing(self, keypoints_sequence: List[Dict]) -> Dict:
        """计算手臂摆动幅度"""
        left_elbow_angles = []
        right_elbow_angles = []

        for kp in keypoints_sequence:
            left_angle = self._calculate_joint_angle_safe(
                kp['landmarks'][11], kp['landmarks'][13], kp['landmarks'][15]
            )
            if not np.isnan(left_angle):
                left_elbow_angles.append(left_angle)

            right_angle = self._calculate_joint_angle_safe(
                kp['landmarks'][12], kp['landmarks'][14], kp['landmarks'][16]
            )
            if not np.isnan(right_angle):
                right_elbow_angles.append(right_angle)

        if not left_elbow_angles or not right_elbow_angles:
            return {'arm_swing_amplitude': 0.0, 'rating': {}}

        left_range = np.max(left_elbow_angles) - np.min(left_elbow_angles)
        right_range = np.max(right_elbow_angles) - np.min(right_elbow_angles)
        avg_amplitude = (left_range + right_range) / 2

        return {
            'arm_swing_amplitude': float(avg_amplitude),
            'left_arm_range': float(left_range),
            'right_arm_range': float(right_range),
            'rating': self._rate_arm_swing(avg_amplitude)
        }

    def _rate_arm_swing(self, amplitude: float) -> Dict:
        """评估手臂摆动"""
        if 30 <= amplitude <= 60:
            return {'level': 'optimal', 'score': 100, 'description': '手臂摆动幅度适中'}
        elif 20 <= amplitude < 30 or 60 < amplitude <= 80:
            return {'level': 'good', 'score': 80, 'description': '手臂摆动良好'}
        elif amplitude < 20:
            return {'level': 'restricted', 'score': 60, 'description': '手臂摆动受限'}
        else:
            return {'level': 'excessive', 'score': 60, 'description': '手臂摆动过大'}

    def _smooth_signal_advanced(self, signal: np.ndarray) -> np.ndarray:
        """高级信号平滑"""
        if len(signal) < 5:
            return signal

        from scipy.ndimage import median_filter
        signal = median_filter(signal, size=3)

        if len(signal) > 6:
            b, a = butter(2, 0.2, btype='low')
            signal = filtfilt(b, a, signal)

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
            'view_angle': 'unknown',
            'trunk_reference': 0,
            'angles': {},
            'vertical_motion': self._get_empty_vertical_motion(),
            'cadence': {},
            'stride_info': {},
            'stability': {},
            'body_lean': {},
            'arm_swing': {},
            'gait_cycle': {},
        }


# 模块测试
if __name__ == "__main__":
    print("=" * 60)
    print("测试重构版运动学分析模块")
    print("=" * 60)

    # 生成模拟数据
    mock_keypoints = []
    fps = 30
    duration = 3
    num_frames = fps * duration

    for i in range(num_frames):
        kp = {'detected': True, 'landmarks': []}
        t = i / fps
        phase = t * 2 * np.pi * 2

        for j in range(33):
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

    analyzer = KinematicAnalyzer()
    results = analyzer.analyze_sequence(mock_keypoints, fps, view_angle='side')

    print(f"\n步频: {results['cadence']['cadence']:.1f} 步/分")
    print(f"垂直振幅(归一化): {results['vertical_motion']['amplitude_normalized']:.2f}%")
    print(f"振幅评级: {results['vertical_motion']['amplitude_rating']}")
    print(f"\n膝关节分阶段分析:")
    phase_analysis = results['angles']['phase_analysis']
    print(f"  触地期平均角度: {phase_analysis['ground_contact']['mean']:.1f}°")
    print(f"  腾空期平均角度: {phase_analysis['flight']['mean']:.1f}°")
    print(f"  最大弯曲角度: {phase_analysis['max_flexion']:.1f}°")
    print(f"  关节活动范围: {phase_analysis['range_of_motion']:.1f}°")

    print("\n✅ 模块测试完成!")
