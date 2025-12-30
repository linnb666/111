# modules/kinematic_analyzer.py
"""
运动学特征解析模块

支持2D和3D分析模式：
- 2D模式：使用MediaPipe的y坐标进行分析（后备方案）
- 3D模式：使用MotionBERT的3D坐标进行分析（推荐）

核心改进：
1. 垂直振幅使用躯干长度归一化
2. 膝关节角度分阶段分析
3. 支持不同视角的分析策略
4. 更精确的步态周期检测
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple, Optional
from config.config import KINEMATIC_CONFIG

# 导入3D分析器
try:
    from modules.kinematic_analyzer_3d import KinematicAnalyzer3D, KinematicAnalyzer3DWrapper
    HAS_3D_ANALYZER = True
except ImportError:
    HAS_3D_ANALYZER = False


def create_kinematic_analyzer(prefer_3d: bool = True):
    """
    创建运动学分析器的工厂函数

    Args:
        prefer_3d: 是否优先使用3D分析器

    Returns:
        运动学分析器实例
    """
    if prefer_3d and HAS_3D_ANALYZER:
        return KinematicAnalyzer3DWrapper()
    return KinematicAnalyzer()


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

            # 稳定性指标（侧面不评估左右对称性）
            'stability': self._calculate_stability_side_view(valid_frames),
            'body_lean': self._calculate_body_lean(valid_frames),
            'arm_swing': self._calculate_arm_swing(valid_frames),

            # 步态周期分析（包含毫秒时间）
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

        # ⭐ 使用新方法计算落地膝角（方案A：取峰值前的帧）
        landing_angles_result = self._calculate_landing_knee_angles(
            keypoints_sequence, fps, knee_left_smooth, knee_right_smooth
        )

        # 将落地膝角结果合并到触地期统计中（包含每步统计）
        if landing_angles_result['landing_count'] > 0:
            phase_angles['ground_contact']['landing_angle_mean'] = landing_angles_result['landing_angle_mean']
            phase_angles['ground_contact']['landing_angle_std'] = landing_angles_result.get('landing_angle_std', 0)
            phase_angles['ground_contact']['landing_count'] = landing_angles_result['landing_count']
            phase_angles['ground_contact']['landing_angles'] = landing_angles_result['landing_angles']
            phase_angles['ground_contact']['per_step_stats'] = landing_angles_result.get('per_step_stats', [])
            # 使用落地瞬间的角度作为主要指标
            phase_angles['ground_contact']['mean'] = landing_angles_result['landing_angle_mean']
            phase_angles['ground_contact']['std'] = landing_angles_result.get('landing_angle_std', 0)

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
        检测每一帧的步态阶段（重写版：基于速度的状态机）

        核心改进：
        1. 使用脚踝Y坐标速度判断状态
        2. 基于脚踝高度（Y值）确定触地/腾空
        3. 自适应阈值（按帧率缩放）
        """
        n_frames = len(keypoints_sequence)
        if n_frames < 3:
            return [self.PHASE_TRANSITION] * n_frames

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

        # 计算每帧取较低脚踝的Y值（触地的脚）
        lower_ankle_y = np.maximum(left_ankle_y, right_ankle_y)  # Y轴向下，值大=位置低

        # 计算速度（Y坐标变化率）
        velocity = np.gradient(lower_ankle_y) * fps

        # 自适应阈值计算
        y_median = np.median(lower_ankle_y)
        y_std = np.std(lower_ankle_y)
        y_max = np.max(lower_ankle_y)  # 最低点（触地）
        y_min = np.min(lower_ankle_y)  # 最高点（腾空）
        y_range = y_max - y_min

        # 阈值设置（基于数据分布）
        ground_threshold = y_max - y_range * 0.25  # 上方25%范围为触地
        flight_threshold = y_min + y_range * 0.35  # 下方35%范围为腾空
        velocity_threshold = y_std * fps * 0.3  # 速度阈值

        # 状态机检测相位
        phases = []
        for i in range(n_frames):
            y = lower_ankle_y[i]
            v = abs(velocity[i]) if i < len(velocity) else 0

            if y >= ground_threshold and v < velocity_threshold * 2:
                # 脚踝位置低且速度小 = 触地
                phases.append(self.PHASE_GROUND_CONTACT)
            elif y <= flight_threshold:
                # 脚踝位置高 = 腾空
                phases.append(self.PHASE_FLIGHT)
            else:
                # 中间区域 = 过渡
                phases.append(self.PHASE_TRANSITION)

        # 后处理：消除孤立状态（至少连续2帧才算有效状态）
        phases = self._smooth_phases(phases, min_duration=2)

        return phases

    def _smooth_phases(self, phases: List[int], min_duration: int = 2) -> List[int]:
        """平滑相位序列，消除孤立的错误检测"""
        if len(phases) < min_duration * 2:
            return phases

        smoothed = phases.copy()

        # 检测并修复孤立状态
        i = 0
        while i < len(smoothed):
            # 找到当前状态的连续区间
            start = i
            current_phase = smoothed[i]
            while i < len(smoothed) and smoothed[i] == current_phase:
                i += 1
            duration = i - start

            # 如果持续时间太短，用前后状态替换
            if duration < min_duration and start > 0 and i < len(smoothed):
                # 使用前一个状态填充
                prev_phase = smoothed[start - 1]
                for j in range(start, i):
                    smoothed[j] = prev_phase

        return smoothed

    def _analyze_angles_by_phase(self, knee_left: List[float],
                                  knee_right: List[float],
                                  phases: List[int]) -> Dict:
        """
        按阶段统计膝关节角度
        方案A重写：使用峰值检测+加速度，取落地前3-5帧（脚即将着地时）
        """
        # 分组收集各阶段的角度
        ground_contact_angles = []
        flight_angles = []
        transition_angles = []

        # 收集各阶段所有帧的角度（用于其他统计）
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

        # 触地期统计
        gc_stats = safe_stats(ground_contact_angles)

        return {
            # 触地期角度（理想范围：155-170°）
            'ground_contact': gc_stats,
            # 腾空期/摆动期角度（理想范围：90-130°，弯曲较大）
            'flight': safe_stats(flight_angles),
            # 过渡期角度
            'transition': safe_stats(transition_angles),
            # 关键指标
            'max_flexion': float(np.min(knee_all)) if knee_all else 0,
            'max_extension': float(np.max(knee_all)) if knee_all else 0,
            'range_of_motion': float(np.max(knee_all) - np.min(knee_all)) if knee_all else 0,
        }

    def _detect_landing_windows(self, keypoints_sequence: List[Dict], fps: float) -> List[Dict]:
        """
        检测落地区间（重构版：基于周期性 + 相对阈值）

        核心改进：
        1. 检测落地"区间"而非单一帧
        2. 使用IQR/百分位数自适应阈值
        3. 周期性验证（相邻落地间隔应相似）
        4. 分别检测左右脚落地

        返回：落地区间列表，每个包含 {start, peak, end, foot, duration_ms}
        """
        n_frames = len(keypoints_sequence)
        if n_frames < 15:
            return []

        # 提取左右脚踝Y坐标
        left_ankle_y = []
        right_ankle_y = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]
            left_y = left['y_norm'] if left['visibility'] > 0.5 else np.nan
            right_y = right['y_norm'] if right['visibility'] > 0.5 else np.nan
            left_ankle_y.append(left_y)
            right_ankle_y.append(right_y)

        left_ankle_y = self._interpolate_nans(np.array(left_ankle_y))
        right_ankle_y = self._interpolate_nans(np.array(right_ankle_y))

        # 轻度平滑（Savitzky-Golay，保留峰值特征）
        window_size = min(7, len(left_ankle_y) // 3)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size >= 3:
            left_ankle_y = savgol_filter(left_ankle_y, window_size, 2)
            right_ankle_y = savgol_filter(right_ankle_y, window_size, 2)

        # 分别检测左右脚落地区间
        left_windows = self._detect_foot_landing_windows(left_ankle_y, fps, 'left')
        right_windows = self._detect_foot_landing_windows(right_ankle_y, fps, 'right')

        # 合并并按时间排序
        all_windows = left_windows + right_windows
        all_windows.sort(key=lambda x: x['peak'])

        # 周期性验证：过滤异常间隔的落地
        validated_windows = self._validate_landing_periodicity(all_windows, fps)

        print(f"  落地区间检测：左脚{len(left_windows)}次，右脚{len(right_windows)}次")
        print(f"  周期性验证后：{len(validated_windows)}次有效落地")

        return validated_windows

    def _detect_foot_landing_windows(self, ankle_y: np.ndarray, fps: float, foot: str) -> List[Dict]:
        """
        检测单脚的落地区间

        使用相对阈值（基于IQR）而非硬编码常数
        """
        n = len(ankle_y)
        if n < 10:
            return []

        # 计算自适应阈值（基于IQR）
        q25 = np.percentile(ankle_y, 25)
        q75 = np.percentile(ankle_y, 75)
        iqr = q75 - q25
        median = np.median(ankle_y)

        # 峰值检测参数（基于数据分布）
        # prominence：至少要突出IQR的30%
        min_prominence = iqr * 0.3
        # distance：基于典型步频（150-200步/分），最小间隔约0.3秒
        min_distance = max(3, int(fps * 0.25))

        # 检测峰值（Y值最大=位置最低=触地）
        peaks, properties = find_peaks(
            ankle_y,
            prominence=max(min_prominence, 0.01),  # 至少0.01防止太小
            distance=min_distance
        )

        if len(peaks) == 0:
            return []

        # 为每个峰值确定落地区间
        landing_windows = []
        prominences = properties['prominences']

        # 区间边界阈值：峰值高度减去prominence的50%
        for i, peak in enumerate(peaks):
            peak_value = ankle_y[peak]
            prominence = prominences[i]
            boundary_threshold = peak_value - prominence * 0.5

            # 向前找区间开始
            start = peak
            for j in range(peak - 1, max(0, peak - int(fps * 0.2)) - 1, -1):
                if ankle_y[j] < boundary_threshold:
                    start = j + 1
                    break

            # 向后找区间结束
            end = peak
            for j in range(peak + 1, min(n, peak + int(fps * 0.2))):
                if ankle_y[j] < boundary_threshold:
                    end = j
                    break

            duration_frames = int(end - start + 1)
            duration_ms = float(duration_frames * 1000.0 / fps)

            # 合理性检查：触地区间应在50-300ms之间
            if 50 <= duration_ms <= 300:
                landing_windows.append({
                    'start': int(start),
                    'peak': int(peak),
                    'end': int(end),
                    'foot': str(foot),
                    'duration_frames': duration_frames,
                    'duration_ms': duration_ms,
                    'peak_value': float(peak_value),
                    'prominence': float(prominence)
                })

        return landing_windows

    def _validate_landing_periodicity(self, windows: List[Dict], fps: float) -> List[Dict]:
        """
        基于周期性验证落地检测的有效性

        核心逻辑：相邻落地间隔应相似（跑步是周期性运动）
        使用IQR方法剔除异常间隔
        """
        if len(windows) < 3:
            return windows

        # 计算相邻落地间隔
        intervals = []
        for i in range(1, len(windows)):
            interval = windows[i]['peak'] - windows[i-1]['peak']
            intervals.append(interval)

        if len(intervals) < 2:
            return windows

        intervals = np.array(intervals)

        # 使用IQR方法确定合理间隔范围
        q25 = np.percentile(intervals, 25)
        q75 = np.percentile(intervals, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        # 同时考虑物理约束：步频通常在120-220步/分
        min_interval = fps * 60 / 220  # 约0.27秒
        max_interval = fps * 60 / 120  # 约0.5秒

        lower_bound = max(lower_bound, min_interval)
        upper_bound = min(upper_bound, max_interval)

        # 标记有效的落地
        valid_windows = [windows[0]]  # 第一个默认保留
        for i in range(1, len(windows)):
            interval = windows[i]['peak'] - windows[i-1]['peak']
            if lower_bound <= interval <= upper_bound:
                valid_windows.append(windows[i])

        return valid_windows

    def _calculate_landing_knee_angles(self, keypoints_sequence: List[Dict],
                                        fps: float,
                                        knee_left: List[float],
                                        knee_right: List[float]) -> Dict:
        """
        计算落地时刻的膝关节角度（生物力学约束版）

        核心原则：
        1. 宁可少输出，也不输出错误的落地膝角
        2. 使用生物力学硬约束过滤不可信的检测
        3. 每个落地都给出通过/拒绝原因

        生物力学约束：
        - 落地膝角范围：145° ≤ angle ≤ 175°（中长跑标准）
        - 伸展趋势：落地前膝关节应处于伸展状态（d(knee)/dt ≥ 0）
        """
        # 生物力学参数（硬约束）
        MIN_LANDING_ANGLE = 145.0  # 最小落地膝角
        MAX_LANDING_ANGLE = 175.0  # 最大落地膝角
        PRE_LANDING_WINDOW_MS = 100.0  # 峰值前的搜索窗口（毫秒）

        # 检测落地区间
        landing_windows = self._detect_landing_windows(keypoints_sequence, fps)

        if not landing_windows:
            return {
                'landing_angle_mean': 0,
                'landing_angle_std': 0,
                'landing_count': 0,
                'valid_count': 0,
                'rejected_count': 0,
                'landing_angles': [],
                'per_step_stats': [],
                'rejected_steps': [],
                'method': 'biomechanical_constrained_v3'
            }

        # 计算膝角变化率（用于伸展趋势判断）
        # 注意：使用有符号的梯度，正值=伸展，负值=屈曲
        knee_left_arr = np.array(knee_left)
        knee_right_arr = np.array(knee_right)
        knee_left_rate = np.gradient(knee_left_arr)  # 有符号
        knee_right_rate = np.gradient(knee_right_arr)

        # 计算峰值前搜索窗口的帧数
        pre_landing_frames = int(PRE_LANDING_WINDOW_MS * fps / 1000.0)
        pre_landing_frames = max(2, min(pre_landing_frames, 6))  # 限制在2-6帧

        # 对每个落地候选应用生物力学约束
        valid_landings = []
        rejected_landings = []

        for window in landing_windows:
            result = self._validate_landing_biomechanics(
                knee_left_arr, knee_right_arr,
                knee_left_rate, knee_right_rate,
                window['peak'], window['foot'],
                pre_landing_frames,
                MIN_LANDING_ANGLE, MAX_LANDING_ANGLE
            )

            step_info = {
                'peak_frame': int(window['peak']),
                'foot': str(window['foot']),
                'duration_ms': float(window['duration_ms'])
            }

            if result['valid']:
                step_info.update({
                    'landing_angle': float(result['landing_angle']),
                    'extension_rate': float(result['extension_rate']),
                    'confidence': result['confidence']
                })
                valid_landings.append(step_info)
            else:
                step_info.update({
                    'rejection_reason': result['rejection_reason'],
                    'actual_angle': float(result.get('actual_angle', 0)),
                    'actual_rate': float(result.get('actual_rate', 0))
                })
                rejected_landings.append(step_info)

        # 整体汇总统计
        landing_angles = [s['landing_angle'] for s in valid_landings]

        print(f"  落地膝角分析（生物力学约束版）：")
        print(f"    候选落地: {len(landing_windows)} 次")
        print(f"    通过约束: {len(valid_landings)} 次")
        print(f"    被拒绝: {len(rejected_landings)} 次")

        if rejected_landings:
            reasons = {}
            for r in rejected_landings:
                reason = r['rejection_reason']
                reasons[reason] = reasons.get(reason, 0) + 1
            print(f"    拒绝原因: {reasons}")

        if landing_angles:
            mean_angle = float(np.mean(landing_angles))
            std_angle = float(np.std(landing_angles)) if len(landing_angles) > 1 else 0.0
            print(f"    有效角度: {[f'{a:.1f}°' for a in landing_angles]}")
            print(f"    平均: {mean_angle:.1f}°, 标准差: {std_angle:.1f}°")

            return {
                'landing_angle_mean': mean_angle,
                'landing_angle_std': std_angle,
                'landing_count': int(len(landing_angles)),
                'valid_count': int(len(valid_landings)),
                'rejected_count': int(len(rejected_landings)),
                'landing_angles': [float(a) for a in landing_angles],
                'per_step_stats': valid_landings,
                'rejected_steps': rejected_landings,
                'method': 'biomechanical_constrained_v3'
            }

        return {
            'landing_angle_mean': 0,
            'landing_angle_std': 0,
            'landing_count': 0,
            'valid_count': 0,
            'rejected_count': int(len(rejected_landings)),
            'landing_angles': [],
            'per_step_stats': [],
            'rejected_steps': rejected_landings,
            'method': 'biomechanical_constrained_v3'
        }

    def _validate_landing_biomechanics(self, knee_left: np.ndarray, knee_right: np.ndarray,
                                        knee_left_rate: np.ndarray, knee_right_rate: np.ndarray,
                                        peak: int, foot: str, window_frames: int,
                                        min_angle: float, max_angle: float) -> Dict:
        """
        使用生物力学约束验证单次落地

        约束条件：
        1. 膝角范围：min_angle ≤ angle ≤ max_angle
        2. 伸展趋势：落地前膝关节应在伸展（rate ≥ 0）

        搜索策略：
        - 在peak前的固定时间窗内搜索
        - 只保留同时满足角度范围+伸展趋势的帧
        - 在合法帧中选取最大角度
        """
        n = len(knee_left)

        # 确定搜索范围：peak前的window_frames帧
        search_start = max(0, peak - window_frames)
        search_end = peak  # 不包括peak本身（peak时已经在缓冲）

        if search_start >= search_end:
            return {'valid': False, 'rejection_reason': 'window_too_small'}

        # 选择对应脚的膝角数据
        if foot == 'left':
            angles = knee_left[search_start:search_end]
            rates = knee_left_rate[search_start:search_end]
        elif foot == 'right':
            angles = knee_right[search_start:search_end]
            rates = knee_right_rate[search_start:search_end]
        else:
            angles = (knee_left[search_start:search_end] + knee_right[search_start:search_end]) / 2
            rates = (knee_left_rate[search_start:search_end] + knee_right_rate[search_start:search_end]) / 2

        # 过滤NaN
        valid_mask = ~np.isnan(angles)
        if not np.any(valid_mask):
            return {'valid': False, 'rejection_reason': 'no_valid_frames'}

        angles = angles[valid_mask]
        rates = rates[valid_mask]

        # 约束1：膝角范围 [min_angle, max_angle]
        angle_valid = (angles >= min_angle) & (angles <= max_angle)

        # 约束2：伸展趋势（rate ≥ -1，允许轻微波动）
        # 使用-1而非0是为了容忍测量噪声
        extension_valid = rates >= -1.0

        # 同时满足两个约束
        both_valid = angle_valid & extension_valid

        # 统计分析
        max_angle_in_window = float(np.max(angles))
        mean_rate_in_window = float(np.mean(rates))

        if not np.any(both_valid):
            # 判断拒绝原因
            if not np.any(angle_valid):
                if max_angle_in_window < min_angle:
                    return {
                        'valid': False,
                        'rejection_reason': 'angle_too_low',
                        'actual_angle': max_angle_in_window,
                        'actual_rate': mean_rate_in_window
                    }
                else:
                    return {
                        'valid': False,
                        'rejection_reason': 'angle_too_high',
                        'actual_angle': max_angle_in_window,
                        'actual_rate': mean_rate_in_window
                    }
            else:
                return {
                    'valid': False,
                    'rejection_reason': 'flexion_trend',
                    'actual_angle': max_angle_in_window,
                    'actual_rate': mean_rate_in_window
                }

        # 在合法帧中选取最大角度
        valid_angles = angles[both_valid]
        valid_rates = rates[both_valid]

        best_angle = float(np.max(valid_angles))
        best_idx = np.argmax(valid_angles)
        best_rate = float(valid_rates[best_idx])

        # 计算置信度（基于合法帧占比和角度稳定性）
        valid_ratio = np.sum(both_valid) / len(angles)
        angle_stability = 1.0 - min(np.std(valid_angles) / 10.0, 1.0) if len(valid_angles) > 1 else 0.5
        confidence = (valid_ratio * 0.5 + angle_stability * 0.5)

        return {
            'valid': True,
            'landing_angle': best_angle,
            'extension_rate': best_rate,
            'confidence': f"{confidence:.0%}",
            'valid_frame_count': int(np.sum(both_valid)),
            'total_frame_count': int(len(angles))
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
        分析完整步态周期（优化版：更精确的触地时间检测）

        帧率自适应：根据fps调整过滤阈值
        """
        # 使用改进的触地检测算法
        ground_contacts, flight_phases = self._detect_ground_contacts_improved(keypoints_sequence, fps)

        frame_duration_ms = 1000.0 / fps

        # 帧率自适应的过滤范围
        # 精英跑者触地时间：160-220ms
        # 普通跑者触地时间：220-300ms
        # 较差跑者触地时间：280-400ms
        min_gc_ms = 120  # 最小触地时间（考虑测量误差）
        max_gc_ms = 450  # 最大触地时间

        # 腾空时间范围
        min_flight_ms = 40  # 最小腾空时间
        max_flight_ms = 350  # 最大腾空时间

        # 计算触地时间（毫秒）
        ground_contact_durations_ms = []
        for gc in ground_contacts:
            duration_ms = gc['duration_frames'] * frame_duration_ms
            if min_gc_ms <= duration_ms <= max_gc_ms:
                ground_contact_durations_ms.append(duration_ms)

        # 计算腾空时间（毫秒）
        flight_durations_ms = []
        for fl in flight_phases:
            duration_ms = fl['duration_frames'] * frame_duration_ms
            if min_flight_ms <= duration_ms <= max_flight_ms:
                flight_durations_ms.append(duration_ms)

        # 使用稳健统计（去除异常值后的平均）
        if len(ground_contact_durations_ms) >= 3:
            # 去除最高和最低值后取平均
            sorted_gc = sorted(ground_contact_durations_ms)
            trimmed_gc = sorted_gc[1:-1] if len(sorted_gc) > 2 else sorted_gc
            avg_ground_contact_ms = float(np.mean(trimmed_gc))
        elif ground_contact_durations_ms:
            avg_ground_contact_ms = float(np.median(ground_contact_durations_ms))
        else:
            avg_ground_contact_ms = 0

        if len(flight_durations_ms) >= 3:
            sorted_fl = sorted(flight_durations_ms)
            trimmed_fl = sorted_fl[1:-1] if len(sorted_fl) > 2 else sorted_fl
            avg_flight_ms = float(np.mean(trimmed_fl))
        elif flight_durations_ms:
            avg_flight_ms = float(np.median(flight_durations_ms))
        else:
            avg_flight_ms = 0

        # 计算步态周期时间
        if avg_ground_contact_ms > 0 and avg_flight_ms > 0:
            avg_cycle_duration_ms = avg_ground_contact_ms + avg_flight_ms
        else:
            avg_cycle_duration_ms = 0

        # 计算比例
        total_time = avg_ground_contact_ms + avg_flight_ms
        if total_time > 0:
            ground_contact_ratio = avg_ground_contact_ms / total_time
            flight_ratio = avg_flight_ms / total_time
        else:
            # 使用默认比例（跑步典型值）
            ground_contact_ratio = 0.45
            flight_ratio = 0.35

        # 过渡期比例（从相位检测中计算）
        phases = self._detect_gait_phases(keypoints_sequence, fps)
        if phases:
            transition_count = sum(1 for p in phases if p == self.PHASE_TRANSITION)
            transition_ratio = transition_count / len(phases)
            # 重新归一化
            total = ground_contact_ratio + flight_ratio + transition_ratio
            if total > 0:
                ground_contact_ratio /= total
                flight_ratio /= total
                transition_ratio /= total
        else:
            transition_ratio = 0.20

        return {
            'phase_distribution': {
                'ground_contact': float(round(ground_contact_ratio, 3)),
                'flight': float(round(flight_ratio, 3)),
                'transition': float(round(transition_ratio, 3)),
            },
            # 各阶段时间（毫秒）
            'phase_duration_ms': {
                'ground_contact': float(round(avg_ground_contact_ms, 1)),
                'flight': float(round(avg_flight_ms, 1)),
                'transition': 0.0,
            },
            'avg_cycle_duration': float(avg_cycle_duration_ms / 1000) if avg_cycle_duration_ms > 0 else 0,
            'avg_cycle_duration_ms': float(round(avg_cycle_duration_ms, 1)),
            'cycle_count': len(ground_contact_durations_ms),
            'ground_contact_times': ground_contact_durations_ms,  # 详细数据
            'flight_times': flight_durations_ms,
            # 评估
            'gait_rating': self._rate_gait_timing(avg_ground_contact_ms),
        }

    def _detect_ground_contacts_improved(self, keypoints_sequence: List[Dict], fps: float) -> Tuple[List[Dict], List[Dict]]:
        """
        改进的触地检测算法（重写版）

        核心改进：
        1. 使用自适应阈值（基于数据分布）
        2. 基于速度变化检测触地/离地时刻
        3. 分别检测左右脚，支持精英跑者的快速步频
        4. 帧率自适应

        预期触地时间：160-250ms（精英到普通跑者）
        """
        n_frames = len(keypoints_sequence)

        # 提取左右脚踝Y坐标
        left_ankle_y = []
        right_ankle_y = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][27]
            right = kp['landmarks'][28]

            left_y = left['y_norm'] if left['visibility'] > 0.5 else np.nan
            right_y = right['y_norm'] if right['visibility'] > 0.5 else np.nan

            left_ankle_y.append(left_y)
            right_ankle_y.append(right_y)

        left_ankle_y = self._interpolate_nans(np.array(left_ankle_y))
        right_ankle_y = self._interpolate_nans(np.array(right_ankle_y))

        # 平滑处理（轻度平滑，保留细节）
        if len(left_ankle_y) > 5:
            # 使用较小的平滑窗口
            window = min(5, len(left_ankle_y) // 2)
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                left_ankle_y = savgol_filter(left_ankle_y, window, 2)
                right_ankle_y = savgol_filter(right_ankle_y, window, 2)

        # 帧率自适应参数
        frame_duration_ms = 1000.0 / fps
        min_gc_frames = max(2, int(160 / frame_duration_ms))  # 最小触地帧数（160ms）
        max_gc_frames = int(350 / frame_duration_ms)  # 最大触地帧数（350ms）
        min_peak_distance = max(3, int(fps * 0.15))  # 最小150ms间隔

        # 自适应prominence计算
        left_range = np.max(left_ankle_y) - np.min(left_ankle_y)
        right_range = np.max(right_ankle_y) - np.min(right_ankle_y)
        prominence = min(left_range, right_range) * 0.15  # 范围的15%作为prominence

        # 检测左脚触地峰值
        left_peaks, _ = find_peaks(left_ankle_y, distance=min_peak_distance, prominence=prominence)
        # 检测右脚触地峰值
        right_peaks, _ = find_peaks(right_ankle_y, distance=min_peak_distance, prominence=prominence)

        ground_contacts = []
        flight_phases = []

        # 处理左脚触地
        for peak in left_peaks:
            gc = self._extract_single_ground_contact(
                left_ankle_y, peak, fps, min_gc_frames, max_gc_frames, 'left'
            )
            if gc:
                ground_contacts.append(gc)

        # 处理右脚触地
        for peak in right_peaks:
            gc = self._extract_single_ground_contact(
                right_ankle_y, peak, fps, min_gc_frames, max_gc_frames, 'right'
            )
            if gc:
                ground_contacts.append(gc)

        # 按时间排序
        ground_contacts.sort(key=lambda x: x['start_frame'])

        # 计算腾空时间（相邻触地之间）
        for i in range(len(ground_contacts) - 1):
            current_end = ground_contacts[i]['end_frame']
            next_start = ground_contacts[i + 1]['start_frame']
            flight_duration = next_start - current_end

            if flight_duration >= 2:  # 至少2帧
                flight_phases.append({
                    'start_frame': current_end,
                    'end_frame': next_start,
                    'duration_frames': flight_duration
                })

        return ground_contacts, flight_phases

    def _extract_single_ground_contact(self, ankle_y: np.ndarray, peak_frame: int,
                                        fps: float, min_frames: int, max_frames: int,
                                        foot: str) -> Optional[Dict]:
        """
        提取单次触地的起止帧

        使用速度过零点检测触地开始和结束
        """
        n = len(ankle_y)
        peak_y = ankle_y[peak_frame]

        # 计算速度
        velocity = np.gradient(ankle_y)

        # 方法1：使用速度符号变化检测
        # 触地开始：速度从正变负或接近零（脚下降到触地）
        # 触地结束：速度从负变正（脚开始抬起）

        # 向前找触地开始（速度从正变为接近零的位置）
        start_frame = peak_frame
        search_start = max(0, peak_frame - max_frames)
        for j in range(peak_frame - 1, search_start, -1):
            # 当Y值明显低于峰值时，认为还没触地
            if ankle_y[j] < peak_y - (peak_y - np.min(ankle_y)) * 0.3:
                start_frame = j + 1
                break
            # 或者速度方向改变
            if j > 0 and velocity[j] > 0 and velocity[j-1] <= 0:
                start_frame = j
                break

        # 向后找触地结束
        end_frame = peak_frame
        search_end = min(n, peak_frame + max_frames)
        for j in range(peak_frame + 1, search_end):
            # 当Y值明显低于峰值时，认为已离地
            if ankle_y[j] < peak_y - (peak_y - np.min(ankle_y)) * 0.3:
                end_frame = j
                break
            # 或者速度方向改变（开始上升）
            if j < n - 1 and velocity[j] < 0 and velocity[j+1] >= 0:
                end_frame = j + 1
                break

        duration_frames = end_frame - start_frame

        # 验证触地时长合理性
        if min_frames <= duration_frames <= max_frames:
            return {
                'start_frame': start_frame,
                'end_frame': end_frame,
                'peak_frame': peak_frame,
                'duration_frames': duration_frames,
                'foot': foot
            }

        return None

    def _rate_gait_timing(self, ground_contact_ms: float) -> Dict:
        """
        评估触地时间（新标准）
        <210ms精英，210-240ms优秀，240-270ms良好，270-300ms一般，>300ms较差
        """
        if ground_contact_ms <= 0:
            return {'level': 'unknown', 'score': 0, 'description': '数据不足'}
        elif ground_contact_ms < 210:
            return {'level': 'elite', 'score': 100, 'description': '精英水平'}
        elif ground_contact_ms < 240:
            return {'level': 'excellent', 'score': 90, 'description': '优秀'}
        elif ground_contact_ms < 270:
            return {'level': 'good', 'score': 75, 'description': '良好'}
        elif ground_contact_ms < 300:
            return {'level': 'fair', 'score': 60, 'description': '一般'}
        else:
            return {'level': 'poor', 'score': 45, 'description': '较差'}

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
        """正面/后方视角分析 - 侧重力线和肩部稳定（移除对称性）"""
        results = {
            'lower_limb_alignment': self._calculate_lower_limb_alignment(valid_frames),
            'lateral_stability': self._calculate_lateral_stability(valid_frames),
            'vertical_motion': self._calculate_vertical_motion_normalized(
                valid_frames, fps, trunk_length
            ),
            'cadence': self._calculate_cadence_improved(valid_frames, fps),
            'stability': self._calculate_stability_front_view(valid_frames),
            'gait_cycle': self._analyze_gait_cycle(valid_frames, fps),
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

        # 统计分析（提高灵敏度）
        def analyze_alignment(angles, side):
            if not angles:
                return {'mean': 0, 'max': 0, 'issue': 'unknown', 'severity': 'unknown'}

            mean_angle = np.mean(angles)
            max_angle = np.max(np.abs(angles))
            std_angle = np.std(angles)

            # 判断问题类型（降低阈值提高灵敏度）
            # 使用2度作为正常/异常的分界点
            if mean_angle > 2:
                issue = 'valgus'  # 膝外翻
                if mean_angle < 5:
                    severity = 'mild'
                elif mean_angle < 8:
                    severity = 'moderate'
                else:
                    severity = 'severe'
            elif mean_angle < -2:
                issue = 'varus'  # 膝内扣
                if mean_angle > -5:
                    severity = 'mild'
                elif mean_angle > -8:
                    severity = 'moderate'
                else:
                    severity = 'severe'
            else:
                issue = 'normal'
                severity = 'none'

            # 添加动态偏移检测（标准差大说明存在不稳定）
            if std_angle > 3:
                if issue == 'normal':
                    issue = 'unstable'
                    severity = 'mild'

            return {
                'mean': float(mean_angle),
                'max': float(max_angle),
                'std': float(std_angle),
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
        """评估下肢力线（提高灵敏度）"""
        if not left_angles or not right_angles:
            return {'level': 'unknown', 'score': 0, 'description': '数据不足'}

        # 使用平均偏移和最大偏移的加权组合
        left_mean = np.mean(np.abs(left_angles)) if left_angles else 0
        right_mean = np.mean(np.abs(right_angles)) if right_angles else 0
        mean_deviation = (left_mean + right_mean) / 2

        max_deviation = max(
            np.max(np.abs(left_angles)) if left_angles else 0,
            np.max(np.abs(right_angles)) if right_angles else 0
        )

        # 综合得分：70%平均偏移 + 30%最大偏移
        combined_deviation = mean_deviation * 0.7 + max_deviation * 0.3

        # 更严格的阈值
        if combined_deviation < 2:
            return {'level': 'excellent', 'score': 100, 'description': '下肢力线非常标准'}
        elif combined_deviation < 4:
            return {'level': 'good', 'score': 85, 'description': '下肢力线良好'}
        elif combined_deviation < 6:
            return {'level': 'fair', 'score': 70, 'description': '存在轻度膝关节偏移'}
        elif combined_deviation < 10:
            return {'level': 'moderate', 'score': 55, 'description': '膝关节偏移需要注意'}
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

        # 只使用踝关节方法（用户要求）
        cadences = [cadence1, cadence2, cadence3]
        step_counts = [step_count1, step_count2, step_count3]
        weights = [0, 1, 0]  # 仅踝关节

        valid_data = [(c, s, w) for c, s, w in zip(cadences, step_counts, weights) if c > 0 and s > 0]
        if valid_data:
            total_weight = sum(w for _, _, w in valid_data)
            weighted_cadence = sum(c * w for c, _, w in valid_data) / total_weight
            # 步数也使用加权平均，保持与步频一致
            weighted_step_count = sum(s * w for _, s, w in valid_data) / total_weight
            avg_step_count = int(round(weighted_step_count))
        else:
            weighted_cadence = 0
            avg_step_count = 0

        duration = len(keypoints_sequence) / fps

        # 计算预期步数（基于加权步频）用于验证
        expected_steps = weighted_cadence * duration / 60 if weighted_cadence > 0 else 0

        return {
            'cadence': float(weighted_cadence),
            'step_count': avg_step_count,
            'duration': float(duration),
            'expected_steps': float(expected_steps),  # 新增：理论步数（用于解释）
            'cadence_knee': float(cadence1),
            'cadence_ankle': float(cadence2),
            'cadence_hip': float(cadence3),
            'step_count_knee': step_count1,
            'step_count_ankle': step_count2,
            'step_count_hip': step_count3,
            'confidence': self._calculate_cadence_confidence([cadence1, cadence2, cadence3]),
            'rating': self._rate_cadence(weighted_cadence),
        }

    def _rate_cadence(self, cadence: float) -> Dict:
        """评估步频（新标准：5个等级）"""
        if cadence >= 185:
            return {'level': 'elite', 'score': 100, 'description': '精英'}
        elif cadence >= 175:
            return {'level': 'excellent', 'score': 90, 'description': '优秀'}
        elif cadence >= 165:
            return {'level': 'good', 'score': 75, 'description': '良好'}
        elif cadence >= 155:
            return {'level': 'fair', 'score': 60, 'description': '一般'}
        else:
            return {'level': 'poor', 'score': 45, 'description': '较差'}

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

    def _calculate_stability_side_view(self, keypoints_sequence: List[Dict]) -> Dict:
        """侧面视角稳定性计算（不包含左右对称性）"""
        trunk_stability = self._calculate_trunk_stability(keypoints_sequence)
        head_stability = self._calculate_head_stability(keypoints_sequence)

        # 侧面视角只评估躯干和头部稳定性
        overall = (trunk_stability * 0.6 + head_stability * 0.4)

        return {
            'overall': float(overall),
            'trunk': float(trunk_stability),
            'head': float(head_stability),
            'rating': self._rate_stability(overall)
        }

    def _calculate_stability_front_view(self, keypoints_sequence: List[Dict]) -> Dict:
        """正面视角稳定性计算（移除对称性，提高肩部晃动权重）"""
        trunk_stability = self._calculate_trunk_stability(keypoints_sequence)
        head_stability = self._calculate_head_stability(keypoints_sequence)
        shoulder_sway = self._calculate_shoulder_sway(keypoints_sequence)

        # 正面视角：提高肩部晃动权重（移除对称性）
        overall = (trunk_stability * 0.35 + head_stability * 0.15 + shoulder_sway * 0.50)

        return {
            'overall': float(overall),
            'trunk': float(trunk_stability),
            'head': float(head_stability),
            'shoulder_sway': float(shoulder_sway),
            'rating': self._rate_stability(overall)
        }

    def _calculate_stability_improved(self, keypoints_sequence: List[Dict]) -> Dict:
        """改进的稳定性计算（兼容用途）"""
        trunk_stability = self._calculate_trunk_stability(keypoints_sequence)
        head_stability = self._calculate_head_stability(keypoints_sequence)
        shoulder_sway = self._calculate_shoulder_sway(keypoints_sequence)

        # 移除对称性，使用肩部晃动
        overall = (trunk_stability * 0.40 + head_stability * 0.20 + shoulder_sway * 0.40)

        return {
            'overall': float(overall),
            'trunk': float(trunk_stability),
            'head': float(head_stability),
            'shoulder_sway': float(shoulder_sway),
            'rating': self._rate_stability(overall)
        }

    def _calculate_shoulder_sway(self, keypoints_sequence: List[Dict]) -> float:
        """计算肩部晃动幅度（用于正面视角对称性评估）"""
        left_shoulder_y = []
        right_shoulder_y = []
        shoulder_diff = []

        for kp in keypoints_sequence:
            left = kp['landmarks'][11]
            right = kp['landmarks'][12]

            if left['visibility'] > 0.5 and right['visibility'] > 0.5:
                left_shoulder_y.append(left['y_norm'])
                right_shoulder_y.append(right['y_norm'])
                # 左右肩膀高度差（正常跑步时应该交替变化）
                shoulder_diff.append(abs(left['y_norm'] - right['y_norm']))

        if len(shoulder_diff) < 10:
            return 50.0

        # 肩部晃动分析
        # 1. 左右高度差的标准差（越小越稳定）
        diff_std = np.std(shoulder_diff) * 100
        # 2. 左右高度差的平均值（过大说明存在不对称）
        diff_mean = np.mean(shoulder_diff) * 100

        # 评分逻辑：差值小且稳定得分高
        score = 100 - min(diff_std * 5 + diff_mean * 3, 100)
        return max(0, score)

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
