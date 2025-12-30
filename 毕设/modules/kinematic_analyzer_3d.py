# modules/kinematic_analyzer_3d.py
"""
3D运动学分析模块

核心改进：
1. 基于3D坐标的落地检测（z坐标 + 速度）
2. 3D膝关节角度计算
3. 膝外翻/内扣检测
4. 骨盆运动分析
5. 3D步态对称性
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple, Optional
from config.config import KINEMATIC_CONFIG, KINEMATIC_3D_CONFIG


class KinematicAnalyzer3D:
    """3D运动学分析器"""

    # 步态阶段定义
    PHASE_GROUND_CONTACT = 0  # 触地期
    PHASE_FLIGHT = 1          # 腾空期
    PHASE_TRANSITION = 2      # 过渡期

    # COCO关键点索引
    JOINT_INDICES = {
        'nose': 0,
        'left_shoulder': 5, 'right_shoulder': 6,
        'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10,
        'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14,
        'left_ankle': 15, 'right_ankle': 16
    }

    def __init__(self):
        """初始化分析器"""
        self.config = KINEMATIC_3D_CONFIG
        self.bio_constraints = self.config['biomechanical_constraints']

    def analyze(self, keypoints_3d: np.ndarray, fps: float,
                view_angle: str = 'side') -> Dict:
        """
        分析3D关键点序列

        Args:
            keypoints_3d: 3D关键点序列 (T, 17, 3)
            fps: 帧率
            view_angle: 视角类型

        Returns:
            分析结果字典
        """
        n_frames = keypoints_3d.shape[0]

        if n_frames < 10:
            return self._get_empty_analysis()

        print(f"3D运动学分析：{n_frames}帧，{fps}fps")

        # 平滑处理
        keypoints_3d = self._smooth_keypoints(keypoints_3d)

        # 计算躯干参考长度（3D）
        trunk_length = self._calculate_trunk_length_3d(keypoints_3d)
        print(f"躯干参考长度: {trunk_length:.4f}")

        results = {
            'fps': fps,
            'total_frames': n_frames,
            'view_angle': view_angle,
            'trunk_reference': trunk_length,
            'is_3d': True,
        }

        # 核心分析
        results['landing_events'] = self._detect_landing_3d(keypoints_3d, fps)
        results['angles'] = self._analyze_joint_angles_3d(keypoints_3d, fps, results['landing_events'])
        results['vertical_motion'] = self._analyze_vertical_motion_3d(keypoints_3d, fps, trunk_length)
        results['cadence'] = self._calculate_cadence(results['landing_events'], fps, n_frames)
        results['gait_cycle'] = self._analyze_gait_cycle_3d(keypoints_3d, fps, results['landing_events'])

        # 3D特有分析
        if self.config['knee_valgus']['enabled']:
            results['knee_valgus'] = self._analyze_knee_valgus_3d(keypoints_3d, fps, results['landing_events'])

        if self.config['pelvic_motion']['enabled']:
            results['pelvic_motion'] = self._analyze_pelvic_motion_3d(keypoints_3d, fps, trunk_length)

        if self.config['symmetry']['enabled']:
            results['symmetry'] = self._analyze_gait_symmetry_3d(keypoints_3d, fps, results['landing_events'])

        # 稳定性分析
        results['stability'] = self._analyze_stability_3d(keypoints_3d, fps)

        # 躯干前倾
        results['body_lean'] = self._analyze_body_lean_3d(keypoints_3d)

        return results

    def _smooth_keypoints(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """平滑3D关键点"""
        n_frames = keypoints_3d.shape[0]
        cfg = self.config['signal_processing']
        window = cfg.get('smooth_window', 7)
        polyorder = cfg.get('smooth_polyorder', 2)

        if n_frames < window:
            return keypoints_3d

        if window % 2 == 0:
            window -= 1

        smoothed = np.zeros_like(keypoints_3d)

        for joint in range(17):
            for dim in range(3):
                smoothed[:, joint, dim] = savgol_filter(
                    keypoints_3d[:, joint, dim], window, polyorder
                )

        return smoothed

    def _calculate_trunk_length_3d(self, keypoints_3d: np.ndarray) -> float:
        """计算躯干3D长度（肩到髋的距离）"""
        left_shoulder = keypoints_3d[:, 5]
        right_shoulder = keypoints_3d[:, 6]
        left_hip = keypoints_3d[:, 11]
        right_hip = keypoints_3d[:, 12]

        # 计算每帧的躯干长度
        trunk_lengths = []

        for t in range(len(keypoints_3d)):
            # 左侧躯干
            left_trunk = np.linalg.norm(left_shoulder[t] - left_hip[t])
            right_trunk = np.linalg.norm(right_shoulder[t] - right_hip[t])

            if left_trunk > 0:
                trunk_lengths.append(left_trunk)
            if right_trunk > 0:
                trunk_lengths.append(right_trunk)

        if trunk_lengths:
            return np.median(trunk_lengths)
        return 0.5  # 默认值

    # ==================== 3D落地检测 ====================

    def _detect_landing_3d(self, keypoints_3d: np.ndarray, fps: float) -> List[Dict]:
        """
        3D落地检测

        使用脚踝的z坐标（垂直方向）和速度进行检测
        z值小 = 位置低 = 接近地面
        """
        cfg = self.config['landing_detection']
        n_frames = keypoints_3d.shape[0]

        if n_frames < 15:
            return []

        # 提取左右脚踝z坐标
        left_ankle_z = keypoints_3d[:, 15, 2]
        right_ankle_z = keypoints_3d[:, 16, 2]

        # 计算速度
        left_velocity = np.gradient(left_ankle_z) * fps
        right_velocity = np.gradient(right_ankle_z) * fps

        # 自适应阈值
        z_percentile = cfg.get('z_threshold_percentile', 15)
        ground_level = np.percentile(
            np.minimum(left_ankle_z, right_ankle_z),
            z_percentile
        )

        # 检测左脚落地
        left_landings = self._detect_foot_landing_3d(
            left_ankle_z, left_velocity, ground_level, fps, 'left'
        )

        # 检测右脚落地
        right_landings = self._detect_foot_landing_3d(
            right_ankle_z, right_velocity, ground_level, fps, 'right'
        )

        # 合并并排序
        all_landings = left_landings + right_landings
        all_landings.sort(key=lambda x: x['frame'])

        # 周期性验证
        validated = self._validate_landing_periodicity(all_landings, fps)

        print(f"  3D落地检测：左脚{len(left_landings)}次，右脚{len(right_landings)}次")
        print(f"  验证后：{len(validated)}次有效落地")

        return validated

    def _detect_foot_landing_3d(self, ankle_z: np.ndarray, velocity: np.ndarray,
                                 ground_level: float, fps: float, foot: str) -> List[Dict]:
        """检测单脚3D落地"""
        n = len(ankle_z)
        cfg = self.config['landing_detection']

        # z值最小的位置就是落地点
        # 寻找z的局部最小值
        min_distance = max(3, int(fps * 0.25))  # 至少250ms间隔

        # 反转z轴（因为z小=落地），寻找峰值
        inverted_z = -ankle_z
        peaks, properties = find_peaks(
            inverted_z,
            distance=min_distance,
            prominence=0.02  # 至少2cm的变化
        )

        landings = []

        for peak in peaks:
            # 检查是否接近地面
            if ankle_z[peak] <= ground_level + 0.05:  # 容差5cm
                # 检查速度是否接近零（落地瞬间）
                if abs(velocity[peak]) < cfg.get('velocity_threshold', 0.5):
                    # 计算落地区间
                    start, end = self._find_landing_window(ankle_z, peak, ground_level)
                    duration_ms = (end - start + 1) * 1000.0 / fps

                    if cfg['min_landing_duration_ms'] <= duration_ms <= cfg['max_landing_duration_ms']:
                        landings.append({
                            'frame': int(peak),
                            'foot': foot,
                            'z_position': float(ankle_z[peak]),
                            'z_velocity': float(velocity[peak]),
                            'start_frame': int(start),
                            'end_frame': int(end),
                            'duration_ms': float(duration_ms)
                        })

        return landings

    def _find_landing_window(self, ankle_z: np.ndarray, peak: int, ground_level: float) -> Tuple[int, int]:
        """找到落地区间的起止帧"""
        n = len(ankle_z)
        threshold = ground_level + 0.03  # 3cm阈值

        # 向前找起点
        start = peak
        for i in range(peak - 1, max(0, peak - 20) - 1, -1):
            if ankle_z[i] > threshold:
                start = i + 1
                break

        # 向后找终点
        end = peak
        for i in range(peak + 1, min(n, peak + 20)):
            if ankle_z[i] > threshold:
                end = i - 1
                break

        return start, end

    def _validate_landing_periodicity(self, landings: List[Dict], fps: float) -> List[Dict]:
        """验证落地周期性"""
        if len(landings) < 3:
            return landings

        # 计算相邻间隔
        intervals = []
        for i in range(1, len(landings)):
            interval = landings[i]['frame'] - landings[i-1]['frame']
            intervals.append(interval)

        intervals = np.array(intervals)

        # IQR过滤
        q25, q75 = np.percentile(intervals, [25, 75])
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr

        # 物理约束：步频120-220步/分
        min_interval = fps * 60 / 220
        max_interval = fps * 60 / 120

        lower = max(lower, min_interval)
        upper = min(upper, max_interval)

        # 保留有效落地
        valid = [landings[0]]
        for i in range(1, len(landings)):
            interval = landings[i]['frame'] - landings[i-1]['frame']
            if lower <= interval <= upper:
                valid.append(landings[i])

        return valid

    # ==================== 3D关节角度分析 ====================

    def _analyze_joint_angles_3d(self, keypoints_3d: np.ndarray, fps: float,
                                  landing_events: List[Dict]) -> Dict:
        """分析3D关节角度"""
        n_frames = keypoints_3d.shape[0]

        # 计算每帧的膝关节角度
        knee_left = []
        knee_right = []

        for t in range(n_frames):
            # 左膝角度（髋-膝-踝）
            left_angle = self._calculate_3d_joint_angle(
                keypoints_3d[t, 11],  # left_hip
                keypoints_3d[t, 13],  # left_knee
                keypoints_3d[t, 15]   # left_ankle
            )
            knee_left.append(left_angle)

            # 右膝角度
            right_angle = self._calculate_3d_joint_angle(
                keypoints_3d[t, 12],  # right_hip
                keypoints_3d[t, 14],  # right_knee
                keypoints_3d[t, 16]   # right_ankle
            )
            knee_right.append(right_angle)

        knee_left = np.array(knee_left)
        knee_right = np.array(knee_right)

        # 计算落地膝角（使用生物力学约束）
        landing_angles = self._calculate_landing_knee_angles_3d(
            knee_left, knee_right, fps, landing_events
        )

        # 分阶段统计
        phases = self._detect_gait_phases_3d(keypoints_3d, fps)
        phase_analysis = self._analyze_angles_by_phase(knee_left, knee_right, phases)

        # 合并落地膝角结果
        if landing_angles['landing_count'] > 0:
            phase_analysis['ground_contact']['landing_angle_mean'] = landing_angles['landing_angle_mean']
            phase_analysis['ground_contact']['landing_angle_std'] = landing_angles['landing_angle_std']
            phase_analysis['ground_contact']['landing_count'] = landing_angles['landing_count']
            phase_analysis['ground_contact']['per_step_stats'] = landing_angles['per_step_stats']
            phase_analysis['ground_contact']['rejected_steps'] = landing_angles.get('rejected_steps', [])

        return {
            'knee_left': knee_left.tolist(),
            'knee_right': knee_right.tolist(),
            'knee_left_mean': float(np.nanmean(knee_left)),
            'knee_right_mean': float(np.nanmean(knee_right)),
            'knee_left_std': float(np.nanstd(knee_left)),
            'knee_right_std': float(np.nanstd(knee_right)),
            'knee_rom': float(np.nanmax(knee_left) - np.nanmin(knee_left)),
            'phase_analysis': phase_analysis,
            'is_3d': True
        }

    def _calculate_3d_joint_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        计算三点形成的3D角度

        Args:
            p1, p2, p3: 三个3D点坐标，p2为角度顶点

        Returns:
            角度（度）
        """
        v1 = p1 - p2
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return np.nan

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def _calculate_landing_knee_angles_3d(self, knee_left: np.ndarray, knee_right: np.ndarray,
                                          fps: float, landing_events: List[Dict]) -> Dict:
        """计算落地膝角（带生物力学约束）"""
        MIN_ANGLE = self.bio_constraints['landing_knee_angle_min']
        MAX_ANGLE = self.bio_constraints['landing_knee_angle_max']
        EXT_THRESHOLD = self.bio_constraints['extension_rate_threshold']

        # 计算变化率
        knee_left_rate = np.gradient(knee_left)
        knee_right_rate = np.gradient(knee_right)

        valid_landings = []
        rejected_landings = []

        pre_landing_frames = max(2, int(100 * fps / 1000))  # 100ms窗口

        for event in landing_events:
            peak = event['frame']
            foot = event['foot']

            # 选择对应脚的膝角
            if foot == 'left':
                angles = knee_left
                rates = knee_left_rate
            else:
                angles = knee_right
                rates = knee_right_rate

            # 搜索窗口
            search_start = max(0, peak - pre_landing_frames)
            search_end = peak

            if search_start >= search_end:
                rejected_landings.append({
                    'foot': foot,
                    'frame': peak,
                    'rejection_reason': 'window_too_small'
                })
                continue

            window_angles = angles[search_start:search_end]
            window_rates = rates[search_start:search_end]

            # 过滤NaN
            valid_mask = ~np.isnan(window_angles)
            if not np.any(valid_mask):
                rejected_landings.append({
                    'foot': foot,
                    'frame': peak,
                    'rejection_reason': 'no_valid_frames'
                })
                continue

            window_angles = window_angles[valid_mask]
            window_rates = window_rates[valid_mask]

            # 生物力学约束
            angle_valid = (window_angles >= MIN_ANGLE) & (window_angles <= MAX_ANGLE)
            extension_valid = window_rates >= EXT_THRESHOLD

            both_valid = angle_valid & extension_valid

            if not np.any(both_valid):
                max_angle = float(np.max(window_angles))
                mean_rate = float(np.mean(window_rates))

                if max_angle < MIN_ANGLE:
                    reason = 'angle_too_low'
                elif max_angle > MAX_ANGLE:
                    reason = 'angle_too_high'
                else:
                    reason = 'flexion_trend'

                rejected_landings.append({
                    'foot': foot,
                    'frame': peak,
                    'rejection_reason': reason,
                    'actual_angle': max_angle,
                    'actual_rate': mean_rate
                })
                continue

            # 在合法帧中选取最大角度
            valid_angles = window_angles[both_valid]
            best_angle = float(np.max(valid_angles))

            valid_landings.append({
                'foot': foot,
                'frame': int(peak),
                'landing_angle': best_angle,
                'duration_ms': event['duration_ms']
            })

        # 统计
        if valid_landings:
            angles = [v['landing_angle'] for v in valid_landings]
            return {
                'landing_angle_mean': float(np.mean(angles)),
                'landing_angle_std': float(np.std(angles)) if len(angles) > 1 else 0.0,
                'landing_count': len(valid_landings),
                'per_step_stats': valid_landings,
                'rejected_steps': rejected_landings,
                'method': '3d_biomechanical_constrained'
            }

        return {
            'landing_angle_mean': 0,
            'landing_angle_std': 0,
            'landing_count': 0,
            'per_step_stats': [],
            'rejected_steps': rejected_landings,
            'method': '3d_biomechanical_constrained'
        }

    def _detect_gait_phases_3d(self, keypoints_3d: np.ndarray, fps: float) -> List[int]:
        """使用3D坐标检测步态阶段"""
        n_frames = keypoints_3d.shape[0]
        if n_frames < 3:
            return [self.PHASE_TRANSITION] * n_frames

        # 使用脚踝z坐标判断
        left_ankle_z = keypoints_3d[:, 15, 2]
        right_ankle_z = keypoints_3d[:, 16, 2]

        # 取较低的脚（正在触地的脚）
        lower_z = np.minimum(left_ankle_z, right_ankle_z)

        # 计算速度
        velocity = np.abs(np.gradient(lower_z) * fps)

        # 自适应阈值
        z_max = np.max(lower_z)
        z_min = np.min(lower_z)
        z_range = z_max - z_min

        ground_threshold = z_min + z_range * 0.25
        flight_threshold = z_max - z_range * 0.35

        phases = []
        for i in range(n_frames):
            z = lower_z[i]
            v = velocity[i] if i < len(velocity) else 0

            if z <= ground_threshold and v < 0.5:
                phases.append(self.PHASE_GROUND_CONTACT)
            elif z >= flight_threshold:
                phases.append(self.PHASE_FLIGHT)
            else:
                phases.append(self.PHASE_TRANSITION)

        return phases

    def _analyze_angles_by_phase(self, knee_left: np.ndarray, knee_right: np.ndarray,
                                  phases: List[int]) -> Dict:
        """按阶段统计角度"""
        gc_angles, fl_angles, tr_angles = [], [], []

        for i, phase in enumerate(phases):
            if i < len(knee_left):
                avg = (knee_left[i] + knee_right[i]) / 2
                if not np.isnan(avg):
                    if phase == self.PHASE_GROUND_CONTACT:
                        gc_angles.append(avg)
                    elif phase == self.PHASE_FLIGHT:
                        fl_angles.append(avg)
                    else:
                        tr_angles.append(avg)

        def stats(angles):
            if angles:
                return {
                    'mean': float(np.mean(angles)),
                    'std': float(np.std(angles)),
                    'min': float(np.min(angles)),
                    'max': float(np.max(angles)),
                    'count': len(angles)
                }
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}

        return {
            'ground_contact': stats(gc_angles),
            'flight': stats(fl_angles),
            'transition': stats(tr_angles)
        }

    # ==================== 3D垂直振幅分析 ====================

    def _analyze_vertical_motion_3d(self, keypoints_3d: np.ndarray, fps: float,
                                     trunk_length: float) -> Dict:
        """3D垂直运动分析"""
        # 骨盆中心z坐标
        pelvis_z = (keypoints_3d[:, 11, 2] + keypoints_3d[:, 12, 2]) / 2

        # 平滑
        if len(pelvis_z) > 7:
            pelvis_z_smooth = savgol_filter(pelvis_z, 7, 2)
        else:
            pelvis_z_smooth = pelvis_z

        # 振幅
        amplitude = np.max(pelvis_z_smooth) - np.min(pelvis_z_smooth)
        amplitude_normalized = (amplitude / trunk_length) * 100 if trunk_length > 0 else 0

        # 频率
        peaks, _ = find_peaks(pelvis_z_smooth, distance=int(fps * 0.2))
        if len(peaks) > 1:
            time_span = len(pelvis_z) / fps
            frequency = len(peaks) / time_span
        else:
            frequency = 0

        return {
            'amplitude': float(amplitude),
            'amplitude_normalized': float(amplitude_normalized),
            'frequency': float(frequency),
            'amplitude_rating': self._rate_vertical_amplitude(amplitude_normalized),
            'is_3d': True
        }

    def _rate_vertical_amplitude(self, amp_normalized: float) -> Dict:
        """评估垂直振幅"""
        if 3 <= amp_normalized <= 6:
            return {'level': 'excellent', 'score': 100, 'description': '垂直振幅非常理想'}
        elif 6 < amp_normalized <= 10:
            return {'level': 'good', 'score': 80, 'description': '垂直振幅良好'}
        elif 10 < amp_normalized <= 15:
            return {'level': 'fair', 'score': 60, 'description': '垂直振幅偏大'}
        elif amp_normalized < 3:
            return {'level': 'low', 'score': 70, 'description': '垂直振幅偏小'}
        else:
            return {'level': 'poor', 'score': 40, 'description': '垂直振幅过大'}

    # ==================== 步频计算 ====================

    def _calculate_cadence(self, landing_events: List[Dict], fps: float, n_frames: int) -> Dict:
        """计算步频"""
        if len(landing_events) < 2:
            return {'cadence': 0, 'step_count': 0, 'duration': 0, 'rating': {}}

        first = landing_events[0]['frame']
        last = landing_events[-1]['frame']
        duration_sec = (last - first) / fps

        if duration_sec <= 0:
            return {'cadence': 0, 'step_count': 0, 'duration': 0, 'rating': {}}

        step_count = len(landing_events)
        cadence = (step_count / duration_sec) * 60

        # 评级
        if cadence >= 185:
            rating = {'level': 'elite', 'score': 100, 'description': '精英'}
        elif cadence >= 175:
            rating = {'level': 'excellent', 'score': 90, 'description': '优秀'}
        elif cadence >= 165:
            rating = {'level': 'good', 'score': 75, 'description': '良好'}
        elif cadence >= 155:
            rating = {'level': 'fair', 'score': 60, 'description': '一般'}
        else:
            rating = {'level': 'poor', 'score': 45, 'description': '较差'}

        return {
            'cadence': float(cadence),
            'step_count': step_count,
            'duration': float(duration_sec),
            'rating': rating
        }

    # ==================== 步态周期分析 ====================

    def _analyze_gait_cycle_3d(self, keypoints_3d: np.ndarray, fps: float,
                                landing_events: List[Dict]) -> Dict:
        """3D步态周期分析"""
        if len(landing_events) < 2:
            return {
                'phase_distribution': {'ground_contact': 0.45, 'flight': 0.35, 'transition': 0.20},
                'phase_duration_ms': {'ground_contact': 0, 'flight': 0},
                'avg_cycle_duration_ms': 0
            }

        # 触地时间统计
        gc_times = [e['duration_ms'] for e in landing_events if 120 <= e['duration_ms'] <= 400]

        # 腾空时间
        flight_times = []
        for i in range(len(landing_events) - 1):
            end_frame = landing_events[i]['end_frame']
            start_frame = landing_events[i+1]['start_frame']
            flight_ms = (start_frame - end_frame) * 1000.0 / fps
            if 40 <= flight_ms <= 350:
                flight_times.append(flight_ms)

        avg_gc = float(np.mean(gc_times)) if gc_times else 0
        avg_flight = float(np.mean(flight_times)) if flight_times else 0
        avg_cycle = avg_gc + avg_flight

        # 比例
        if avg_cycle > 0:
            gc_ratio = avg_gc / avg_cycle
            fl_ratio = avg_flight / avg_cycle
        else:
            gc_ratio, fl_ratio = 0.45, 0.35

        return {
            'phase_distribution': {
                'ground_contact': float(round(gc_ratio, 3)),
                'flight': float(round(fl_ratio, 3)),
                'transition': float(round(1 - gc_ratio - fl_ratio, 3))
            },
            'phase_duration_ms': {
                'ground_contact': float(round(avg_gc, 1)),
                'flight': float(round(avg_flight, 1))
            },
            'avg_cycle_duration_ms': float(round(avg_cycle, 1))
        }

    # ==================== 膝外翻/内扣分析 ====================

    def _analyze_knee_valgus_3d(self, keypoints_3d: np.ndarray, fps: float,
                                 landing_events: List[Dict]) -> Dict:
        """3D膝外翻/内扣分析"""
        cfg = self.config['knee_valgus']

        left_valgus_angles = []
        right_valgus_angles = []

        # 在落地时刻分析
        for event in landing_events:
            frame = event['frame']

            # 左腿
            left_angle = self._calculate_frontal_knee_angle(keypoints_3d[frame], 'left')
            if not np.isnan(left_angle):
                left_valgus_angles.append(left_angle)

            # 右腿
            right_angle = self._calculate_frontal_knee_angle(keypoints_3d[frame], 'right')
            if not np.isnan(right_angle):
                right_valgus_angles.append(right_angle)

        def analyze_side(angles):
            if not angles:
                return {'mean': 0, 'max': 0, 'severity': 'unknown'}

            mean_angle = float(np.mean(angles))
            max_angle = float(np.max(np.abs(angles)))

            if max_angle < cfg['normal_range'][1]:
                severity = 'normal'
            elif max_angle < cfg['mild_range'][1]:
                severity = 'mild'
            elif max_angle < cfg['moderate_range'][1]:
                severity = 'moderate'
            else:
                severity = 'severe'

            direction = 'valgus' if mean_angle > 0 else 'varus' if mean_angle < 0 else 'neutral'

            return {
                'mean': mean_angle,
                'max': max_angle,
                'direction': direction,
                'severity': severity
            }

        return {
            'left': analyze_side(left_valgus_angles),
            'right': analyze_side(right_valgus_angles),
            'is_3d': True
        }

    def _calculate_frontal_knee_angle(self, keypoints: np.ndarray, side: str) -> float:
        """计算冠状面膝关节偏移角度"""
        if side == 'left':
            hip = keypoints[11]
            knee = keypoints[13]
            ankle = keypoints[15]
        else:
            hip = keypoints[12]
            knee = keypoints[14]
            ankle = keypoints[16]

        # 投影到冠状面（XZ平面）
        hip_xz = np.array([hip[0], hip[2]])
        knee_xz = np.array([knee[0], knee[2]])
        ankle_xz = np.array([ankle[0], ankle[2]])

        # 理想直线：髋-踝连线
        ideal_vec = ankle_xz - hip_xz

        # 实际位置：膝关节相对于理想线的偏移
        hip_to_knee = knee_xz - hip_xz

        if np.linalg.norm(ideal_vec) < 1e-6:
            return np.nan

        # 计算偏移角度
        # 正值 = 外翻（膝盖向内）
        # 负值 = 内扣（膝盖向外）
        cross = np.cross(ideal_vec, hip_to_knee)
        dot = np.dot(ideal_vec, hip_to_knee)
        angle = np.arctan2(cross, dot)

        return np.degrees(angle)

    # ==================== 骨盆运动分析 ====================

    def _analyze_pelvic_motion_3d(self, keypoints_3d: np.ndarray, fps: float,
                                   trunk_length: float) -> Dict:
        """3D骨盆运动分析"""
        left_hip = keypoints_3d[:, 11]
        right_hip = keypoints_3d[:, 12]

        # 骨盆中心
        pelvis_center = (left_hip + right_hip) / 2

        # 垂直位移（Z轴）
        vertical_range = np.max(pelvis_center[:, 2]) - np.min(pelvis_center[:, 2])
        vertical_normalized = (vertical_range / trunk_length) * 100 if trunk_length > 0 else 0

        # 横向摆动（X轴）
        lateral_range = np.max(pelvis_center[:, 0]) - np.min(pelvis_center[:, 0])
        lateral_normalized = (lateral_range / trunk_length) * 100 if trunk_length > 0 else 0

        # 前后位移（Y轴）
        ap_range = np.max(pelvis_center[:, 1]) - np.min(pelvis_center[:, 1])

        # 骨盆倾斜角度
        tilt_angles = []
        for t in range(len(keypoints_3d)):
            # 左右髋关节的z差异
            z_diff = left_hip[t, 2] - right_hip[t, 2]
            hip_width = np.linalg.norm(left_hip[t, :2] - right_hip[t, :2])
            if hip_width > 0:
                tilt = np.degrees(np.arctan2(z_diff, hip_width))
                tilt_angles.append(tilt)

        mean_tilt = float(np.mean(tilt_angles)) if tilt_angles else 0
        tilt_range = float(np.max(tilt_angles) - np.min(tilt_angles)) if tilt_angles else 0

        return {
            'vertical_displacement': float(vertical_normalized),
            'lateral_sway': float(lateral_normalized),
            'ap_displacement': float(ap_range),
            'tilt_mean': mean_tilt,
            'tilt_range': tilt_range,
            'stability_score': self._rate_pelvic_stability(vertical_normalized, lateral_normalized),
            'is_3d': True
        }

    def _rate_pelvic_stability(self, vertical: float, lateral: float) -> float:
        """评估骨盆稳定性"""
        cfg = self.config['pelvic_motion']

        v_score = 100 if vertical <= cfg['vertical_excellent_max'] * 100 else max(50, 100 - vertical * 2)
        l_score = 100 if lateral <= cfg['lateral_excellent_max'] * 100 else max(50, 100 - lateral * 2)

        return (v_score + l_score) / 2

    # ==================== 步态对称性分析 ====================

    def _analyze_gait_symmetry_3d(self, keypoints_3d: np.ndarray, fps: float,
                                   landing_events: List[Dict]) -> Dict:
        """3D步态对称性分析"""
        cfg = self.config['symmetry']

        # 按脚分类落地事件
        left_events = [e for e in landing_events if e['foot'] == 'left']
        right_events = [e for e in landing_events if e['foot'] == 'right']

        # 触地时间对称性
        left_durations = [e['duration_ms'] for e in left_events]
        right_durations = [e['duration_ms'] for e in right_events]

        if left_durations and right_durations:
            left_mean = np.mean(left_durations)
            right_mean = np.mean(right_durations)
            gc_symmetry = (1 - abs(left_mean - right_mean) / max(left_mean, right_mean)) * 100
        else:
            gc_symmetry = 100

        # 步长对称性（使用脚踝位置估算）
        step_lengths_left = []
        step_lengths_right = []

        for i in range(1, len(landing_events)):
            curr = landing_events[i]
            prev = landing_events[i-1]

            if curr['foot'] != prev['foot']:
                # 计算步长
                curr_pos = keypoints_3d[curr['frame'], 15 if curr['foot'] == 'left' else 16]
                prev_pos = keypoints_3d[prev['frame'], 15 if prev['foot'] == 'left' else 16]
                step_length = np.linalg.norm(curr_pos - prev_pos)

                if curr['foot'] == 'left':
                    step_lengths_left.append(step_length)
                else:
                    step_lengths_right.append(step_length)

        if step_lengths_left and step_lengths_right:
            left_step_mean = np.mean(step_lengths_left)
            right_step_mean = np.mean(step_lengths_right)
            step_symmetry = (1 - abs(left_step_mean - right_step_mean) / max(left_step_mean, right_step_mean)) * 100
        else:
            step_symmetry = 100

        # 总体对称性
        overall = (gc_symmetry + step_symmetry) / 2

        # 确定主导侧
        if left_durations and right_durations:
            if np.mean(left_durations) < np.mean(right_durations):
                dominant = 'left'
            elif np.mean(right_durations) < np.mean(left_durations):
                dominant = 'right'
            else:
                dominant = 'balanced'
        else:
            dominant = 'unknown'

        return {
            'ground_contact_symmetry': float(gc_symmetry),
            'step_length_symmetry': float(step_symmetry),
            'overall_symmetry': float(overall),
            'dominant_side': dominant,
            'rating': self._rate_symmetry(overall, cfg),
            'is_3d': True
        }

    def _rate_symmetry(self, symmetry: float, cfg: Dict) -> str:
        """评估对称性等级"""
        if symmetry >= cfg['excellent_threshold']:
            return 'excellent'
        elif symmetry >= cfg['good_threshold']:
            return 'good'
        elif symmetry >= cfg['fair_threshold']:
            return 'fair'
        else:
            return 'poor'

    # ==================== 稳定性分析 ====================

    def _analyze_stability_3d(self, keypoints_3d: np.ndarray, fps: float) -> Dict:
        """3D稳定性分析"""
        # 头部稳定性（鼻子位置变化）
        nose = keypoints_3d[:, 0]
        head_stability = 100 - np.std(nose[:, 2]) * 100  # z轴变化

        # 躯干稳定性（肩膀中心变化）
        shoulder_center = (keypoints_3d[:, 5] + keypoints_3d[:, 6]) / 2
        trunk_stability = 100 - np.std(shoulder_center[:, 2]) * 100

        # 综合稳定性
        overall = (head_stability * 0.4 + trunk_stability * 0.6)
        overall = max(0, min(100, overall))

        return {
            'head_stability': float(max(0, head_stability)),
            'trunk_stability': float(max(0, trunk_stability)),
            'overall': float(overall),
            'is_3d': True
        }

    # ==================== 躯干前倾分析 ====================

    def _analyze_body_lean_3d(self, keypoints_3d: np.ndarray) -> Dict:
        """3D躯干前倾分析"""
        # 使用肩膀和髋部的位置计算前倾角度
        shoulders = (keypoints_3d[:, 5] + keypoints_3d[:, 6]) / 2
        hips = (keypoints_3d[:, 11] + keypoints_3d[:, 12]) / 2

        lean_angles = []
        for t in range(len(keypoints_3d)):
            # 躯干向量
            trunk_vec = shoulders[t] - hips[t]

            # 垂直向量（Z轴）
            vertical = np.array([0, 0, 1])

            # 计算角度
            cos_angle = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            # 前倾角度 = 90 - 与垂直的夹角
            lean = 90 - angle
            lean_angles.append(lean)

        mean_lean = float(np.mean(lean_angles))

        # 评级
        if 5 <= mean_lean <= 15:
            rating = {'level': 'optimal', 'score': 100, 'description': '前倾角度理想'}
        elif 3 <= mean_lean < 5 or 15 < mean_lean <= 20:
            rating = {'level': 'acceptable', 'score': 75, 'description': '前倾角度可接受'}
        else:
            rating = {'level': 'needs_work', 'score': 50, 'description': '前倾角度需调整'}

        return {
            'forward_lean': mean_lean,
            'lean_std': float(np.std(lean_angles)),
            'rating': rating,
            'is_3d': True
        }

    def _get_empty_analysis(self) -> Dict:
        """返回空分析结果"""
        return {
            'fps': 0,
            'total_frames': 0,
            'view_angle': 'unknown',
            'trunk_reference': 0,
            'is_3d': True,
            'error': '帧数不足，无法分析'
        }


# 兼容性包装器
class KinematicAnalyzer3DWrapper:
    """
    3D分析器包装器

    提供与原有KinematicAnalyzer兼容的接口
    """

    def __init__(self):
        self.analyzer_3d = KinematicAnalyzer3D()

    def analyze_sequence(self, keypoints_sequence: List[Dict], fps: float,
                         view_angle: str = 'side') -> Dict:
        """
        兼容接口：分析关键点序列

        自动检测是否包含3D数据，选择相应的分析方法
        """
        # 检查是否有3D数据
        has_3d = False
        if keypoints_sequence and 'keypoints_3d' in keypoints_sequence[0]:
            has_3d = True

        if has_3d:
            # 提取3D关键点
            keypoints_3d = np.array([kp['keypoints_3d'] for kp in keypoints_sequence])
            return self.analyzer_3d.analyze(keypoints_3d, fps, view_angle)
        else:
            # 回退到2D分析（导入原有模块）
            from modules.kinematic_analyzer import KinematicAnalyzer
            analyzer_2d = KinematicAnalyzer()
            return analyzer_2d.analyze_sequence(keypoints_sequence, fps, view_angle)


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试3D运动学分析模块")
    print("=" * 60)

    # 创建模拟3D数据
    n_frames = 100
    fps = 30.0

    # 模拟跑步动作
    t = np.linspace(0, 2 * np.pi, n_frames)
    keypoints_3d = np.zeros((n_frames, 17, 3))

    # 设置基础位置
    for i in range(n_frames):
        # 髋部
        keypoints_3d[i, 11] = [0.1, 0, 0.9 + 0.02 * np.sin(t[i] * 2)]
        keypoints_3d[i, 12] = [-0.1, 0, 0.9 + 0.02 * np.sin(t[i] * 2)]

        # 膝盖
        keypoints_3d[i, 13] = [0.1, 0.05, 0.5 + 0.05 * np.sin(t[i] * 2)]
        keypoints_3d[i, 14] = [-0.1, 0.05, 0.5 + 0.05 * np.sin(t[i] * 2 + np.pi)]

        # 脚踝
        keypoints_3d[i, 15] = [0.1, 0, 0.05 + 0.05 * np.abs(np.sin(t[i] * 2))]
        keypoints_3d[i, 16] = [-0.1, 0, 0.05 + 0.05 * np.abs(np.sin(t[i] * 2 + np.pi))]

        # 肩膀
        keypoints_3d[i, 5] = [0.15, 0, 1.5]
        keypoints_3d[i, 6] = [-0.15, 0, 1.5]

        # 鼻子
        keypoints_3d[i, 0] = [0, 0, 1.7]

    # 分析
    analyzer = KinematicAnalyzer3D()
    results = analyzer.analyze(keypoints_3d, fps)

    print(f"\n分析结果：")
    print(f"  落地事件: {len(results.get('landing_events', []))}次")
    print(f"  步频: {results.get('cadence', {}).get('cadence', 0):.1f}步/分")
    print(f"  垂直振幅: {results.get('vertical_motion', {}).get('amplitude_normalized', 0):.1f}%")

    if 'angles' in results:
        phase = results['angles'].get('phase_analysis', {})
        gc = phase.get('ground_contact', {})
        print(f"  落地膝角: {gc.get('landing_angle_mean', 0):.1f}°")

    print("\n测试完成!")
