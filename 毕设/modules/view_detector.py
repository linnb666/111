# modules/view_detector.py
"""
视频视角自动识别模块
通过分析人体关键点的空间分布来判断拍摄视角
支持：侧面(side)、正面(front)、后方(back)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


class ViewAngleDetector:
    """视频视角检测器"""

    # 视角类型
    VIEW_SIDE = 'side'      # 侧面
    VIEW_FRONT = 'front'    # 正面
    VIEW_BACK = 'back'      # 后方
    VIEW_MIXED = 'mixed'    # 混合视角
    VIEW_UNKNOWN = 'unknown'

    def __init__(self):
        """初始化检测器"""
        # 检测阈值
        self.shoulder_width_threshold = 0.15  # 肩宽阈值（归一化）
        self.hip_width_threshold = 0.12       # 髋宽阈值
        self.depth_variance_threshold = 0.02  # 深度变化阈值

    def detect_view_angle(self, keypoints_sequence: List[Dict],
                          sample_frames: int = 30) -> Dict:
        """
        检测视频的主要拍摄视角

        Args:
            keypoints_sequence: 关键点时间序列
            sample_frames: 采样帧数

        Returns:
            包含视角信息的字典
        """
        # 采样有效帧
        valid_frames = [kp for kp in keypoints_sequence if kp['detected']]
        if len(valid_frames) < 10:
            return self._get_unknown_result("有效帧数不足")

        # 均匀采样
        if len(valid_frames) > sample_frames:
            indices = np.linspace(0, len(valid_frames) - 1, sample_frames, dtype=int)
            sampled_frames = [valid_frames[i] for i in indices]
        else:
            sampled_frames = valid_frames

        # 对每帧进行视角判断
        frame_views = []
        frame_confidences = []

        for frame in sampled_frames:
            view, confidence, features = self._analyze_single_frame(frame)
            frame_views.append(view)
            frame_confidences.append(confidence)

        # 统计视角分布
        view_counts = Counter(frame_views)
        total_frames = len(frame_views)

        # 确定主要视角
        most_common_view, most_common_count = view_counts.most_common(1)[0]
        view_ratio = most_common_count / total_frames

        # 判断是否为混合视角
        if view_ratio < 0.6:
            primary_view = self.VIEW_MIXED
            confidence = view_ratio
        else:
            primary_view = most_common_view
            confidence = np.mean([c for v, c in zip(frame_views, frame_confidences)
                                  if v == most_common_view])

        # 检测视角变化点（用于分段分析）
        view_changes = self._detect_view_changes(frame_views)

        return {
            'primary_view': primary_view,
            'confidence': float(confidence),
            'view_distribution': {
                self.VIEW_SIDE: view_counts.get(self.VIEW_SIDE, 0) / total_frames,
                self.VIEW_FRONT: view_counts.get(self.VIEW_FRONT, 0) / total_frames,
                self.VIEW_BACK: view_counts.get(self.VIEW_BACK, 0) / total_frames,
            },
            'is_mixed': view_ratio < 0.6,
            'view_changes': view_changes,
            'recommendation': self._get_recommendation(primary_view, confidence),
        }

    def _analyze_single_frame(self, keypoints: Dict) -> Tuple[str, float, Dict]:
        """
        分析单帧的视角

        通过以下特征判断：
        1. 肩宽比例 - 侧面时肩膀重叠，宽度小
        2. 髋宽比例 - 侧面时髋部重叠
        3. 左右关键点可见性差异
        4. 深度信息（如果有）
        """
        landmarks = keypoints['landmarks']

        # 提取关键点
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        nose = landmarks[0]
        left_ear = landmarks[7]
        right_ear = landmarks[8]

        features = {}

        # 特征1: 肩部宽度（X方向距离）
        if left_shoulder['visibility'] > 0.5 and right_shoulder['visibility'] > 0.5:
            shoulder_width = abs(left_shoulder['x_norm'] - right_shoulder['x_norm'])
            features['shoulder_width'] = shoulder_width
        else:
            shoulder_width = 0
            features['shoulder_width'] = None

        # 特征2: 髋部宽度
        if left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
            hip_width = abs(left_hip['x_norm'] - right_hip['x_norm'])
            features['hip_width'] = hip_width
        else:
            hip_width = 0
            features['hip_width'] = None

        # 特征3: 耳朵可见性差异（侧面时一边更可见）
        ear_visibility_diff = abs(left_ear['visibility'] - right_ear['visibility'])
        features['ear_visibility_diff'] = ear_visibility_diff

        # 特征4: 肩膀可见性差异
        shoulder_visibility_diff = abs(left_shoulder['visibility'] - right_shoulder['visibility'])
        features['shoulder_visibility_diff'] = shoulder_visibility_diff

        # 特征5: 鼻子与肩部中心的X偏移（正面时接近中心）
        if left_shoulder['visibility'] > 0.5 and right_shoulder['visibility'] > 0.5:
            shoulder_center_x = (left_shoulder['x_norm'] + right_shoulder['x_norm']) / 2
            nose_offset = abs(nose['x_norm'] - shoulder_center_x) if nose['visibility'] > 0.5 else 0
            features['nose_offset'] = nose_offset
        else:
            nose_offset = 0
            features['nose_offset'] = None

        # 特征6: 深度信息（Z坐标差异）
        if hasattr(landmarks[11], 'z') or 'z' in landmarks[11]:
            left_z = landmarks[11].get('z', 0)
            right_z = landmarks[12].get('z', 0)
            z_diff = abs(left_z - right_z) if left_z and right_z else 0
            features['z_diff'] = z_diff
        else:
            z_diff = 0
            features['z_diff'] = None

        # 综合判断视角
        view, confidence = self._determine_view(
            shoulder_width, hip_width, ear_visibility_diff,
            shoulder_visibility_diff, nose_offset, z_diff
        )

        return view, confidence, features

    def _determine_view(self, shoulder_width: float, hip_width: float,
                        ear_diff: float, shoulder_diff: float,
                        nose_offset: float, z_diff: float) -> Tuple[str, float]:
        """
        根据特征确定视角
        """
        scores = {
            self.VIEW_SIDE: 0,
            self.VIEW_FRONT: 0,
            self.VIEW_BACK: 0
        }

        # 规则1: 肩宽 - 侧面时肩宽很小
        if shoulder_width < 0.08:
            scores[self.VIEW_SIDE] += 3
        elif shoulder_width < 0.12:
            scores[self.VIEW_SIDE] += 2
        elif shoulder_width > 0.15:
            scores[self.VIEW_FRONT] += 2
            scores[self.VIEW_BACK] += 2

        # 规则2: 髋宽
        if hip_width < 0.06:
            scores[self.VIEW_SIDE] += 2
        elif hip_width > 0.10:
            scores[self.VIEW_FRONT] += 1
            scores[self.VIEW_BACK] += 1

        # 规则3: 耳朵可见性差异 - 侧面时差异大
        if ear_diff > 0.4:
            scores[self.VIEW_SIDE] += 2
        elif ear_diff < 0.2:
            scores[self.VIEW_FRONT] += 1
            scores[self.VIEW_BACK] += 1

        # 规则4: 肩膀可见性差异
        if shoulder_diff > 0.3:
            scores[self.VIEW_SIDE] += 1
        elif shoulder_diff < 0.15:
            scores[self.VIEW_FRONT] += 1
            scores[self.VIEW_BACK] += 1

        # 规则5: 鼻子偏移（区分正面和后方）
        if nose_offset is not None:
            if nose_offset < 0.05:
                # 鼻子在中心，可能是正面或后方
                # 需要结合其他特征判断
                pass
            else:
                scores[self.VIEW_SIDE] += 1

        # 规则6: Z深度差异
        if z_diff > 0.1:
            scores[self.VIEW_SIDE] += 1

        # 确定最终视角
        max_score = max(scores.values())
        if max_score == 0:
            return self.VIEW_UNKNOWN, 0.0

        # 找到得分最高的视角
        for view, score in scores.items():
            if score == max_score:
                confidence = min(1.0, score / 6.0)  # 归一化置信度
                return view, confidence

        return self.VIEW_UNKNOWN, 0.0

    def _detect_view_changes(self, frame_views: List[str]) -> List[Dict]:
        """
        检测视角变化点

        Returns:
            视角变化点列表
        """
        changes = []
        current_view = frame_views[0]
        segment_start = 0

        for i, view in enumerate(frame_views[1:], 1):
            if view != current_view:
                changes.append({
                    'frame_index': i,
                    'from_view': current_view,
                    'to_view': view,
                    'segment_length': i - segment_start
                })
                current_view = view
                segment_start = i

        return changes

    def _get_recommendation(self, view: str, confidence: float) -> str:
        """
        根据视角给出分析建议
        """
        if view == self.VIEW_SIDE:
            return "侧面视角适合分析：步频、膝关节角度、垂直振幅、躯干前倾、触地/腾空阶段"
        elif view == self.VIEW_FRONT:
            return "正面视角适合分析：下肢力线、膝内扣/外翻、步态对称性、横向稳定性"
        elif view == self.VIEW_BACK:
            return "后方视角适合分析：下肢力线、步态对称性、后足着地模式"
        elif view == self.VIEW_MIXED:
            return "检测到混合视角，建议分段分析或选择主要视角的片段"
        else:
            return "无法确定视角，建议检查视频质量"

    def _get_unknown_result(self, reason: str) -> Dict:
        """返回未知视角结果"""
        return {
            'primary_view': self.VIEW_UNKNOWN,
            'confidence': 0.0,
            'view_distribution': {},
            'is_mixed': False,
            'view_changes': [],
            'recommendation': f"无法检测视角: {reason}",
        }

    def get_analysis_strategy(self, view: str) -> Dict:
        """
        根据视角返回推荐的分析策略

        Args:
            view: 视角类型

        Returns:
            分析策略配置
        """
        strategies = {
            self.VIEW_SIDE: {
                'primary_metrics': [
                    'cadence',           # 步频
                    'knee_angle',        # 膝关节角度
                    'vertical_amplitude', # 垂直振幅
                    'body_lean',         # 躯干前倾
                    'arm_swing',         # 手臂摆动
                    'gait_phase',        # 步态阶段
                ],
                'secondary_metrics': [
                    'stability',         # 稳定性
                    'stride_length',     # 步长
                ],
                'analysis_focus': '侧面动作分析',
                'key_angles': ['knee', 'hip', 'ankle'],
            },
            self.VIEW_FRONT: {
                'primary_metrics': [
                    'lower_limb_alignment',  # 下肢力线
                    'gait_symmetry',         # 步态对称性
                    'lateral_stability',     # 横向稳定性
                    'knee_valgus',           # 膝外翻/内扣
                ],
                'secondary_metrics': [
                    'cadence',
                    'vertical_amplitude',
                ],
                'analysis_focus': '正面力线分析',
                'key_angles': ['frontal_knee', 'hip_drop'],
            },
            self.VIEW_BACK: {
                'primary_metrics': [
                    'lower_limb_alignment',
                    'gait_symmetry',
                    'heel_strike_pattern',   # 后足着地模式
                    'trunk_rotation',        # 躯干旋转
                ],
                'secondary_metrics': [
                    'lateral_stability',
                    'cadence',
                ],
                'analysis_focus': '后方步态分析',
                'key_angles': ['rear_knee', 'ankle_pronation'],
            },
            self.VIEW_MIXED: {
                'primary_metrics': [
                    'cadence',
                    'stability',
                    'vertical_amplitude',
                ],
                'secondary_metrics': [],
                'analysis_focus': '综合分析（混合视角）',
                'key_angles': ['knee', 'hip'],
            },
        }

        return strategies.get(view, strategies[self.VIEW_MIXED])


class AdaptiveAnalyzer:
    """
    自适应分析器
    根据检测到的视角自动选择合适的分析策略
    """

    def __init__(self):
        self.view_detector = ViewAngleDetector()
        self._kinematic_analyzer = None

    @property
    def kinematic_analyzer(self):
        """延迟加载 KinematicAnalyzer"""
        if self._kinematic_analyzer is None:
            from modules.kinematic_analyzer import KinematicAnalyzer
            self._kinematic_analyzer = KinematicAnalyzer()
        return self._kinematic_analyzer

    def analyze(self, keypoints_sequence: List[Dict],
                fps: float,
                view_angle: str = 'side') -> Dict:
        """
        执行运动学分析

        Args:
            keypoints_sequence: 关键点序列
            fps: 帧率
            view_angle: 视角类型 ('side', 'front', 'back', 'mixed')

        Returns:
            分析结果
        """
        # 使用 KinematicAnalyzer 进行分析
        results = self.kinematic_analyzer.analyze_sequence(
            keypoints_sequence, fps, view_angle=view_angle
        )

        # 添加分析策略信息
        results['view_angle'] = view_angle
        results['analysis_strategy'] = self.view_detector.get_analysis_strategy(view_angle)

        return results

    def analyze_with_auto_view(self, keypoints_sequence: List[Dict],
                                fps: float,
                                kinematic_analyzer) -> Dict:
        """
        自动检测视角并执行对应的分析

        Args:
            keypoints_sequence: 关键点序列
            fps: 帧率
            kinematic_analyzer: KinematicAnalyzer实例

        Returns:
            分析结果
        """
        # 检测视角
        view_result = self.view_detector.detect_view_angle(keypoints_sequence)
        primary_view = view_result['primary_view']

        print(f"检测到视角: {primary_view} (置信度: {view_result['confidence']:.2f})")
        print(f"分析建议: {view_result['recommendation']}")

        # 根据视角执行分析
        if view_result['is_mixed']:
            # 混合视角：分段分析
            results = self._analyze_mixed_view(
                keypoints_sequence, fps, view_result, kinematic_analyzer
            )
        else:
            # 单一视角：直接分析
            results = kinematic_analyzer.analyze_sequence(
                keypoints_sequence, fps, view_angle=primary_view
            )

        # 添加视角信息
        results['view_detection'] = view_result
        results['analysis_strategy'] = self.view_detector.get_analysis_strategy(primary_view)

        return results

    def _analyze_mixed_view(self, keypoints_sequence: List[Dict],
                            fps: float,
                            view_result: Dict,
                            kinematic_analyzer) -> Dict:
        """
        处理混合视角的分析
        """
        # 找到主要视角的帧
        view_dist = view_result['view_distribution']
        dominant_view = max(view_dist, key=view_dist.get)

        # 使用主要视角进行分析
        results = kinematic_analyzer.analyze_sequence(
            keypoints_sequence, fps, view_angle=dominant_view
        )

        # 添加混合视角警告
        results['warnings'] = results.get('warnings', [])
        results['warnings'].append(
            f"检测到混合视角，主要使用{dominant_view}视角分析策略"
        )

        return results


# 模块测试
if __name__ == "__main__":
    print("=" * 60)
    print("测试视角检测模块")
    print("=" * 60)

    # 生成模拟侧面视角数据
    print("\n1. 测试侧面视角检测:")
    side_keypoints = []
    for i in range(50):
        kp = {'detected': True, 'landmarks': []}
        for j in range(33):
            # 侧面视角：肩宽小，左右关键点X坐标接近
            if j == 11:  # left_shoulder
                x_norm = 0.5
            elif j == 12:  # right_shoulder
                x_norm = 0.52  # 很接近
            elif j == 23:  # left_hip
                x_norm = 0.5
            elif j == 24:  # right_hip
                x_norm = 0.51
            elif j == 7:  # left_ear
                x_norm = 0.48
            elif j == 8:  # right_ear
                x_norm = 0.55
            else:
                x_norm = 0.5

            kp['landmarks'].append({
                'x_norm': x_norm,
                'y_norm': 0.5,
                'visibility': 0.9 if j in [11, 12, 23, 24] else 0.7
            })
        side_keypoints.append(kp)

    detector = ViewAngleDetector()
    result = detector.detect_view_angle(side_keypoints)
    print(f"检测结果: {result['primary_view']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"建议: {result['recommendation']}")

    # 生成模拟正面视角数据
    print("\n2. 测试正面视角检测:")
    front_keypoints = []
    for i in range(50):
        kp = {'detected': True, 'landmarks': []}
        for j in range(33):
            # 正面视角：肩宽大，左右关键点X坐标差异大
            if j == 11:  # left_shoulder
                x_norm = 0.35
            elif j == 12:  # right_shoulder
                x_norm = 0.65
            elif j == 23:  # left_hip
                x_norm = 0.40
            elif j == 24:  # right_hip
                x_norm = 0.60
            elif j == 7:  # left_ear
                x_norm = 0.42
            elif j == 8:  # right_ear
                x_norm = 0.58
            elif j == 0:  # nose
                x_norm = 0.50
            else:
                x_norm = 0.5

            # 正面时两边可见性相近
            visibility = 0.9 if j in [11, 12, 23, 24, 7, 8, 0] else 0.7

            kp['landmarks'].append({
                'x_norm': x_norm,
                'y_norm': 0.5,
                'visibility': visibility
            })
        front_keypoints.append(kp)

    result = detector.detect_view_angle(front_keypoints)
    print(f"检测结果: {result['primary_view']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"建议: {result['recommendation']}")

    print("\n✅ 视角检测模块测试完成!")
