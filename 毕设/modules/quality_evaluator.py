# modules/quality_evaluator.py
"""
重构版技术质量评价模块
核心改进：
1. 适配新的归一化振幅指标
2. 支持分阶段膝关节角度评估
3. 针对不同视角的评价策略
4. 更专业的评分逻辑和解释性输出
"""
import numpy as np
from typing import Dict, List, Optional
from config.config import QUALITY_WEIGHTS, QUALITY_THRESHOLDS


class QualityEvaluator:
    """重构版技术质量评价器"""

    def __init__(self):
        """初始化评价器"""
        self.weights = QUALITY_WEIGHTS
        self.thresholds = QUALITY_THRESHOLDS

        # 专业评价标准（基于运动科学研究）
        self.standards = {
            # 垂直振幅标准（相对躯干长度的百分比）
            'vertical_amplitude': {
                'excellent': (3, 6),    # 3-6%
                'good': (6, 10),        # 6-10%
                'fair': (10, 15),       # 10-15%
                'poor_high': (15, 100), # >15%
                'poor_low': (0, 3),     # <3%
            },
            # 步频标准（保留用户原有设定）
            'cadence': {
                'optimal': (180, 200),
                'good_low': (170, 180),
                'good_high': (200, 210),
                'fair_low': (160, 170),
                'fair_high': (210, 220),
            },
            # 膝关节角度标准
            'knee_angle': {
                # 触地期：膝微屈，155-170度
                'ground_contact': {'ideal': (155, 170), 'acceptable': (145, 175)},
                # 最大弯曲：摆动期，90-130度
                'max_flexion': {'ideal': (90, 130), 'acceptable': (80, 140)},
                # 活动范围：应有足够的运动幅度
                'rom': {'ideal': (40, 70), 'acceptable': (30, 80)},
            },
            # 身体前倾标准（度）
            'body_lean': {
                'optimal': (5, 15),
                'acceptable': (3, 20),
            },
            # 稳定性标准
            'stability': {
                'excellent': 85,
                'good': 70,
                'fair': 55,
            }
        }

    def evaluate(self, kinematic_results: Dict, temporal_results: Dict,
                 view_angle: str = 'side') -> Dict:
        """
        综合评价跑步技术质量

        Args:
            kinematic_results: 运动学分析结果
            temporal_results: 时序模型分析结果
            view_angle: 视频视角

        Returns:
            评价结果
        """
        # 根据视角选择评估策略
        if view_angle == 'side':
            scores = self._evaluate_side_view(kinematic_results, temporal_results)
        elif view_angle in ['front', 'back']:
            scores = self._evaluate_frontal_view(kinematic_results, temporal_results)
        else:
            scores = self._evaluate_side_view(kinematic_results, temporal_results)

        # 提取各维度得分
        stability_score = scores['stability']
        efficiency_score = scores['efficiency']
        form_score = scores['form']

        # 加权计算总分
        total_score = (
            stability_score * self.weights['stability'] +
            efficiency_score * self.weights['efficiency'] +
            form_score * self.weights['form']
        )

        # 生成评级
        rating = self._get_rating(total_score)

        # 生成详细分析
        detailed_analysis = self._generate_detailed_analysis(
            kinematic_results, scores, view_angle
        )

        # 生成建议
        suggestions = self._generate_suggestions(
            kinematic_results, scores, view_angle
        )

        return {
            'total_score': round(total_score, 2),
            'rating': rating,
            'dimension_scores': {
                'stability': round(stability_score, 2),
                'efficiency': round(efficiency_score, 2),
                'form': round(form_score, 2)
            },
            'detailed_analysis': detailed_analysis,
            'suggestions': suggestions,
            'strengths': self._identify_strengths(scores),
            'weaknesses': self._identify_weaknesses(scores),
            'view_angle': view_angle,
        }

    def _evaluate_side_view(self, kinematic: Dict, temporal: Dict) -> Dict:
        """侧面视角评估"""
        scores = {}

        # 1. 稳定性评估
        scores['stability'] = self._evaluate_stability(kinematic, temporal)

        # 2. 效率评估（垂直振幅 + 步频）
        scores['efficiency'] = self._evaluate_efficiency_improved(kinematic)

        # 3. 跑姿评估（分阶段膝关节角度 + 前倾）
        scores['form'] = self._evaluate_form_improved(kinematic)

        return scores

    def _evaluate_frontal_view(self, kinematic: Dict, temporal: Dict) -> Dict:
        """正面/后方视角评估"""
        scores = {}

        # 稳定性（重点：横向稳定性）
        if 'lateral_stability' in kinematic:
            lateral = kinematic['lateral_stability']
            scores['stability'] = lateral.get('stability_score', 50)
        else:
            scores['stability'] = self._evaluate_stability(kinematic, temporal)

        # 效率
        scores['efficiency'] = self._evaluate_efficiency_improved(kinematic)

        # 跑姿（重点：下肢力线）
        if 'lower_limb_alignment' in kinematic:
            alignment = kinematic['lower_limb_alignment']
            rating = alignment.get('overall_rating', {})
            scores['form'] = rating.get('score', 60)
        else:
            scores['form'] = 60

        return scores

    def _evaluate_stability(self, kinematic: Dict, temporal: Dict) -> float:
        """评估动作稳定性"""
        scores = []

        # 来自运动学分析的稳定性
        if 'stability' in kinematic:
            stability_data = kinematic['stability']
            if isinstance(stability_data, dict) and 'overall' in stability_data:
                scores.append(stability_data['overall'])

        # 来自深度学习模型的稳定性
        if 'stability_score' in temporal:
            scores.append(temporal['stability_score'])

        # 膝关节角度稳定性
        if 'angles' in kinematic:
            knee_std_left = kinematic['angles'].get('knee_left_std', 0)
            knee_std_right = kinematic['angles'].get('knee_right_std', 0)
            avg_knee_std = (knee_std_left + knee_std_right) / 2
            # 标准差越小越稳定，但需要有一定的运动幅度
            if avg_knee_std > 0:
                knee_stability = max(0, 100 - avg_knee_std * 1.5)
                scores.append(knee_stability)

        return np.mean(scores) if scores else 50.0

    def _evaluate_efficiency_improved(self, kinematic: Dict) -> float:
        """改进的效率评估"""
        scores = []

        # 1. 垂直振幅评估（使用归一化振幅）
        if 'vertical_motion' in kinematic:
            vm = kinematic['vertical_motion']

            # 优先使用归一化振幅
            if 'amplitude_normalized' in vm:
                amp_norm = vm['amplitude_normalized']
                amp_score = self._score_vertical_amplitude(amp_norm)
            elif 'amplitude_rating' in vm:
                amp_score = vm['amplitude_rating'].get('score', 60)
            else:
                # 兼容旧版：使用原始振幅的启发式评分
                amp = vm.get('amplitude', 0)
                # 假设原始振幅在0.01-0.15范围
                amp_norm = amp * 100 / 0.25  # 近似转换
                amp_score = self._score_vertical_amplitude(amp_norm)

            scores.append(amp_score)

        # 2. 步频评估（保留用户原有阈值）
        if 'cadence' in kinematic:
            cadence = kinematic['cadence'].get('cadence', 0)
            cadence_score = self._score_cadence(cadence)
            scores.append(cadence_score)

        # 3. 触地/腾空比例评估
        if 'gait_cycle' in kinematic:
            gait = kinematic['gait_cycle']
            if 'gait_rating' in gait:
                gait_score = gait['gait_rating'].get('score', 70)
                scores.append(gait_score * 0.5)  # 权重较低

        return np.mean(scores) if scores else 60.0

    def _score_vertical_amplitude(self, amplitude_normalized: float) -> float:
        """
        评分垂直振幅
        基于归一化振幅（相对躯干长度的百分比）
        """
        stds = self.standards['vertical_amplitude']

        if stds['excellent'][0] <= amplitude_normalized <= stds['excellent'][1]:
            return 100
        elif stds['good'][0] < amplitude_normalized <= stds['good'][1]:
            return 80
        elif stds['fair'][0] < amplitude_normalized <= stds['fair'][1]:
            return 60
        elif amplitude_normalized < stds['poor_low'][1]:
            return 70  # 振幅过小，但不算严重问题
        else:
            return 40  # 振幅过大

    def _score_cadence(self, cadence: float) -> float:
        """评分步频（保留用户原有阈值逻辑）"""
        stds = self.standards['cadence']

        if stds['optimal'][0] <= cadence <= stds['optimal'][1]:
            return 100
        elif stds['good_low'][0] <= cadence < stds['good_low'][1]:
            return 85
        elif stds['good_high'][0] < cadence <= stds['good_high'][1]:
            return 85
        elif stds['fair_low'][0] <= cadence < stds['fair_low'][1]:
            return 70
        elif stds['fair_high'][0] < cadence <= stds['fair_high'][1]:
            return 70
        else:
            return 50

    def _evaluate_form_improved(self, kinematic: Dict) -> float:
        """改进的跑姿评估"""
        scores = []

        # 1. 分阶段膝关节角度评估
        if 'angles' in kinematic and 'phase_analysis' in kinematic['angles']:
            phase = kinematic['angles']['phase_analysis']

            # 触地期角度
            gc_angle = phase['ground_contact'].get('mean', 0)
            if gc_angle > 0:
                gc_score = self._score_knee_angle(gc_angle, 'ground_contact')
                scores.append(gc_score)

            # 最大弯曲角度
            max_flex = phase.get('max_flexion', 0)
            if max_flex > 0:
                flex_score = self._score_knee_angle(max_flex, 'max_flexion')
                scores.append(flex_score)

            # 关节活动范围
            rom = phase.get('range_of_motion', 0)
            if rom > 0:
                rom_score = self._score_knee_angle(rom, 'rom')
                scores.append(rom_score)

        # 兼容旧版：使用平均膝关节角度
        elif 'angles' in kinematic:
            knee_mean_left = kinematic['angles'].get('knee_left_mean', 0)
            knee_mean_right = kinematic['angles'].get('knee_right_mean', 0)

            for knee_angle in [knee_mean_left, knee_mean_right]:
                if knee_angle > 0:
                    # 使用宽松的判断标准
                    if 130 <= knee_angle <= 170:
                        scores.append(80)
                    elif 120 <= knee_angle < 130 or 170 < knee_angle <= 175:
                        scores.append(70)
                    else:
                        scores.append(55)

        # 2. 身体前倾评估
        if 'body_lean' in kinematic:
            lean = kinematic['body_lean']
            if 'rating' in lean:
                lean_score = lean['rating'].get('score', 70)
                scores.append(lean_score)

        return np.mean(scores) if scores else 65.0

    def _score_knee_angle(self, angle: float, phase_type: str) -> float:
        """评分膝关节角度"""
        stds = self.standards['knee_angle'][phase_type]

        ideal = stds['ideal']
        acceptable = stds['acceptable']

        if ideal[0] <= angle <= ideal[1]:
            return 100
        elif acceptable[0] <= angle <= acceptable[1]:
            return 75
        else:
            return 50

    def _get_rating(self, score: float) -> str:
        """根据分数获取评级"""
        if score >= self.thresholds['excellent']:
            return '优秀'
        elif score >= self.thresholds['good']:
            return '良好'
        elif score >= self.thresholds['fair']:
            return '一般'
        else:
            return '待改进'

    def _generate_detailed_analysis(self, kinematic: Dict, scores: Dict,
                                     view_angle: str) -> Dict:
        """生成详细分析"""
        analysis = {}

        # 垂直振幅分析
        if 'vertical_motion' in kinematic:
            vm = kinematic['vertical_motion']
            amp_norm = vm.get('amplitude_normalized', 0)

            if amp_norm > 0:
                if amp_norm <= 6:
                    analysis['vertical_amplitude'] = {
                        'value': f'{amp_norm:.1f}%',
                        'assessment': '垂直振幅控制优秀，能量利用效率高',
                        'level': 'excellent'
                    }
                elif amp_norm <= 10:
                    analysis['vertical_amplitude'] = {
                        'value': f'{amp_norm:.1f}%',
                        'assessment': '垂直振幅在可接受范围，有优化空间',
                        'level': 'good'
                    }
                else:
                    analysis['vertical_amplitude'] = {
                        'value': f'{amp_norm:.1f}%',
                        'assessment': '垂直振幅偏大，导致能量浪费',
                        'level': 'needs_improvement'
                    }

        # 膝关节角度分析
        if 'angles' in kinematic and 'phase_analysis' in kinematic['angles']:
            phase = kinematic['angles']['phase_analysis']
            gc_mean = phase['ground_contact'].get('mean', 0)
            max_flex = phase.get('max_flexion', 0)

            analysis['knee_angles'] = {
                'ground_contact': f'{gc_mean:.1f}°' if gc_mean > 0 else 'N/A',
                'max_flexion': f'{max_flex:.1f}°' if max_flex > 0 else 'N/A',
                'assessment': self._assess_knee_angles(gc_mean, max_flex)
            }

        # 步频分析
        if 'cadence' in kinematic:
            cadence = kinematic['cadence'].get('cadence', 0)
            analysis['cadence'] = {
                'value': f'{cadence:.0f} 步/分',
                'assessment': self._assess_cadence(cadence)
            }

        return analysis

    def _assess_knee_angles(self, gc_angle: float, max_flex: float) -> str:
        """评估膝关节角度"""
        assessments = []

        if gc_angle > 0:
            if 155 <= gc_angle <= 170:
                assessments.append('触地时膝关节缓冲良好')
            elif gc_angle > 170:
                assessments.append('触地时膝关节伸展过直，冲击力较大')
            else:
                assessments.append('触地时膝关节弯曲过多')

        if max_flex > 0:
            if 90 <= max_flex <= 130:
                assessments.append('摆动期膝关节弯曲充分')
            elif max_flex < 90:
                assessments.append('摆动期膝关节弯曲不足')
            else:
                assessments.append('摆动期膝关节弯曲偏少')

        return '；'.join(assessments) if assessments else '数据不足'

    def _assess_cadence(self, cadence: float) -> str:
        """评估步频"""
        if 180 <= cadence <= 200:
            return '步频处于最佳范围，跑步经济性好'
        elif 170 <= cadence < 180:
            return '步频略低于最佳范围，可适当提高'
        elif 200 < cadence <= 210:
            return '步频略高，注意避免过度紧张'
        elif cadence < 160:
            return '步频偏低，可能步幅过大，建议调整'
        elif cadence > 220:
            return '步频过高，可能影响跑步效率'
        else:
            return '步频有优化空间'

    def _generate_suggestions(self, kinematic: Dict, scores: Dict,
                               view_angle: str) -> List[str]:
        """生成改进建议"""
        suggestions = []

        # 基于各维度得分生成建议
        if scores['stability'] < 70:
            suggestions.append("加强核心力量训练，提高躯干稳定性")

        if scores['efficiency'] < 70:
            # 具体分析效率问题来源
            if 'vertical_motion' in kinematic:
                amp_norm = kinematic['vertical_motion'].get('amplitude_normalized', 0)
                if amp_norm > 10:
                    suggestions.append("减少垂直振幅，想象'贴地飞行'的感觉")
                elif amp_norm < 3:
                    suggestions.append("适当增加步幅，避免过于保守的跑姿")

            if 'cadence' in kinematic:
                cadence = kinematic['cadence'].get('cadence', 0)
                if cadence < 170:
                    suggestions.append("尝试提高步频至170-180步/分，可借助节拍器练习")

        if scores['form'] < 70:
            if 'angles' in kinematic and 'phase_analysis' in kinematic['angles']:
                phase = kinematic['angles']['phase_analysis']
                gc_angle = phase['ground_contact'].get('mean', 0)
                if gc_angle > 170:
                    suggestions.append("落地时保持膝关节微屈，减少冲击力")
                elif gc_angle < 145:
                    suggestions.append("落地时不要过度屈膝，保持自然姿态")

            if 'body_lean' in kinematic:
                lean = kinematic['body_lean'].get('forward_lean', 0)
                if lean < 5:
                    suggestions.append("适当前倾身体，利用重力辅助前进")
                elif lean > 20:
                    suggestions.append("避免过度前倾，保持自然跑姿")

        # 如果没有明显问题
        if not suggestions:
            suggestions.append("继续保持良好状态，可适当增加训练量挑战自己")

        return suggestions

    def _identify_strengths(self, scores: Dict) -> List[str]:
        """识别优势项"""
        strengths = []
        score_names = {
            'stability': '动作稳定性',
            'efficiency': '跑步效率',
            'form': '跑姿标准度'
        }

        for key, name in score_names.items():
            if scores.get(key, 0) >= 80:
                strengths.append(name)

        return strengths if strengths else ['暂无突出优势']

    def _identify_weaknesses(self, scores: Dict) -> List[str]:
        """识别薄弱项"""
        weaknesses = []
        score_names = {
            'stability': '动作稳定性',
            'efficiency': '跑步效率',
            'form': '跑姿标准度'
        }

        for key, name in score_names.items():
            if scores.get(key, 0) < 65:
                weaknesses.append(name)

        return weaknesses if weaknesses else ['无明显薄弱项']


# 模块测试
if __name__ == "__main__":
    print("=" * 60)
    print("测试重构版质量评估模块")
    print("=" * 60)

    # 模拟运动学分析结果
    mock_kinematic = {
        'vertical_motion': {
            'amplitude': 0.02,
            'amplitude_normalized': 7.5,
            'amplitude_rating': {'level': 'good', 'score': 80}
        },
        'cadence': {
            'cadence': 175,
            'confidence': 0.85
        },
        'angles': {
            'knee_left_mean': 155,
            'knee_right_mean': 158,
            'knee_left_std': 8,
            'knee_right_std': 7,
            'phase_analysis': {
                'ground_contact': {'mean': 165, 'std': 5},
                'flight': {'mean': 110, 'std': 8},
                'max_flexion': 105,
                'range_of_motion': 55
            }
        },
        'stability': {
            'overall': 78,
            'trunk': 82,
            'head': 75,
            'symmetry': 80
        },
        'body_lean': {
            'forward_lean': 10,
            'rating': {'score': 100}
        },
        'gait_cycle': {
            'phase_distribution': {
                'ground_contact': 0.45,
                'flight': 0.35,
                'transition': 0.20
            },
            'gait_rating': {'score': 80}
        }
    }

    mock_temporal = {
        'quality_score': 75,
        'stability_score': 72
    }

    evaluator = QualityEvaluator()
    results = evaluator.evaluate(mock_kinematic, mock_temporal, view_angle='side')

    print(f"\n总分: {results['total_score']}")
    print(f"评级: {results['rating']}")
    print(f"\n各维度得分:")
    for dim, score in results['dimension_scores'].items():
        print(f"  {dim}: {score}")

    print(f"\n优势: {results['strengths']}")
    print(f"薄弱项: {results['weaknesses']}")

    print(f"\n改进建议:")
    for i, sug in enumerate(results['suggestions'], 1):
        print(f"  {i}. {sug}")

    print("\n✅ 模块测试完成!")
