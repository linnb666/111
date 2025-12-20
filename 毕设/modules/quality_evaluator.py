import numpy as np
from typing import Dict
from config.config import QUALITY_WEIGHTS, QUALITY_THRESHOLDS
from typing import List

class QualityEvaluator:
    """技术质量评价器"""

    def __init__(self):
        """初始化评价器"""
        self.weights = QUALITY_WEIGHTS
        self.thresholds = QUALITY_THRESHOLDS

    def evaluate(self, kinematic_results: Dict, temporal_results: Dict) -> Dict:
        """
        综合评价跑步技术质量
        Args:
            kinematic_results: 运动学分析结果
            temporal_results: 时序模型分析结果
        Returns:
            评价结果
        """
        # 计算各维度得分
        stability_score = self._evaluate_stability(kinematic_results, temporal_results)
        efficiency_score = self._evaluate_efficiency(kinematic_results)
        form_score = self._evaluate_form(kinematic_results)
        rhythm_score = self._evaluate_rhythm(kinematic_results)

        # 加权计算总分
        total_score = (
                stability_score * self.weights['stability'] +
                efficiency_score * self.weights['efficiency'] +
                form_score * self.weights['form'] +
                rhythm_score * self.weights['rhythm']
        )

        # 生成评级
        rating = self._get_rating(total_score)

        # 生成建议
        suggestions = self._generate_suggestions(
            stability_score, efficiency_score, form_score, rhythm_score
        )

        return {
            'total_score': round(total_score, 2),
            'rating': rating,
            'dimension_scores': {
                'stability': round(stability_score, 2),
                'efficiency': round(efficiency_score, 2),
                'form': round(form_score, 2),
                'rhythm': round(rhythm_score, 2)
            },
            'suggestions': suggestions,
            'strengths': self._identify_strengths(stability_score, efficiency_score,
                                                  form_score, rhythm_score),
            'weaknesses': self._identify_weaknesses(stability_score, efficiency_score,
                                                    form_score, rhythm_score)
        }

    def _evaluate_stability(self, kinematic: Dict, temporal: Dict) -> float:
        """评估动作稳定性"""
        # 综合多个稳定性指标
        scores = []

        # 来自运动学分析的稳定性
        if 'stability' in kinematic and 'overall' in kinematic['stability']:
            scores.append(kinematic['stability']['overall'])

        # 来自深度学习模型的稳定性
        if 'stability_score' in temporal:
            scores.append(temporal['stability_score'])

        # 膝关节角度稳定性
        if 'angles' in kinematic:
            knee_std_left = kinematic['angles'].get('knee_left_std', 0)
            knee_std_right = kinematic['angles'].get('knee_right_std', 0)
            avg_knee_std = (knee_std_left + knee_std_right) / 2
            knee_stability = max(0, 100 - avg_knee_std * 2)
            scores.append(knee_stability)

        return np.mean(scores) if scores else 50.0

    def _evaluate_efficiency(self, kinematic: Dict) -> float:
        """评估动作效率"""
        scores = []

        # 垂直振幅评估（较小的垂直振幅意味着更高效）
        if 'vertical_motion' in kinematic:
            amplitude = kinematic['vertical_motion'].get('amplitude', 0)
            # 理想振幅范围：5-15像素
            if 5 <= amplitude <= 15:
                amplitude_score = 100
            elif amplitude < 5:
                amplitude_score = 80 - (5 - amplitude) * 10
            else:
                amplitude_score = max(50, 100 - (amplitude - 15) * 2)
            scores.append(amplitude_score)

        # 步频评估（理想步频：180-200步/分）
        if 'cadence' in kinematic:
            cadence = kinematic['cadence'].get('cadence', 0)
            if 180 <= cadence <= 200:
                cadence_score = 100
            elif 170 <= cadence < 180 or 200 < cadence <= 210:
                cadence_score = 85
            elif 160 <= cadence < 170 or 210 < cadence <= 230:
                cadence_score = 70
            else:
                cadence_score = 50
            scores.append(cadence_score)

        return np.mean(scores) if scores else 60.0

    def _evaluate_form(self, kinematic: Dict) -> float:
        """评估跑姿标准度"""
        scores = []

        # 膝关节角度评估
        if 'angles' in kinematic:
            knee_mean_left = kinematic['angles'].get('knee_left_mean', 0)
            knee_mean_right = kinematic['angles'].get('knee_right_mean', 0)

            # 理想膝关节角度范围：140-165度
            for knee_angle in [knee_mean_left, knee_mean_right]:
                if 140 <= knee_angle <= 165:
                    scores.append(100)
                elif 130 <= knee_angle < 140 or 165 < knee_angle <= 175:
                    scores.append(80)
                else:
                    scores.append(60)

        return np.mean(scores) if scores else 65.0

    def _evaluate_rhythm(self, kinematic: Dict) -> float:
        """评估节奏一致性"""
        scores = []

        # 步频稳定性
        if 'cadence' in kinematic:
            step_count = kinematic['cadence'].get('step_count', 0)
            duration = kinematic['cadence'].get('duration', 1)
            if step_count > 5 and duration > 2:
                # 基于步数的节奏评分
                rhythm_score = min(100, step_count / duration * 30)
                scores.append(rhythm_score)

        # 垂直运动一致性
        if 'vertical_motion' in kinematic:
            std_position = kinematic['vertical_motion'].get('std_position', 0)
            consistency_score = max(0, 100 - std_position)
            scores.append(consistency_score)

        return np.mean(scores) if scores else 65.0

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

    def _generate_suggestions(self, stability: float, efficiency: float,
                              form: float, rhythm: float) -> List[str]:
        """生成改进建议"""
        suggestions = []

        if stability < 70:
            suggestions.append("建议加强核心力量训练，提高动作稳定性")

        if efficiency < 70:
            suggestions.append("注意减少垂直振幅，保持步频在160-180步/分")

        if form < 70:
            suggestions.append("注意膝关节角度，保持在140-165度范围")

        if rhythm < 70:
            suggestions.append("保持步频节奏一致，避免速度波动过大")

        if not suggestions:
            suggestions.append("整体表现良好，继续保持！")

        return suggestions

    def _identify_strengths(self, stability: float, efficiency: float,
                            form: float, rhythm: float) -> List[str]:
        """识别优势项"""
        strengths = []
        scores = {
            '动作稳定性': stability,
            '跑步效率': efficiency,
            '跑姿标准度': form,
            '节奏一致性': rhythm
        }

        for name, score in scores.items():
            if score >= 80:
                strengths.append(name)

        return strengths if strengths else ['暂无突出优势']

    def _identify_weaknesses(self, stability: float, efficiency: float,
                             form: float, rhythm: float) -> List[str]:
        """识别薄弱项"""
        weaknesses = []
        scores = {
            '动作稳定性': stability,
            '跑步效率': efficiency,
            '跑姿标准度': form,
            '节奏一致性': rhythm
        }

        for name, score in scores.items():
            if score < 65:
                weaknesses.append(name)

        return weaknesses if weaknesses else ['无明显薄弱项']