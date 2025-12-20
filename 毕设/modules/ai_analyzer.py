import os
import json
import requests
from typing import Dict, Optional
from config.config import AI_CONFIG


class AIAnalyzer:
    """AI文本分析器"""

    def __init__(self):
        """初始化AI分析器"""
        self.enabled = AI_CONFIG['enabled']
        self.provider = AI_CONFIG['provider']
        self.api_key = AI_CONFIG['api_key']
        self.model = AI_CONFIG['model']

    def generate_analysis_report(self, analysis_results: Dict) -> str:
        """
        生成AI增强的分析报告
        Args:
            analysis_results: 结构化分析结果
        Returns:
            AI生成的文本报告
        """
        if not self.enabled or not self.api_key:
            return self._generate_rule_based_report(analysis_results)

        try:
            if self.provider == 'openai':
                return self._call_openai_api(analysis_results)
            elif self.provider == 'anthropic':
                return self._call_anthropic_api(analysis_results)
            elif self.provider == 'qwen':
                return self._call_qwen_api(analysis_results)
            else:
                return self._generate_rule_based_report(analysis_results)
        except Exception as e:
            print(f"AI API调用失败: {e}")
            return self._generate_rule_based_report(analysis_results)

    def _call_openai_api(self, results: Dict) -> str:
        """调用OpenAI API"""
        prompt = self._build_prompt(results)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': '你是一位专业的跑步教练，请根据分析数据给出专业建议。'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': AI_CONFIG['max_tokens'],
            'temperature': AI_CONFIG['temperature']
        }

        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"API请求失败: {response.status_code}")

    def _call_anthropic_api(self, results: Dict) -> str:
        """调用Anthropic Claude API"""
        prompt = self._build_prompt(results)

        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }

        data = {
            'model': 'claude-3-sonnet-20240229',
            'max_tokens': AI_CONFIG['max_tokens'],
            'messages': [
                {'role': 'user', 'content': prompt}
            ]
        }

        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            raise Exception(f"API请求失败: {response.status_code}")

    def _call_qwen_api(self, results: Dict) -> str:
        """调用通义千问API"""
        # 根据实际API文档实现
        return self._generate_rule_based_report(results)

    def _build_prompt(self, results: Dict) -> str:
        """构建AI提示词"""
        quality = results.get('quality_evaluation', {})

        prompt = f"""
请根据以下跑步分析数据，生成一份专业、友好的技术分析报告：

总体评分：{quality.get('total_score', 0)}/100
评级：{quality.get('rating', '未知')}

各维度得分：
- 动作稳定性：{quality.get('dimension_scores', {}).get('stability', 0)}/100
- 跑步效率：{quality.get('dimension_scores', {}).get('efficiency', 0)}/100
- 跑姿标准度：{quality.get('dimension_scores', {}).get('form', 0)}/100
- 节奏一致性：{quality.get('dimension_scores', {}).get('rhythm', 0)}/100

优势：{', '.join(quality.get('strengths', []))}
薄弱项：{', '.join(quality.get('weaknesses', []))}

请生成：
1. 整体评价（2-3句话）
2. 优势分析
3. 改进建议（3-5条具体可行的建议）
4. 鼓励语

要求：专业、友好、具体、可操作
"""
        return prompt

    def _generate_rule_based_report(self, results: Dict) -> str:
        """生成基于规则的报告（无需API）"""
        quality = results.get('quality_evaluation', {})

        report = f"""
【跑步技术分析报告】

一、总体评价
您的跑步技术总体得分为 {quality.get('total_score', 0):.1f} 分，评级：{quality.get('rating', '待评估')}。

二、各维度表现
• 动作稳定性：{quality.get('dimension_scores', {}).get('stability', 0):.1f} 分
• 跑步效率：{quality.get('dimension_scores', {}).get('efficiency', 0):.1f} 分
• 跑姿标准度：{quality.get('dimension_scores', {}).get('form', 0):.1f} 分
• 节奏一致性：{quality.get('dimension_scores', {}).get('rhythm', 0):.1f} 分

三、优势分析
您在以下方面表现出色：
{self._format_list(quality.get('strengths', []))}

四、改进建议
{self._format_list(quality.get('suggestions', []))}

五、总结
{'继续保持良好状态，稳步提升！' if quality.get('total_score', 0) >= 70 else '通过针对性训练，您的跑步技术将会有明显提升！'}
"""
        return report

    def _format_list(self, items: list) -> str:
        """格式化列表"""
        return '\n'.join(f"• {item}" for item in items)