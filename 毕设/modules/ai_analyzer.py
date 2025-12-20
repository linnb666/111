# modules/ai_analyzer.py
"""
重构版AI分析模块
支持多种大模型API：
1. OpenAI (GPT-4/GPT-4V)
2. Anthropic (Claude)
3. 通义千问 (Qwen)
4. 智谱AI (GLM)
5. 百度文心一言
6. 本地规则引擎（无API时使用）

预留多模态视频分析接口
"""
import os
import json
import base64
import requests
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from pathlib import Path
from config.config import AI_CONFIG


class BaseAIProvider(ABC):
    """AI提供商基类"""

    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """生成文本"""
        pass

    @abstractmethod
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """分析图像（多模态）"""
        pass

    def analyze_video_frames(self, frame_paths: List[str], prompt: str) -> str:
        """分析视频帧序列（多模态）"""
        # 默认实现：分析关键帧
        if frame_paths:
            return self.analyze_image(frame_paths[0], prompt)
        return "无法分析：未提供帧"


class OpenAIProvider(BaseAIProvider):
    """OpenAI API提供商"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """生成文本"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        data = {
            'model': self.model,
            'messages': messages,
            'max_tokens': 1000,
            'temperature': 0.7
        }

        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API请求失败: {response.status_code}"
        except Exception as e:
            return f"请求错误: {str(e)}"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """使用GPT-4V分析图像"""
        if not self.model.startswith('gpt-4'):
            return "当前模型不支持图像分析，请使用 gpt-4-vision-preview"

        # 读取并编码图像
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': 'gpt-4-vision-preview',
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{image_data}'
                        }
                    }
                ]
            }],
            'max_tokens': 1000
        }

        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API请求失败: {response.status_code}"
        except Exception as e:
            return f"请求错误: {str(e)}"


class AnthropicProvider(BaseAIProvider):
    """Anthropic Claude API提供商"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """生成文本"""
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }

        data = {
            'model': self.model,
            'max_tokens': 1000,
            'messages': [{'role': 'user', 'content': prompt}]
        }

        if system_prompt:
            data['system'] = system_prompt

        try:
            response = requests.post(
                f'{self.base_url}/messages',
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()['content'][0]['text']
            else:
                return f"API请求失败: {response.status_code}"
        except Exception as e:
            return f"请求错误: {str(e)}"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """使用Claude分析图像"""
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # 获取图像类型
        ext = Path(image_path).suffix.lower()
        media_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                       '.png': 'image/png', '.gif': 'image/gif'}
        media_type = media_types.get(ext, 'image/jpeg')

        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }

        data = {
            'model': self.model,
            'max_tokens': 1000,
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': media_type,
                            'data': image_data
                        }
                    },
                    {'type': 'text', 'text': prompt}
                ]
            }]
        }

        try:
            response = requests.post(
                f'{self.base_url}/messages',
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()['content'][0]['text']
            else:
                return f"API请求失败: {response.status_code}"
        except Exception as e:
            return f"请求错误: {str(e)}"


class QwenProvider(BaseAIProvider):
    """通义千问API提供商"""

    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1"

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """生成文本"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        data = {
            'model': self.model,
            'input': {'messages': messages},
            'parameters': {'max_tokens': 1000}
        }

        try:
            response = requests.post(
                f'{self.base_url}/services/aigc/text-generation/generation',
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('output', {}).get('text', 'API返回异常')
            else:
                return f"API请求失败: {response.status_code}"
        except Exception as e:
            return f"请求错误: {str(e)}"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """使用通义千问VL分析图像"""
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': 'qwen-vl-plus',
            'input': {
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'image': f'data:image/jpeg;base64,{image_data}'},
                        {'text': prompt}
                    ]
                }]
            }
        }

        try:
            response = requests.post(
                f'{self.base_url}/services/aigc/multimodal-generation/generation',
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('output', {}).get('choices', [{}])[0].get('message', {}).get('content', 'API返回异常')
            else:
                return f"API请求失败: {response.status_code}"
        except Exception as e:
            return f"请求错误: {str(e)}"


class ZhipuProvider(BaseAIProvider):
    """智谱AI GLM API提供商"""

    def __init__(self, api_key: str, model: str = "glm-4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """生成文本"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        data = {
            'model': self.model,
            'messages': messages,
            'max_tokens': 1000
        }

        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API请求失败: {response.status_code}"
        except Exception as e:
            return f"请求错误: {str(e)}"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """使用GLM-4V分析图像"""
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': 'glm-4v',
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_data}'}}
                ]
            }],
            'max_tokens': 1000
        }

        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API请求失败: {response.status_code}"
        except Exception as e:
            return f"请求错误: {str(e)}"


class LocalRuleEngine(BaseAIProvider):
    """本地规则引擎（无需API）"""

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """基于规则生成文本"""
        return "请使用 generate_analysis_report 方法获取分析报告"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """本地无法分析图像"""
        return "本地模式不支持图像分析，请配置AI API"

    def generate_analysis_report(self, results: Dict) -> str:
        """基于规则生成分析报告"""
        quality = results.get('quality_evaluation', {})
        kinematic = results.get('kinematic_analysis', {})

        # 提取详细分析
        detailed = quality.get('detailed_analysis', {})

        # 构建报告
        report = f"""
【跑步技术分析报告】

一、总体评价
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
您的跑步技术总体得分为 {quality.get('total_score', 0):.1f} 分
评级：{quality.get('rating', '待评估')}

二、各维度表现
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        dims = quality.get('dimension_scores', {})
        dim_names = {
            'stability': '动作稳定性',
            'efficiency': '跑步效率',
            'form': '跑姿标准度',
            'rhythm': '节奏一致性'
        }

        for key, name in dim_names.items():
            score = dims.get(key, 0)
            level = '优秀' if score >= 85 else '良好' if score >= 70 else '一般' if score >= 55 else '待改进'
            bar = '█' * int(score / 10) + '░' * (10 - int(score / 10))
            report += f"  {name}: {score:.1f}  [{bar}]  {level}\n"

        # 关键指标
        report += """
三、关键技术指标
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        if 'cadence' in detailed:
            report += f"  步频: {detailed['cadence'].get('value', 'N/A')}\n"
            report += f"       {detailed['cadence'].get('assessment', '')}\n\n"

        if 'vertical_amplitude' in detailed:
            report += f"  垂直振幅: {detailed['vertical_amplitude'].get('value', 'N/A')}\n"
            report += f"           {detailed['vertical_amplitude'].get('assessment', '')}\n\n"

        if 'knee_angles' in detailed:
            ka = detailed['knee_angles']
            report += f"  膝关节角度:\n"
            report += f"    触地期: {ka.get('ground_contact', 'N/A')}\n"
            report += f"    最大弯曲: {ka.get('max_flexion', 'N/A')}\n"
            report += f"    评估: {ka.get('assessment', '')}\n\n"

        # 优势与不足
        report += """
四、优势分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for strength in quality.get('strengths', ['暂无突出优势']):
            report += f"  ✓ {strength}\n"

        report += """
五、改进建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for i, suggestion in enumerate(quality.get('suggestions', ['继续保持']), 1):
            report += f"  {i}. {suggestion}\n"

        # 总结
        score = quality.get('total_score', 0)
        if score >= 85:
            closing = "您的跑步技术非常出色！继续保持，挑战更高目标！"
        elif score >= 70:
            closing = "整体表现良好，针对建议进行练习，您会有明显进步！"
        elif score >= 55:
            closing = "基础已具备，通过系统训练，您的跑步技术会有质的飞跃！"
        else:
            closing = "每个人都有提升空间，坚持科学训练，进步指日可待！"

        report += f"""
六、总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{closing}
"""
        return report


class AIAnalyzer:
    """AI分析器主类"""

    # 支持的提供商
    PROVIDERS = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'qwen': QwenProvider,
        'zhipu': ZhipuProvider,
        'local': LocalRuleEngine,
    }

    def __init__(self, provider: str = None, api_key: str = None):
        """
        初始化AI分析器

        Args:
            provider: AI提供商 ('openai', 'anthropic', 'qwen', 'zhipu', 'local')
            api_key: API密钥
        """
        # 使用配置或参数
        self.enabled = AI_CONFIG.get('enabled', False)
        provider = provider or AI_CONFIG.get('provider', 'local')
        api_key = api_key or AI_CONFIG.get('api_key', '')

        # 初始化提供商
        if provider in self.PROVIDERS:
            if provider == 'local' or not api_key:
                self.provider = LocalRuleEngine()
                self.provider_name = 'local'
            else:
                self.provider = self.PROVIDERS[provider](api_key)
                self.provider_name = provider
        else:
            self.provider = LocalRuleEngine()
            self.provider_name = 'local'

        # 本地规则引擎实例（始终保留作为后备）
        self.local_engine = LocalRuleEngine()

    def generate_analysis_report(self, analysis_results: Dict) -> str:
        """
        生成AI增强的分析报告

        Args:
            analysis_results: 结构化分析结果

        Returns:
            AI生成的文本报告
        """
        # 如果使用本地规则引擎
        if self.provider_name == 'local':
            return self.local_engine.generate_analysis_report(analysis_results)

        # 尝试使用AI API
        try:
            prompt = self._build_analysis_prompt(analysis_results)
            system_prompt = self._get_system_prompt()

            response = self.provider.generate_text(prompt, system_prompt)

            # 检查响应是否有效
            if response and not response.startswith('API请求失败') and not response.startswith('请求错误'):
                return response
            else:
                print(f"AI API响应异常: {response}")
                return self.local_engine.generate_analysis_report(analysis_results)

        except Exception as e:
            print(f"AI分析出错: {e}")
            return self.local_engine.generate_analysis_report(analysis_results)

    def analyze_pose_image(self, image_path: str) -> str:
        """
        分析姿态图像（多模态）

        Args:
            image_path: 图像路径

        Returns:
            AI分析结果
        """
        if self.provider_name == 'local':
            return "本地模式不支持图像分析，请配置AI API"

        prompt = """请分析这张跑步姿态图像：
1. 描述跑者的整体姿态
2. 分析膝关节、髋关节的角度是否合理
3. 评估躯干前倾程度
4. 判断手臂摆动是否协调
5. 指出可能的技术问题
6. 给出改进建议"""

        try:
            return self.provider.analyze_image(image_path, prompt)
        except Exception as e:
            return f"图像分析失败: {str(e)}"

    def analyze_video_sequence(self, frame_paths: List[str]) -> str:
        """
        分析视频帧序列（多模态）

        Args:
            frame_paths: 关键帧路径列表

        Returns:
            AI分析结果
        """
        if self.provider_name == 'local':
            return "本地模式不支持视频分析，请配置AI API"

        prompt = """请分析这些跑步视频关键帧：
1. 描述跑者的整体技术水平
2. 分析步态周期的完整性
3. 评估动作的稳定性和一致性
4. 指出技术优势和不足
5. 提供专业的训练建议"""

        try:
            return self.provider.analyze_video_frames(frame_paths, prompt)
        except Exception as e:
            return f"视频分析失败: {str(e)}"

    def _build_analysis_prompt(self, results: Dict) -> str:
        """构建分析提示词"""
        quality = results.get('quality_evaluation', {})
        kinematic = results.get('kinematic_analysis', {})

        prompt = f"""请根据以下跑步分析数据，生成一份专业、详细的技术分析报告：

【总体评分】{quality.get('total_score', 0):.1f}/100
【评级】{quality.get('rating', '未知')}

【各维度得分】
- 动作稳定性：{quality.get('dimension_scores', {}).get('stability', 0):.1f}/100
- 跑步效率：{quality.get('dimension_scores', {}).get('efficiency', 0):.1f}/100
- 跑姿标准度：{quality.get('dimension_scores', {}).get('form', 0):.1f}/100
- 节奏一致性：{quality.get('dimension_scores', {}).get('rhythm', 0):.1f}/100

【运动学指标】
- 步频：{kinematic.get('cadence', {}).get('cadence', 0):.0f} 步/分
- 垂直振幅：{kinematic.get('vertical_motion', {}).get('amplitude_normalized', 0):.1f}%（相对躯干）
- 稳定性评分：{kinematic.get('stability', {}).get('overall', 0):.1f}

【优势】{', '.join(quality.get('strengths', []))}
【薄弱项】{', '.join(quality.get('weaknesses', []))}

请生成：
1. 整体评价（2-3句话概括）
2. 技术亮点分析
3. 需要改进的方面
4. 具体可操作的训练建议（3-5条）
5. 鼓励性总结

要求：
- 语言专业但易懂
- 建议具体可操作
- 态度积极正面
"""
        return prompt

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一位经验丰富的跑步教练和运动科学专家。你擅长：
- 分析跑步技术和生物力学
- 识别跑姿问题和改进空间
- 提供个性化的训练建议
- 用通俗易懂的语言解释专业概念

请根据提供的数据，给出专业、友好、有建设性的分析报告。"""


# 工厂函数
def create_ai_analyzer(provider: str = None, api_key: str = None) -> AIAnalyzer:
    """
    创建AI分析器

    Args:
        provider: 提供商名称
        api_key: API密钥

    Returns:
        AIAnalyzer实例
    """
    return AIAnalyzer(provider, api_key)


# 模块测试
if __name__ == "__main__":
    print("=" * 60)
    print("测试AI分析模块")
    print("=" * 60)

    # 模拟分析结果
    mock_results = {
        'quality_evaluation': {
            'total_score': 75.5,
            'rating': '良好',
            'dimension_scores': {
                'stability': 78,
                'efficiency': 72,
                'form': 76,
                'rhythm': 75
            },
            'detailed_analysis': {
                'cadence': {'value': '175 步/分', 'assessment': '步频良好'},
                'vertical_amplitude': {'value': '7.5%', 'assessment': '振幅在可接受范围'},
                'knee_angles': {
                    'ground_contact': '165°',
                    'max_flexion': '105°',
                    'assessment': '膝关节角度正常'
                }
            },
            'strengths': ['动作稳定性'],
            'weaknesses': [],
            'suggestions': ['可适当提高步频', '保持当前良好状态']
        },
        'kinematic_analysis': {
            'cadence': {'cadence': 175},
            'vertical_motion': {'amplitude_normalized': 7.5},
            'stability': {'overall': 78}
        }
    }

    # 测试本地规则引擎
    print("\n测试本地规则引擎...")
    analyzer = AIAnalyzer(provider='local')
    report = analyzer.generate_analysis_report(mock_results)
    print(report)

    print("\n✅ AI分析模块测试完成!")
