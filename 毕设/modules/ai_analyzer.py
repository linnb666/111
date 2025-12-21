# modules/ai_analyzer.py
"""
AI分析模块 - 精简版
仅使用智谱AI (zai库 + glm-4.6模型)
"""
import os
import base64
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


class ZhipuProvider(BaseAIProvider):
    """智谱AI提供商 - 使用zai库"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化zai客户端"""
        try:
            from zai import ZhipuAiClient
            self.client = ZhipuAiClient(api_key=self.api_key)
            print("智谱AI客户端初始化成功")
        except ImportError:
            print("警告: zai库未安装，请运行 pip install zai")
            self.client = None
        except Exception as e:
            print(f"智谱AI客户端初始化失败: {e}")
            self.client = None

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """使用glm-4.6生成文本"""
        if not self.client:
            return "智谱AI客户端未初始化，请检查API密钥和zai库安装"

        try:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})

            response = self.client.chat.completions.create(
                model="glm-4.6",
                messages=messages,
                temperature=0.6,
                max_tokens=2000
            )

            # 提取响应内容
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            elif isinstance(response, dict):
                return response.get('choices', [{}])[0].get('message', {}).get('content', 'API返回异常')
            else:
                return str(response)

        except Exception as e:
            return f"智谱AI请求错误: {str(e)}"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """使用glm-4v分析图像"""
        if not self.client:
            return "智谱AI客户端未初始化"

        try:
            # 读取并编码图像
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_data}'}}
                ]
            }]

            response = self.client.chat.completions.create(
                model="glm-4v",
                messages=messages,
                temperature=0.6,
                max_tokens=1500
            )

            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            elif isinstance(response, dict):
                return response.get('choices', [{}])[0].get('message', {}).get('content', 'API返回异常')
            else:
                return str(response)

        except Exception as e:
            return f"图像分析错误: {str(e)}"

    def analyze_video_frames(self, frame_paths: List[str], prompt: str) -> str:
        """分析多个视频帧"""
        if not frame_paths:
            return "无法分析：未提供帧"

        # 逐帧分析并汇总
        frame_analyses = []
        for i, frame_path in enumerate(frame_paths):
            frame_prompt = f"""请分析这张跑步姿态图片（第{i+1}帧）：
1. 描述跑者当前的姿态
2. 识别可能存在的技术问题
3. 评估动作质量（好/一般/需改进）

请用简洁的语言回答。"""

            result = self.analyze_image(frame_path, frame_prompt)
            if not result.startswith("智谱AI") and not result.startswith("图像分析错误"):
                frame_analyses.append(f"**帧 {i+1}**: {result}")

        if not frame_analyses:
            return "多模态分析失败：无法获取有效的帧分析结果"

        # 汇总分析
        summary_prompt = f"""基于以下各帧的分析结果，请生成一份综合的跑步技术问题报告：

{chr(10).join(frame_analyses)}

请按照以下格式输出：
1. 整体技术评估
2. 各时间段发现的问题
3. 需要重点关注的技术细节
4. 改进建议"""

        return self.generate_text(summary_prompt)

    def analyze_time_segments(self, keyframe_data: List[Dict], kinematic_results: Dict) -> str:
        """时间段问题分析"""
        if not keyframe_data:
            return "无法进行时间段分析：未提供关键帧数据"

        # 构建运动学数据摘要
        cadence = kinematic_results.get('cadence', {}).get('cadence', 0)
        vertical_amp = kinematic_results.get('vertical_motion', {}).get('amplitude_normalized', 0)
        stability = kinematic_results.get('stability', {}).get('overall', 0)

        # 获取相位时间信息
        gait_cycle = kinematic_results.get('gait_cycle', {})
        phase_duration = gait_cycle.get('phase_duration_ms', {})

        context = f"""
运动学数据参考：
- 步频: {cadence:.1f} 步/分
- 垂直振幅: {vertical_amp:.2f}% 躯干长度
- 稳定性评分: {stability:.1f}/100
- 触地时间: {phase_duration.get('ground_contact', 0):.1f}ms
- 腾空时间: {phase_duration.get('flight', 0):.1f}ms
"""

        # 分析每个关键帧
        segment_analyses = []
        for i, kf in enumerate(keyframe_data):
            if not kf.get('detected', False):
                segment_analyses.append(f"时间 {kf['time_sec']:.2f}s: 未检测到姿态")
                continue

            frame_prompt = f"""你是一位专业的跑步教练。请分析这张跑步姿态图片。

当前时间点: {kf['time_sec']:.2f}秒

{context}

请重点分析：
1. 此时刻的身体姿态是否正确
2. 膝关节、髋关节角度是否合理
3. 躯干前倾程度
4. 是否存在明显的技术问题

请用2-3句话简要描述你观察到的问题或亮点。"""

            try:
                result = self.analyze_image(kf['path'], frame_prompt)
                if not result.startswith("智谱AI") and not result.startswith("图像分析错误"):
                    segment_analyses.append(f"**{kf['time_sec']:.2f}秒**: {result}")
            except Exception as e:
                segment_analyses.append(f"时间 {kf['time_sec']:.2f}s: 分析失败 - {str(e)}")

        if not segment_analyses:
            return "时间段分析失败：无法获取有效的分析结果"

        # 生成综合报告
        final_prompt = f"""请基于以下各时间点的跑步姿态分析，生成一份专业的时间段问题分析报告：

{chr(10).join(segment_analyses)}

{context}

请按照以下格式输出报告：

## 时间段问题分析

### 问题时间段识别
（列出存在明显问题的时间段，说明问题类型）

### 技术问题汇总
（总结视频中反复出现的技术问题）

### 改进优先级
（按重要程度排序建议改进的方面）

### 训练建议
（提供具体可操作的训练方法）"""

        return self.generate_text(final_prompt)


class LocalRuleEngine(BaseAIProvider):
    """本地规则引擎（无需API）"""

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """基于规则生成文本"""
        return "请使用 generate_analysis_report 方法获取分析报告"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """本地无法分析图像"""
        return "本地模式不支持图像分析，请配置智谱AI API"

    def generate_analysis_report(self, results: Dict) -> str:
        """基于规则生成分析报告"""
        quality = results.get('quality_evaluation', {})
        kinematic = results.get('kinematic_analysis', {})
        view_angle = results.get('view_angle', 'side')

        score = quality.get('total_score', 0)

        # 构建报告
        report = f"""## 跑步技术分析报告

### 一、总体评价

| 指标 | 结果 |
|------|------|
| **总体评分** | **{score:.1f}/100** |
| **技术评级** | {quality.get('rating', '待评估')} |
| **分析视角** | {self._get_view_name(view_angle)} |

"""
        # 评分解读
        if score >= 85:
            report += "> 您的跑步技术处于**优秀**水平，动作协调高效！\n\n"
        elif score >= 70:
            report += "> 您的跑步技术**良好**，有一定基础，存在提升空间。\n\n"
        elif score >= 55:
            report += "> 您的跑步技术处于**一般**水平，建议针对性改进。\n\n"
        else:
            report += "> 您的跑步技术有较大**提升空间**，建议系统训练。\n\n"

        # 各维度表现（移除节奏一致性）
        report += "### 二、各维度表现\n\n"
        dims = quality.get('dimension_scores', {})
        dim_names = {
            'stability': ('动作稳定性', '身体控制和核心力量'),
            'efficiency': ('跑步效率', '能量利用和步态经济性'),
            'form': ('跑姿标准度', '关节角度和身体姿态')
        }

        report += "| 维度 | 得分 | 等级 | 说明 |\n"
        report += "|------|------|------|------|\n"

        for key, (name, desc) in dim_names.items():
            dim_score = dims.get(key, 0)
            level = '优秀' if dim_score >= 85 else '良好' if dim_score >= 70 else '一般' if dim_score >= 55 else '待改进'
            report += f"| {name} | {dim_score:.1f} | {level} | {desc} |\n"

        report += "\n"

        # 关键运动学指标
        report += "### 三、关键技术指标\n\n"

        # 步频分析（新标准）
        cadence_data = kinematic.get('cadence', {})
        if cadence_data:
            cadence = cadence_data.get('cadence', 0)
            step_count = cadence_data.get('step_count', 0)
            duration = cadence_data.get('duration', 0)
            rating = cadence_data.get('rating', {})

            report += f"**步频分析**\n"
            report += f"- 步频: **{cadence:.1f} 步/分**\n"
            report += f"- 检测步数: {step_count} 步 (视频时长 {duration:.1f} 秒)\n"
            if rating:
                report += f"- 等级: {rating.get('description', '')}\n"

            # 新的步频评估标准
            if cadence >= 185:
                report += f"- 分析: 步频达到精英水平，跑步效率极佳\n"
            elif cadence >= 175:
                report += f"- 分析: 步频优秀，有助于减少触地时间和受伤风险\n"
            elif cadence >= 165:
                report += f"- 分析: 步频良好，可尝试逐步提高至175+步/分\n"
            elif cadence >= 155:
                report += f"- 分析: 步频一般，建议进行节拍器训练提高步频\n"
            else:
                report += f"- 分析: 步频较低，建议系统性训练提高步频至165+步/分\n"
            report += "\n"

        # 垂直振幅
        vertical = kinematic.get('vertical_motion', {})
        if vertical:
            amplitude = vertical.get('amplitude_normalized', 0)
            rating = vertical.get('amplitude_rating', {})

            report += f"**垂直振幅**\n"
            report += f"- 振幅: **{amplitude:.2f}%** (相对躯干长度)\n"
            if rating:
                report += f"- 等级: {rating.get('level', '未知')}\n"
                report += f"- 评估: {rating.get('description', '')}\n"

            if 3 <= amplitude <= 6:
                report += f"- 分析: 垂直振幅非常理想，能量利用效率高\n"
            elif amplitude < 3:
                report += f"- 分析: 振幅偏小，步态可能过于保守\n"
            elif amplitude <= 10:
                report += f"- 分析: 振幅在可接受范围，可尝试降低\n"
            else:
                report += f"- 分析: 振幅偏大，建议改善跑姿减少能量浪费\n"
            report += "\n"

        # 步态周期时间（毫秒）+ 触地时间评级
        gait_cycle = kinematic.get('gait_cycle', {})
        if gait_cycle and 'phase_duration_ms' in gait_cycle:
            phase_ms = gait_cycle['phase_duration_ms']
            ground_contact_ms = phase_ms.get('ground_contact', 0)
            flight_ms = phase_ms.get('flight', 0)

            report += "**步态周期时间**\n\n"

            # 触地时间评级（新标准）
            if ground_contact_ms > 0:
                if ground_contact_ms < 210:
                    gc_level = "精英"
                elif ground_contact_ms < 240:
                    gc_level = "优秀"
                elif ground_contact_ms < 270:
                    gc_level = "良好"
                elif ground_contact_ms < 300:
                    gc_level = "一般"
                else:
                    gc_level = "较差"
                report += f"- 触地时间: **{ground_contact_ms:.1f} ms** ({gc_level})\n"
            else:
                report += f"- 触地时间: 数据不足\n"

            report += f"- 腾空时间: {flight_ms:.1f} ms\n" if flight_ms > 0 else "- 腾空时间: 数据不足\n"

            if gait_cycle.get('avg_cycle_duration_ms', 0) > 0:
                report += f"- 完整周期: {gait_cycle.get('avg_cycle_duration_ms', 0):.1f} ms\n"
            report += "\n"

        # 膝关节角度（侧面视角）
        angles = kinematic.get('angles', {})
        if 'phase_analysis' in angles and view_angle == 'side':
            phase_analysis = angles['phase_analysis']

            report += f"**膝关节角度（分阶段）**\n\n"
            report += "| 阶段 | 平均角度 | 范围 | 参考值 |\n"
            report += "|------|----------|------|--------|\n"

            gc = phase_analysis.get('ground_contact', {})
            if gc.get('count', 0) > 0:
                report += f"| 触地期 | {gc.get('mean', 0):.1f}° | {gc.get('min', 0):.1f}°-{gc.get('max', 0):.1f}° | 155-170° |\n"

            fl = phase_analysis.get('flight', {})
            if fl.get('count', 0) > 0:
                report += f"| 腾空期 | {fl.get('mean', 0):.1f}° | {fl.get('min', 0):.1f}°-{fl.get('max', 0):.1f}° | 90-130° |\n"

            tr = phase_analysis.get('transition', {})
            if tr.get('count', 0) > 0:
                report += f"| 过渡期 | {tr.get('mean', 0):.1f}° | {tr.get('min', 0):.1f}°-{tr.get('max', 0):.1f}° | - |\n"

            report += "\n"

        # 稳定性分析（移除对称性，突出肩部稳定性）
        stability = kinematic.get('stability', {})
        if stability:
            report += f"**稳定性分析**\n"
            report += f"- 综合稳定性: {stability.get('overall', 0):.1f}/100\n"
            report += f"- 躯干稳定: {stability.get('trunk', 0):.1f}/100\n"
            report += f"- 头部稳定: {stability.get('head', 0):.1f}/100\n"
            # 正面视角显示肩部晃动（移除对称性）
            if 'shoulder_sway' in stability:
                report += f"- 肩部稳定: {stability.get('shoulder_sway', 0):.1f}/100\n"
            report += "\n"

        # 优势分析
        strengths = quality.get('strengths', [])
        if strengths and strengths != ['暂无突出优势']:
            report += "### 四、技术优势\n\n"
            for strength in strengths:
                report += f"- {strength}\n"
            report += "\n"

        # 薄弱项
        weaknesses = quality.get('weaknesses', [])
        if weaknesses and weaknesses != ['无明显薄弱项']:
            report += "### 五、待改进项\n\n"
            for weakness in weaknesses:
                report += f"- {weakness}\n"
            report += "\n"

        # 改进建议
        suggestions = quality.get('suggestions', [])
        if suggestions:
            report += "### 六、改进建议\n\n"
            for i, suggestion in enumerate(suggestions, 1):
                report += f"{i}. {suggestion}\n"
            report += "\n"

        # 总结
        report += "### 七、总结\n\n"
        if score >= 85:
            report += "您的跑步技术非常出色！动作协调、能量利用高效。继续保持当前状态，可以尝试挑战更高配速或更长距离。\n"
        elif score >= 70:
            report += "整体表现良好，具备较好的跑步基础。针对上述建议进行专项练习，您的跑步技术会有明显提升。\n"
        elif score >= 55:
            report += "基础动作已经掌握，但存在一些可以优化的环节。建议从步频控制和核心稳定性入手，循序渐进地改善跑姿。\n"
        else:
            report += "每个跑者都有提升空间，不要气馁！建议从基础开始，重点关注身体姿态和步频节奏。坚持科学训练，进步指日可待。\n"

        report += "\n---\n*本报告由跑步动作分析系统自动生成*\n"

        return report

    def _get_view_name(self, view: str) -> str:
        """获取视角中文名称"""
        names = {
            'side': '侧面视角',
            'front': '正面视角'
        }
        return names.get(view, view)


class AIAnalyzer:
    """AI分析器主类"""

    def __init__(self, api_key: str = None):
        """
        初始化AI分析器

        Args:
            api_key: 智谱AI API密钥
        """
        self.enabled = AI_CONFIG.get('enabled', False)
        api_key = api_key or AI_CONFIG.get('api_key', '')

        # 初始化智谱AI提供商
        if api_key:
            self.provider = ZhipuProvider(api_key)
            self.provider_name = 'zhipu'
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

        # 尝试使用智谱AI
        try:
            prompt = self._build_analysis_prompt(analysis_results)
            system_prompt = self._get_system_prompt()

            response = self.provider.generate_text(prompt, system_prompt)

            # 检查响应是否有效
            if response and not response.startswith('智谱AI') and not response.startswith('请求错误'):
                return response
            else:
                print(f"智谱AI响应异常: {response}")
                return self.local_engine.generate_analysis_report(analysis_results)

        except Exception as e:
            print(f"AI分析出错: {e}")
            return self.local_engine.generate_analysis_report(analysis_results)

    def analyze_pose_image(self, image_path: str) -> str:
        """分析姿态图像"""
        if self.provider_name == 'local':
            return "本地模式不支持图像分析，请配置智谱AI API"

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
        """分析视频帧序列"""
        if self.provider_name == 'local':
            return "本地模式不支持视频分析，请配置智谱AI API"

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

    def analyze_time_segments(self, keyframe_data: List[Dict], kinematic_results: Dict) -> str:
        """多模态时间段问题分析"""
        if self.provider_name == 'local':
            return self._local_time_segment_analysis(keyframe_data, kinematic_results)

        try:
            return self.provider.analyze_time_segments(keyframe_data, kinematic_results)
        except Exception as e:
            print(f"多模态时间段分析失败: {e}")
            return self._local_time_segment_analysis(keyframe_data, kinematic_results)

    def _local_time_segment_analysis(self, keyframe_data: List[Dict], kinematic_results: Dict) -> str:
        """本地时间段分析（基于规则）"""
        if not keyframe_data:
            return "无法进行时间段分析：未提供关键帧数据"

        cadence = kinematic_results.get('cadence', {}).get('cadence', 0)
        vertical_amp = kinematic_results.get('vertical_motion', {}).get('amplitude_normalized', 0)
        stability = kinematic_results.get('stability', {}).get('overall', 0)

        # 识别问题时间段
        problem_segments = []
        angles = kinematic_results.get('angles', {})
        phase_analysis = angles.get('phase_analysis', {})

        if cadence < 160:
            problem_segments.append("全程: 步频偏低，建议提高节奏")
        elif cadence > 210:
            problem_segments.append("全程: 步频过高，注意控制")

        if vertical_amp > 10:
            problem_segments.append("全程: 垂直振幅偏大，能量损耗较多")

        if stability < 60:
            problem_segments.append("全程: 动作稳定性不足，需加强核心训练")

        gc = phase_analysis.get('ground_contact', {})
        if gc.get('mean', 180) < 145:
            problem_segments.append("触地阶段: 膝关节弯曲过大")

        fl = phase_analysis.get('flight', {})
        if fl.get('mean', 90) > 140:
            problem_segments.append("腾空阶段: 腿部后摆不足")

        report = "## 时间段问题分析\n\n"

        if problem_segments:
            report += "### 识别到的问题\n\n"
            for i, problem in enumerate(problem_segments, 1):
                report += f"{i}. {problem}\n"
            report += "\n"
        else:
            report += "### 分析结果\n\n未发现明显技术问题，整体表现良好。\n\n"

        report += "### 关键帧时间点\n\n"
        for kf in keyframe_data:
            status = "姿态正常" if kf.get('detected', False) else "未检测到姿态"
            report += f"- {kf['time_sec']:.2f}s: {status}\n"

        report += "\n*如需更精确的多模态分析，请启用智谱AI*\n"

        return report

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

要求：语言专业但易懂，建议具体可操作，态度积极正面。"""

        return prompt

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一位经验丰富的跑步教练和运动科学专家。请根据提供的数据，给出专业、友好、有建设性的分析报告。"""


def create_ai_analyzer(api_key: str = None) -> AIAnalyzer:
    """
    创建AI分析器

    Args:
        api_key: 智谱AI API密钥

    Returns:
        AIAnalyzer实例
    """
    return AIAnalyzer(api_key)


# 模块测试
if __name__ == "__main__":
    print("=" * 60)
    print("测试AI分析模块（智谱AI版）")
    print("=" * 60)

    mock_results = {
        'quality_evaluation': {
            'total_score': 75.5,
            'rating': '良好',
            'dimension_scores': {
                'stability': 78,
                'efficiency': 72,
                'form': 76
            },
            'strengths': ['动作稳定性'],
            'weaknesses': [],
            'suggestions': ['可适当提高步频', '保持当前良好状态']
        },
        'kinematic_analysis': {
            'cadence': {'cadence': 175, 'step_count': 15, 'duration': 5.0},
            'vertical_motion': {'amplitude_normalized': 7.5},
            'stability': {'overall': 78, 'trunk': 82, 'head': 75},
            'gait_cycle': {
                'phase_duration_ms': {
                    'ground_contact': 180.5,
                    'flight': 120.3,
                    'transition': 45.2
                },
                'avg_cycle_duration_ms': 345.0
            }
        }
    }

    print("\n测试本地规则引擎...")
    analyzer = AIAnalyzer()
    report = analyzer.local_engine.generate_analysis_report(mock_results)
    print(report)

    print("\n智谱AI分析模块测试完成!")
