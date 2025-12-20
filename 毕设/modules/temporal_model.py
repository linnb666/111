# modules/temporal_model.py (修复版)
"""
时序深度学习分析模块（修复版）
修复：适配CNN模型的5维输出
"""
import torch
import numpy as np
from typing import List, Dict
from pathlib import Path
from config.config import MODEL_CONFIG, CHECKPOINT_DIR


class TemporalModelAnalyzer:
    """时序模型分析器（修复版）"""

    def __init__(self, device='cpu'):
        """初始化模型"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 导入分离后的模型
        from models.lstm_model import RunningPhaseLSTM
        from models.cnn_model import RunningQualityCNN

        # 实例化模型
        self.phase_model = RunningPhaseLSTM().to(self.device)
        self.quality_model = RunningQualityCNN().to(self.device)

        # 加载预训练权重（如果存在）
        self._load_pretrained_weights()

        self.phase_model.eval()
        self.quality_model.eval()

    def _load_pretrained_weights(self):
        """加载预训练权重"""
        phase_checkpoint = CHECKPOINT_DIR / 'phase_model.pth'
        quality_checkpoint = CHECKPOINT_DIR / 'quality_model.pth'

        if phase_checkpoint.exists():
            try:
                self.phase_model.load_state_dict(
                    torch.load(phase_checkpoint, map_location=self.device)
                )
                print(f"加载阶段模型权重: {phase_checkpoint}")
            except Exception as e:
                print(f"⚠️  警告: 无法加载阶段模型权重 - {e}")
        else:
            print(f"ℹ️  未找到阶段模型权重文件，使用随机初始化")

        if quality_checkpoint.exists():
            try:
                self.quality_model.load_state_dict(
                    torch.load(quality_checkpoint, map_location=self.device)
                )
                print(f"加载质量模型权重: {quality_checkpoint}")
            except Exception as e:
                print(f"⚠️  警告: 无法加载质量模型权重 - {e}")
        else:
            print(f"ℹ️  未找到质量模型权重文件，使用随机初始化")

    def analyze(self, keypoints_sequence: List[Dict]) -> Dict:
        """
        分析关键点序列（修复版 - 适配5维输出）
        Args:
            keypoints_sequence: 关键点时间序列
        Returns:
            分析结果
        """
        # 准备输入数据
        input_tensor = self._prepare_input(keypoints_sequence)

        if input_tensor is None:
            print("⚠️  输入数据不足，返回空结果")
            return self._get_empty_results()

        with torch.no_grad():
            # 阶段分类
            phase_output = self.phase_model(input_tensor)
            phase_probs = torch.softmax(phase_output, dim=-1)
            phase_labels = torch.argmax(phase_probs, dim=-1)

            # 质量评分（现在输出5个维度）
            quality_scores = self.quality_model(input_tensor)  # (batch, 5)

        # 解析5维质量评分
        quality_scores_np = quality_scores.cpu().numpy()[0]  # 获取第一个样本

        results = {
            'phase_sequence': phase_labels.cpu().numpy().tolist()[0],
            'phase_distribution': {
                'ground_contact': float(torch.sum(phase_labels == 0).item() / len(phase_labels[0])),
                'flight': float(torch.sum(phase_labels == 1).item() / len(phase_labels[0])),
                'transition': float(torch.sum(phase_labels == 2).item() / len(phase_labels[0]))
            },
            # 修复：解析5个维度的评分
            'quality_score': float(quality_scores_np[0]),  # 总分
            'quality_stability': float(quality_scores_np[1]),  # 稳定性
            'quality_efficiency': float(quality_scores_np[2]),  # 效率
            'quality_form': float(quality_scores_np[3]),  # 跑姿
            'quality_rhythm': float(quality_scores_np[4]),  # 节奏
            'stability_score': self._calculate_stability_from_phases(phase_labels[0])
        }

        return results

    def _prepare_input(self, keypoints_sequence: List[Dict]) -> torch.Tensor:
        """准备模型输入"""
        valid_frames = [kp for kp in keypoints_sequence if kp['detected']]

        if len(valid_frames) < MODEL_CONFIG['sequence_length']:
            print(f"⚠️  有效帧数 {len(valid_frames)} < 最小序列长度 {MODEL_CONFIG['sequence_length']}")
            return None

        # 提取关键点坐标（归一化）
        features = []
        for kp in valid_frames[:MODEL_CONFIG['sequence_length']]:
            frame_features = []
            for landmark in kp['landmarks']:
                frame_features.extend([landmark['x_norm'], landmark['y_norm']])
            features.append(frame_features)

        # 转换为tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # 数据归一化（与训练时保持一致）
        mean = input_tensor.mean()
        std = input_tensor.std() + 1e-6
        input_tensor = (input_tensor - mean) / std

        return input_tensor

    def _calculate_stability_from_phases(self, phase_sequence: torch.Tensor) -> float:
        """从阶段序列计算稳定性"""
        # 计算阶段转换次数
        transitions = torch.sum(phase_sequence[:-1] != phase_sequence[1:]).item()
        stability = max(0, 100 - transitions * 2)
        return float(stability)

    def _get_empty_results(self) -> Dict:
        """返回空结果"""
        return {
            'phase_sequence': [],
            'phase_distribution': {'ground_contact': 0, 'flight': 0, 'transition': 0},
            'quality_score': 0.0,
            'quality_stability': 0.0,
            'quality_efficiency': 0.0,
            'quality_form': 0.0,
            'quality_rhythm': 0.0,
            'stability_score': 0.0
        }


# 模块测试代码
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("=" * 60)
    print("测试修复后的时序深度学习分析模块")
    print("=" * 60)

    # 创建模拟关键点数据
    print("\n生成模拟关键点数据...")
    mock_keypoints = []
    for i in range(35):  # 35帧
        kp = {
            'detected': True,
            'landmarks': []
        }
        for j in range(33):  # 33个关键点
            kp['landmarks'].append({
                'x_norm': np.random.rand(),
                'y_norm': np.random.rand(),
                'visibility': 0.9
            })
        mock_keypoints.append(kp)

    # 创建分析器
    print("\n初始化时序分析器...")
    analyzer = TemporalModelAnalyzer()

    # 执行分析
    print("\n执行深度学习分析...")
    results = analyzer.analyze(mock_keypoints)

    # 打印结果
    print("\n" + "=" * 60)
    print("分析结果:")
    print("=" * 60)
    print(f"质量总分: {results['quality_score']:.2f}")
    print(f"  - 稳定性: {results['quality_stability']:.2f}")
    print(f"  - 效率: {results['quality_efficiency']:.2f}")
    print(f"  - 跑姿: {results['quality_form']:.2f}")
    print(f"  - 节奏: {results['quality_rhythm']:.2f}")
    print(f"阶段稳定性: {results['stability_score']:.2f}")
    print(f"\n阶段分布:")
    for phase, ratio in results['phase_distribution'].items():
        print(f"  {phase}: {ratio * 100:.1f}%")

    print("\n✅ 模块测试完成!")