#!/usr/bin/env python
"""
深度学习模型训练脚本

支持训练：
1. Transformer阶段分类模型
2. 多尺度TCN质量评估模型
3. 联合模型（阶段分类 + 质量评估）

使用方法：
    python scripts/train_models.py --model joint --epochs 50 --batch_size 32

适用于毕业设计：基于深度学习的跑步动作视频解析与技术质量评价系统
"""

import sys
import argparse
from pathlib import Path
import time
import json

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

from config.config import CHECKPOINT_DIR, OUTPUT_DIR
from models.dataset import RunningDataset, MixedViewDataset, create_dataloaders


class Trainer:
    """模型训练器"""

    def __init__(self, model, model_name: str, device: str = 'cpu'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              lr: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 10,
              save_best: bool = True) -> dict:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            save_best: 是否保存最佳模型

        Returns:
            训练历史
        """
        # 优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # 学习率调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        # 早停
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\n{'='*70}")
        print(f"开始训练: {self.model_name}")
        print(f"{'='*70}")
        print(f"设备: {self.device}")
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
        print(f"批次大小: {train_loader.batch_size}")
        print(f"学习率: {lr}")
        print(f"训练轮数: {epochs}")
        print(f"{'='*70}\n")

        start_time = time.time()

        for epoch in range(epochs):
            # 训练阶段
            train_loss, train_acc = self._train_epoch(train_loader, optimizer)

            # 验证阶段
            val_loss, val_acc = self._validate_epoch(val_loader)

            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            # 打印进度
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self._save_checkpoint(f'{self.model_name}_best.pth')
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= patience:
                print(f"\n早停触发！验证损失连续 {patience} 轮未改善")
                break

        # 保存最终模型
        self._save_checkpoint(f'{self.model_name}.pth')

        # 保存训练历史
        self._save_history()

        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"训练完成!")
        print(f"总用时: {elapsed_time/60:.2f} 分钟")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"{'='*70}")

        return self.history

    def _train_epoch(self, train_loader: DataLoader, optimizer) -> tuple:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            keypoints, phase_labels, quality_scores, view_ids = batch
            keypoints = keypoints.to(self.device)
            phase_labels = phase_labels.to(self.device)
            quality_scores = quality_scores.to(self.device)
            view_ids = view_ids.to(self.device)

            optimizer.zero_grad()

            # 前向传播
            loss, acc = self._compute_loss_and_acc(
                keypoints, phase_labels, quality_scores, view_ids
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += acc * len(keypoints)
            total += len(keypoints)

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total * 100

        return avg_loss, avg_acc

    def _validate_epoch(self, val_loader: DataLoader) -> tuple:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                keypoints, phase_labels, quality_scores, view_ids = batch
                keypoints = keypoints.to(self.device)
                phase_labels = phase_labels.to(self.device)
                quality_scores = quality_scores.to(self.device)
                view_ids = view_ids.to(self.device)

                loss, acc = self._compute_loss_and_acc(
                    keypoints, phase_labels, quality_scores, view_ids
                )

                total_loss += loss.item()
                correct += acc * len(keypoints)
                total += len(keypoints)

        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / total * 100

        return avg_loss, avg_acc

    def _compute_loss_and_acc(self, keypoints, phase_labels, quality_scores, view_ids):
        """计算损失和准确率 - 需要子类实现"""
        raise NotImplementedError

    def _save_checkpoint(self, filename: str):
        """保存模型权重"""
        checkpoint_path = CHECKPOINT_DIR / filename
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"  💾 保存模型: {checkpoint_path}")

    def _save_history(self):
        """保存训练历史"""
        history_path = OUTPUT_DIR / f'{self.model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


class TransformerTrainer(Trainer):
    """Transformer模型训练器"""

    def __init__(self, device: str = 'cpu'):
        from models.transformer_model import RunningPhaseTransformer

        model = RunningPhaseTransformer(
            d_model=128,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        )

        super().__init__(model, 'transformer_phase_model', device)
        self.criterion = nn.CrossEntropyLoss()

    def _compute_loss_and_acc(self, keypoints, phase_labels, quality_scores, view_ids):
        """计算损失和准确率"""
        logits = self.model(keypoints, view_ids)  # (batch, seq, 3)

        # 重塑用于交叉熵
        logits_flat = logits.reshape(-1, 3)
        labels_flat = phase_labels.reshape(-1)

        loss = self.criterion(logits_flat, labels_flat)

        # 计算准确率
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == phase_labels).float().mean().item()

        return loss, acc


class QualityModelTrainer(Trainer):
    """质量评估模型训练器"""

    def __init__(self, device: str = 'cpu'):
        from models.quality_model import RunningQualityModel

        model = RunningQualityModel(
            hidden_dim=128,
            num_levels=4,
            dropout=0.2
        )

        super().__init__(model, 'quality_tcn_model', device)
        self.criterion = nn.MSELoss()

    def _compute_loss_and_acc(self, keypoints, phase_labels, quality_scores, view_ids):
        """计算损失和准确率"""
        outputs = self.model(keypoints, view_ids)
        pred_scores = outputs['scores']

        loss = self.criterion(pred_scores, quality_scores)

        # 计算"准确率"（预测误差小于5分的比例）
        error = torch.abs(pred_scores - quality_scores)
        acc = (error < 5).float().mean().item()

        return loss, acc


class JointModelTrainer(Trainer):
    """联合模型训练器"""

    def __init__(self, device: str = 'cpu'):
        from models.quality_model import JointPhaseQualityModel

        model = JointPhaseQualityModel(
            hidden_dim=128,
            num_levels=4,
            dropout=0.2
        )

        super().__init__(model, 'joint_model', device)
        self.phase_criterion = nn.CrossEntropyLoss()
        self.quality_criterion = nn.MSELoss()

        # 损失权重
        self.phase_weight = 1.0
        self.quality_weight = 0.5

    def _compute_loss_and_acc(self, keypoints, phase_labels, quality_scores, view_ids):
        """计算损失和准确率"""
        outputs = self.model(keypoints, view_ids)

        # 阶段分类损失
        phase_logits = outputs['phase_logits'].reshape(-1, 3)
        phase_labels_flat = phase_labels.reshape(-1)
        phase_loss = self.phase_criterion(phase_logits, phase_labels_flat)

        # 质量评估损失
        quality_loss = self.quality_criterion(outputs['quality_scores'], quality_scores)

        # 总损失
        loss = self.phase_weight * phase_loss + self.quality_weight * quality_loss

        # 计算准确率（使用阶段分类准确率）
        preds = torch.argmax(outputs['phase_logits'], dim=-1)
        acc = (preds == phase_labels).float().mean().item()

        return loss, acc


def train_all_models(args):
    """训练所有模型"""
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"使用设备: {device}")

    # 创建数据加载器
    print("\n创建数据集...")
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        num_workers=args.num_workers
    )

    models_to_train = []

    if args.model in ['all', 'transformer']:
        models_to_train.append(('Transformer阶段分类', TransformerTrainer(device)))

    if args.model in ['all', 'quality']:
        models_to_train.append(('TCN质量评估', QualityModelTrainer(device)))

    if args.model in ['all', 'joint']:
        models_to_train.append(('联合模型', JointModelTrainer(device)))

    # 训练每个模型
    results = {}
    for name, trainer in models_to_train:
        print(f"\n\n{'#'*70}")
        print(f"# 训练模型: {name}")
        print(f"{'#'*70}")

        history = trainer.train(
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience
        )
        results[name] = history

    # 打印总结
    print(f"\n\n{'='*70}")
    print("训练总结")
    print('='*70)
    for name, history in results.items():
        best_val_loss = min(history['val_loss'])
        best_val_acc = max(history['val_acc'])
        print(f"{name}:")
        print(f"  最佳验证损失: {best_val_loss:.4f}")
        print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print('='*70)


def main():
    parser = argparse.ArgumentParser(description='训练跑步分析深度学习模型')

    # 模型选择
    parser.add_argument('--model', type=str, default='joint',
                        choices=['transformer', 'quality', 'joint', 'all'],
                        help='要训练的模型类型')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')

    # 数据参数
    parser.add_argument('--num_train', type=int, default=2000, help='训练样本数')
    parser.add_argument('--num_val', type=int, default=500, help='验证样本数')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')

    # 其他
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 开始训练
    train_all_models(args)


if __name__ == '__main__':
    main()
