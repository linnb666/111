# train_model.py (改进版)
"""
改进的模型训练脚本
使用更好的训练策略和数据增强
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from models.lstm_model import RunningPhaseLSTM
from models.cnn_model import RunningQualityCNN, ImprovedRunningDataset
from config.config import MODEL_CONFIG, CHECKPOINT_DIR


def train_phase_model_improved(epochs=100, batch_size=32, learning_rate=0.001):
    """
    改进的阶段分类模型训练
    增加学习率调度、早停等机制
    """
    print("\n" + "=" * 80)
    print("训练改进版阶段分类模型（LSTM + Attention）")
    print("=" * 80)

    # 准备数据
    train_dataset = ImprovedRunningDataset(num_samples=2000)
    val_dataset = ImprovedRunningDataset(num_samples=400)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = RunningPhaseLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 学习率调度器（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_data, batch_phase, _ in train_loader:
            batch_data = batch_data.to(device)
            batch_phase = batch_phase.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)

            # 重塑用于计算loss
            outputs_flat = outputs.reshape(-1, MODEL_CONFIG['output_dim'])
            batch_phase_flat = batch_phase.reshape(-1)

            loss = criterion(outputs_flat, batch_phase_flat)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs_flat, 1)
            train_total += batch_phase_flat.size(0)
            train_correct += (predicted == batch_phase_flat).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data, batch_phase, _ in val_loader:
                batch_data = batch_data.to(device)
                batch_phase = batch_phase.to(device)

                outputs = model(batch_data)

                outputs_flat = outputs.reshape(-1, MODEL_CONFIG['output_dim'])
                batch_phase_flat = batch_phase.reshape(-1)

                loss = criterion(outputs_flat, batch_phase_flat)
                val_loss += loss.item()

                _, predicted = torch.max(outputs_flat, 1)
                val_total += batch_phase_flat.size(0)
                val_correct += (predicted == batch_phase_flat).sum().item()

        # 计算平均值
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 学习率调度
        scheduler.step()

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'phase_model.pth')
            best_epoch = epoch + 1
        else:
            patience_counter += 1

        # 打印进度
        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发！最佳模型在 Epoch {best_epoch}")
            break

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 'phase_model')

    print(f"\n✅ 训练完成！最佳验证Loss: {best_val_loss:.4f}")
    print(f"模型已保存: {CHECKPOINT_DIR / 'phase_model.pth'}")

    return train_losses, val_losses


def train_quality_model_improved(epochs=100, batch_size=32, learning_rate=0.001):
    """
    改进的质量评估模型训练
    多任务学习（同时预测5个维度）
    """
    print("\n" + "=" * 80)
    print("训练改进版质量评估模型（Multi-scale CNN + Attention）")
    print("=" * 80)

    # 准备数据
    train_dataset = ImprovedRunningDataset(num_samples=2000)
    val_dataset = ImprovedRunningDataset(num_samples=400)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = RunningQualityCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练历史
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0

        for batch_data, _, batch_quality in train_loader:
            batch_data = batch_data.to(device)
            batch_quality = batch_quality.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)

            # 多任务损失
            loss = criterion(outputs, batch_quality)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_data, _, batch_quality in val_loader:
                batch_data = batch_data.to(device)
                batch_quality = batch_quality.to(device)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_quality)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step()

        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'quality_model.pth')
            best_epoch = epoch + 1
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if patience_counter >= patience:
            print(f"\n早停触发！最佳模型在 Epoch {best_epoch}")
            break

    # 绘制训练曲线
    plot_loss_curves(train_losses, val_losses, 'quality_model')

    print(f"\n✅ 训练完成！最佳验证Loss: {best_val_loss:.4f}")
    print(f"模型已保存: {CHECKPOINT_DIR / 'quality_model.pth'}")

    return train_losses, val_losses


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy曲线
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / f'{model_name}_training_curves.png', dpi=150)
    print(f"训练曲线已保存: {CHECKPOINT_DIR / f'{model_name}_training_curves.png'}")
    plt.close()


def plot_loss_curves(train_losses, val_losses, model_name):
    """绘制Loss曲线"""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / f'{model_name}_loss_curves.png', dpi=150)
    print(f"Loss曲线已保存: {CHECKPOINT_DIR / f'{model_name}_loss_curves.png'}")
    plt.close()


def evaluate_models():
    """评估训练好的模型"""
    print("\n" + "=" * 80)
    print("评估训练好的模型")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    phase_model = RunningPhaseLSTM().to(device)
    quality_model = RunningQualityCNN().to(device)

    phase_model.load_state_dict(torch.load(CHECKPOINT_DIR / 'phase_model.pth', map_location=device))
    quality_model.load_state_dict(torch.load(CHECKPOINT_DIR / 'quality_model.pth', map_location=device))

    phase_model.eval()
    quality_model.eval()

    # 测试数据
    test_dataset = ImprovedRunningDataset(num_samples=100)
    test_loader = DataLoader(test_dataset, batch_size=10)

    phase_correct = 0
    phase_total = 0
    quality_errors = []

    with torch.no_grad():
        for batch_data, batch_phase, batch_quality in test_loader:
            batch_data = batch_data.to(device)
            batch_phase = batch_phase.to(device)
            batch_quality = batch_quality.to(device)

            # 阶段分类评估
            phase_outputs = phase_model(batch_data)
            _, predicted = torch.max(phase_outputs.reshape(-1, 3), 1)
            phase_total += batch_phase.reshape(-1).size(0)
            phase_correct += (predicted == batch_phase.reshape(-1)).sum().item()

            # 质量评估
            quality_outputs = quality_model(batch_data)
            errors = torch.abs(quality_outputs - batch_quality).mean(dim=0)
            quality_errors.append(errors.cpu().numpy())

    phase_accuracy = 100 * phase_correct / phase_total
    avg_quality_errors = np.mean(quality_errors, axis=0)

    print(f"\n阶段分类准确率: {phase_accuracy:.2f}%")
    print(f"\n质量评估平均误差:")
    print(f"  总分: {avg_quality_errors[0]:.2f}")
    print(f"  稳定性: {avg_quality_errors[1]:.2f}")
    print(f"  效率: {avg_quality_errors[2]:.2f}")
    print(f"  跑姿: {avg_quality_errors[3]:.2f}")
    print(f"  节奏: {avg_quality_errors[4]:.2f}")


def main():
    """主训练函数"""
    print("=" * 80)
    print("改进版深度学习模型训练脚本")
    print("=" * 80)
    print("\n特性:")
    print("  ✓ 基于生物力学规则的数据生成")
    print("  ✓ 注意力机制增强特征提取")
    print("  ✓ 学习率调度和早停机制")
    print("  ✓ 多任务学习（5个质量维度）")
    print("  ✓ 训练曲线可视化")
    print("=" * 80)

    # 创建目录
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)

        # 训练模型
        train_phase_model_improved(epochs=100, batch_size=32, learning_rate=0.001)
        train_quality_model_improved(epochs=100, batch_size=32, learning_rate=0.001)

        # 评估模型
        evaluate_models()

        print("\n" + "=" * 80)
        print("✅ 所有模型训练完成！")
        print("=" * 80)
        print("\n生成的文件:")
        print(f"  - {CHECKPOINT_DIR / 'phase_model.pth'}")
        print(f"  - {CHECKPOINT_DIR / 'quality_model.pth'}")
        print(f"  - {CHECKPOINT_DIR / 'phase_model_training_curves.png'}")
        print(f"  - {CHECKPOINT_DIR / 'quality_model_loss_curves.png'}")
        print("\n现在可以运行完整系统:")
        print("  python main.py test_video.mp4")
        print("  streamlit run web/streamlit_app.py")

    except KeyboardInterrupt:
        print("\n\n⚠️  训练被中断")
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()