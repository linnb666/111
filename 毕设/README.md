# 基于深度学习的跑步动作视频解析与技术质量评价系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)](https://mediapipe.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

本项目是一个完整的跑步动作分析系统，基于深度学习技术，能够从普通RGB视频中自动提取跑步动作特征，进行技术质量评价，并生成详细的分析报告。

### 核心特性

- ✅ **深度学习姿态估计**: 基于MediaPipe Pose的实时人体关键点检测
- ✅ **时序深度学习**: LSTM/CNN模型进行跑步阶段识别和质量评估
- ✅ **运动学分析**: 自动计算关节角度、步频、步态周期等指标
- ✅ **技术质量评价**: 多维度综合评分系统
- ✅ **AI文本分析**: 支持接入大模型API生成智能分析报告
- ✅ **模块化设计**: 完全解耦的架构，易于扩展和维护
- ✅ **Web界面**: Streamlit快速展示 + Flask RESTful API
- ✅ **数据持久化**: SQLite数据库存储历史记录

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      输入层                                  │
│                  跑步视频(MP4/AVI/MOV)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                   1️⃣ 视频预处理模块                          │
│          帧抽取 | 分辨率统一 | 基本信息解析                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│            2️⃣ 深度学习姿态估计模块 (MediaPipe)               │
│              33个2D关键点 | 火柴人可视化                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────┴────────┐ ┌───┴────────┐ ┌──┴──────────────┐
│ 3️⃣ 运动学分析  │ │ 4️⃣ 时序DL   │ │ 可视化工具      │
│ 角度|步频|振幅 │ │ LSTM|CNN   │ │ 火柴人|曲线图   │
└───────┬────────┘ └───┬────────┘ └─────────────────┘
        │              │
        └──────┬───────┘
               │
┌──────────────┴──────────────────────────────────────────────┐
│                5️⃣ 技术质量评价模块                           │
│     稳定性 | 效率 | 跑姿标准度 | 节奏一致性                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────┴────────┐ ┌───┴────────┐ ┌──┴──────────────┐
│ 6️⃣ AI文本分析  │ │ 7️⃣ 数据库   │ │ 8️⃣ API/Web界面 │
│ 智能报告生成  │ │ SQLite存储 │ │ Flask|Streamlit│
└────────────────┘ └────────────┘ └─────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyCharm / VS Code / Jupyter
- 8GB+ RAM
- （可选）NVIDIA GPU + CUDA

### 安装步骤

1️⃣ **克隆项目**

```bash
git clone https://github.com/yourusername/running_analysis_system.git
cd running_analysis_system
```

2️⃣ **创建虚拟环境**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3️⃣ **安装依赖**

```bash
pip install -r requirements.txt
```

4️⃣ **配置环境变量（可选）**

```bash
cp .env.example .env
# 编辑.env文件，配置AI API密钥
```

5️⃣ **训练模型（首次运行）**

```bash
python train_model.py
```

### 使用方式

#### 方式一：命令行运行

```bash
# 基本分析
python main.py path/to/your/running_video.mp4

# 生成可视化结果
python main.py video.mp4 --visualize --save-db

# 指定输出目录
python main.py video.mp4 --output ./results --visualize
```

#### 方式二：Streamlit Web界面

```bash
streamlit run web/streamlit_app.py
```

访问 `http://localhost:8501`

#### 方式三：Flask API服务

```bash
python api/api_server.py
```

API端点:
- `POST /api/analyze_video` - 上传视频分析
- `GET /api/result/<id>` - 获取分析结果
- `GET /api/history` - 历史记录
- `GET /api/statistics` - 统计信息

#### 方式四：在代码中调用

```python
from modules.video_processor import VideoProcessor
from modules.pose_estimator import PoseEstimator
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.temporal_model import TemporalModelAnalyzer
from modules.quality_evaluator import QualityEvaluator

# 完整分析流程
processor = VideoProcessor('video.mp4')
frames, fps = processor.extract_frames()

estimator = PoseEstimator()
keypoints = estimator.process_frames(frames)

kinematic_analyzer = KinematicAnalyzer()
kinematic_results = kinematic_analyzer.analyze_sequence(keypoints, fps)

temporal_analyzer = TemporalModelAnalyzer()
temporal_results = temporal_analyzer.analyze(keypoints)

evaluator = QualityEvaluator()
quality_results = evaluator.evaluate(kinematic_results, temporal_results)

print(f"技术质量评分: {quality_results['total_score']:.2f}")
```

## 📊 输出结果

系统输出包括：

1. **火柴人可视化** - 姿态识别结果
2. **运动学指标**
   - 步频（cadence）
   - 膝关节/髋关节角度
   - 垂直振幅
   - 步态周期
3. **技术质量评分**
   - 总体评分（0-100）
   - 四维度评分：稳定性、效率、跑姿、节奏
   - 评级：优秀/良好/一般/待改进
4. **深度学习分析**
   - 跑步阶段识别
   - AI质量评分
   - 动作稳定性评估
5. **AI智能报告** - 文字形式的专业建议

## 🧪 模块测试

每个核心模块都支持独立测试：

```bash
# 测试视频处理
python modules/video_processor.py video.mp4

# 测试姿态估计
python modules/pose_estimator.py video.mp4

# 测试运动学分析
python modules/kinematic_analyzer.py

# 测试深度学习模型
python modules/temporal_model.py
```

## 📁 项目结构

```
running_analysis_system/
├── config/                      # 配置文件
│   ├── __init__.py
│   └── config.py               # 系统配置
├── modules/                     # 核心模块
│   ├── video_processor.py      # 视频预处理
│   ├── pose_estimator.py       # 姿态估计
│   ├── kinematic_analyzer.py   # 运动学分析
│   ├── temporal_model.py       # 时序深度学习
│   ├── quality_evaluator.py    # 质量评价
│   ├── ai_analyzer.py          # AI文本分析
│   └── database.py             # 数据库管理
├── models/                      # 模型定义
│   ├── lstm_model.py           # LSTM模型
│   └── cnn_model.py            # CNN模型（与LSTM在同一文件）
├── utils/                       # 工具函数
│   └── visualization.py        # 可视化工具
├── api/                         # API服务
│   └── api_server.py           # Flask API
├── web/                         # Web界面
│   └── streamlit_app.py        # Streamlit界面
├── data/                        # 数据目录
│   ├── checkpoints/            # 模型权重
│   └── database.db             # SQLite数据库
├── output/                      # 输出目录
│   ├── videos/                 # 输出视频
│   └── visualizations/         # 可视化结果
├── tests/                       # 测试文件
├── main.py                      # 主程序入口
├── train_model.py              # 模型训练
├── requirements.txt            # 依赖列表
├── .env.example                # 环境变量示例
└── README.md                   # 项目文档
```

## 🔧 配置说明

主要配置文件：`config/config.py`

```python
# 视频处理配置
VIDEO_CONFIG = {
    'target_width': 640,
    'target_height': 480,
    'fps': 30
}

# MediaPipe配置
POSE_CONFIG = {
    'model_complexity': 1,
    'min_detection_confidence': 0.5
}

# 深度学习模型配置
MODEL_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'sequence_length': 30
}

# AI分析配置
AI_CONFIG = {
    'enabled': False,  # 启用AI增强
    'provider': 'openai',
    'api_key': os.getenv('AI_API_KEY')
}
```

## 🎓 论文对应关系

本项目结构清晰映射到毕业论文章节：

| 论文章节 | 对应模块 | 技术要点 |
|---------|---------|---------|
| 第2章 相关技术 | `config/`, `models/` | MediaPipe、LSTM、CNN |
| 第3章 系统设计 | 所有模块接口 | 模块化架构设计 |
| 3.1 视频处理 | `video_processor.py` | OpenCV、帧抽取 |
| 3.2 姿态估计 | `pose_estimator.py` | MediaPipe深度学习 |
| 3.3 特征提取 | `kinematic_analyzer.py` | 运动学计算 |
| 3.4 深度学习 | `temporal_model.py` | LSTM/CNN时序分析 |
| 3.5 质量评价 | `quality_evaluator.py` | 多维度评分模型 |
| 第4章 系统实现 | `main.py`, `api_server.py` | 完整系统集成 |
| 第5章 测试分析 | `tests/`, 实验结果 | 性能测试、案例分析 |

## 🔬 核心算法

### 1. MediaPipe Pose（深度学习姿态估计）

```python
# 使用预训练的BlazePose模型
# 33个2D关键点检测
# 实时推理性能：30+ FPS
```

### 2. LSTM时序分类

```python
# 输入: 关键点时间序列 (batch, seq_len, 66)
# 输出: 跑步阶段 (触地/腾空/过渡)
# 结构: Bi-LSTM + 全连接层
```

### 3. 1D-CNN质量评估

```python
# 输入: 关键点时间序列
# 输出: 质量评分 (0-100)
# 结构: 多层1D卷积 + 池化 + 全连接
```

### 4. 运动学计算

- **关节角度**: 三点向量夹角计算
- **步频**: 基于峰值检测的步态周期识别
- **垂直振幅**: 髋部中心Y坐标变化
- **稳定性**: 关键点位置标准差

## 📈 扩展建议

### 后期可扩展功能

1. **Web前端**: Vue.js / React开发完整前端界面
2. **移动App**: Flutter / React Native开发移动应用
3. **多角度分析**: 支持前、侧、后多视角融合
4. **实时分析**: WebRTC实现在线视频流分析
5. **对比功能**: 与专业运动员动作对比
6. **训练计划**: 基于分析结果生成个性化训练方案
7. **社交功能**: 用户社区、数据分享
8. **增强模型**: 更大规模数据集训练，提升准确性

### API扩展示例

```python
# 1. 对接移动App
POST /api/mobile/analyze
{
    "video_base64": "...",
    "user_id": "123"
}

# 2. 批量处理
POST /api/batch/analyze
{
    "video_urls": ["url1", "url2"],
    "callback_url": "..."
}

# 3. 实时WebSocket
WS /api/realtime/analyze
```

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 开发日志

- **v1.0.0** (2024-01) - 初始版本发布
  - ✅ 完成9大核心模块
  - ✅ 实现深度学习分析
  - ✅ 添加Web界面
  - ✅ 支持数据库存储

## ⚠️ 注意事项

1. **视频质量**: 建议使用侧面拍摄、清晰稳定的视频
2. **人物遮挡**: MediaPipe对严重遮挡的情况检测效果较差
3. **光照条件**: 避免逆光、过暗的环境
4. **运行环境**: 首次运行会下载MediaPipe模型文件
5. **GPU加速**: 安装CUDA版PyTorch可显著提升性能

## 🐛 故障排除

### 常见问题

**Q1: MediaPipe导入失败**
```bash
# 重新安装
pip uninstall mediapipe
pip install mediapipe==0.10.3
```

**Q2: PyTorch安装问题**
```bash
# CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU版本（CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Q3: 视频无法打开**
```bash
# 检查OpenCV是否正确安装
python -c "import cv2; print(cv2.__version__)"
```

## 📚 参考资料

- [MediaPipe官方文档](https://mediapipe.dev/)
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [运动生物力学基础](https://example.com)

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👥 作者

- **姓名** - 计算机科学与技术专业
- **导师** - XXX教授
- **学校** - XXX大学

## 🙏 致谢

感谢以下开源项目：
- MediaPipe - Google
- PyTorch - Facebook AI
- OpenCV - OpenCV Foundation
