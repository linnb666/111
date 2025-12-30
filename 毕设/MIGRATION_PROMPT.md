# 跑步动作分析系统 - MMPose 3D 重构方案

## 📋 项目背景

### 项目名称
**基于深度学习的跑步动作视频解析与技术质量评价系统的设计与实现**

### 项目性质
本科毕业设计

### 当前问题
原系统使用 MediaPipe 2D 姿态估计，存在以下核心问题：
1. **膝关节角度不准确**：2D投影导致角度失真，显示99°-130°（实际应为155-170°）
2. **落地检测不可靠**：基于ankle_y的检测在触地附近误差大
3. **数据可信度低**：多次调试后大部分指标仍不准确
4. **视角依赖性强**：不同拍摄角度导致结果差异大

### 解决方案
迁移到 **MMPose + MotionBERT** 3D姿态估计架构，获得真实3D关节坐标，从根本上解决2D投影失真问题。

---

## 🎯 重构目标

### 核心原则
> **"宁可处理慢一点，也要数据可信"**
> **"不追求SOTA，追求方法正确、数据可信、结构清晰"**

### 具体目标
1. 使用3D坐标计算膝关节角度，消除透视失真
2. 使用foot_z + z速度检测落地，比y_norm更可靠
3. 保持深度学习主题（这是毕设核心）
4. 保留现有系统架构思想和评分维度设计
5. 支持离线分析（非实时），优先保证准确性

---

## 💻 硬件环境约束

```
GPU: NVIDIA RTX 2050
显存: 4GB（较紧张，需要优化策略）
运行方式: 本地离线分析

可接受的优化策略:
- 降低输入FPS（如30fps→15fps）
- 分段处理长视频
- 使用半精度(FP16)推理
- 批处理优化
```

---

## 📁 原项目结构（供参考）

```
毕设/
├── config/
│   └── config.py                 # 配置文件（保留结构，更新参数）
├── modules/
│   ├── video_processor.py        # 视频处理（保留）
│   ├── pose_estimator.py         # 姿态估计（完全重写→MMPose）
│   ├── view_detector.py          # 视角检测（3D后可简化）
│   ├── kinematic_analyzer.py     # 运动学分析（核心重写→3D算法）
│   ├── temporal_model.py         # 时序深度学习（保留/调整）
│   ├── quality_evaluator.py      # 质量评价（保留框架，更新指标）
│   ├── ai_analyzer.py            # AI文本分析（保留）
│   └── database.py               # 数据库（保留，字段升级）
├── models/
│   ├── lstm_model.py             # LSTM模型（可选保留）
│   ├── transformer_model.py      # Transformer（可选保留）
│   ├── cnn_model.py              # CNN模型
│   ├── quality_model.py          # 质量评估模型
│   └── dataset.py                # 数据集生成
├── web/
│   └── streamlit_app.py          # Web界面（保留，更新显示）
├── api/
│   └── api_server.py             # Flask API（保留）
├── utils/
│   └── visualization.py          # 可视化（扩展3D可视化）
└── main.py                       # 主入口
```

---

## 🔄 保留 vs 重构 清单

### ✅ 明确保留（不要删除或大改）

| 模块 | 文件 | 保留原因 |
|------|------|----------|
| 系统架构思想 | 整体 | 视频→姿态→特征→评分→文本 的pipeline |
| 评分维度设计 | quality_evaluator.py | 稳定性/效率/跑姿 三维度 |
| AI文本分析 | ai_analyzer.py | 智谱AI + 本地规则引擎 |
| 视频处理 | video_processor.py | 帧提取、基本信息 |
| 数据库结构 | database.py | SQLite记录管理 |
| Web界面框架 | streamlit_app.py | Streamlit UI结构 |
| 深度学习主题 | models/*.py | 毕设核心主题 |

### ❌ 完全替换

| 模块 | 原实现 | 新实现 |
|------|--------|--------|
| 姿态估计 | MediaPipe 2D | MMPose + MotionBERT 3D |
| 落地检测 | ankle_y 峰值 | foot_z + z速度 |
| 膝角计算 | 2D投影角度 | 3D真实关节角度 |
| 视角检测 | 复杂规则 | 3D后可大幅简化 |

### 🔄 需要重构

| 模块 | 改动范围 | 说明 |
|------|----------|------|
| kinematic_analyzer.py | 核心重写 | 所有算法改为3D版本 |
| temporal_model.py | 输入适配 | 输入从66维改为99维(33×3) |
| quality_evaluator.py | 指标更新 | 评分逻辑适配3D指标 |
| config.py | 参数更新 | 新增MMPose配置 |

---

## 🏗️ 新架构设计

### 数据流架构

```
视频输入 (MP4)
    ↓
┌─────────────────────────────────────────┐
│ 1. 视频预处理 (VideoProcessor)          │
│    - 帧提取、分辨率调整                  │
│    - 降采样优化(30fps→15fps)             │
└─────────────────────┬───────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. 2D姿态估计 (MMPose 2D)               │
│    - HRNet / RTMPose                     │
│    - 输出: 17个COCO关键点 × T帧          │
└─────────────────────┬───────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. 2D→3D提升 (MotionBERT)               │  ⭐ 核心深度学习
│    - 时序Transformer                     │
│    - 输入: 2D序列 (T, 17, 2)             │
│    - 输出: 3D序列 (T, 17, 3)             │
└─────────────────────┬───────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. 3D运动学分析 (KinematicAnalyzer3D)   │  ⭐ 核心算法
│    - 3D膝关节角度（无透视失真）          │
│    - 3D落地检测（foot_z + z速度）        │
│    - 3D步态对称性                        │
│    - COM垂直振幅                         │
└─────────────────────┬───────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. 质量评价 (QualityEvaluator)          │
│    - 稳定性: 35%                         │
│    - 效率: 35%                           │
│    - 跑姿: 30%                           │
└─────────────────────┬───────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 6. AI文本分析 (智谱AI / 本地规则)        │
└─────────────────────────────────────────┘
    ↓
输出: 评分 + 报告 + 3D可视化
```

### 新目录结构

```
毕设/
├── config/
│   ├── __init__.py
│   └── config.py                    # 更新：新增MMPose/MotionBERT配置
│
├── modules/
│   ├── __init__.py
│   ├── video_processor.py           # 保留：视频处理
│   ├── pose_estimator_3d.py         # 新建：MMPose + MotionBERT封装
│   ├── kinematic_analyzer_3d.py     # 新建：3D运动学分析（核心）
│   ├── landing_detector_3d.py       # 新建：3D落地检测（独立模块）
│   ├── gait_analyzer_3d.py          # 新建：3D步态分析
│   ├── quality_evaluator.py         # 重构：适配3D指标
│   ├── ai_analyzer.py               # 保留：AI文本分析
│   └── database.py                  # 保留：数据库（字段升级）
│
├── models/
│   ├── __init__.py
│   ├── motionbert/                  # 新建：MotionBERT模型目录
│   │   ├── model.py                 # 模型定义
│   │   ├── inference.py             # 推理封装
│   │   └── checkpoints/             # 预训练权重
│   ├── phase_classifier.py          # 可选：步态阶段分类器
│   └── quality_model.py             # 保留/调整：质量评估模型
│
├── web/
│   ├── streamlit_app.py             # 重构：更新UI显示3D结果
│   └── components/                  # 新建：UI组件
│       ├── video_player.py
│       ├── metrics_display.py
│       └── pose_viewer_3d.py        # 新建：3D骨架查看器
│
├── utils/
│   ├── visualization.py             # 扩展：3D可视化
│   ├── angle_utils.py               # 新建：3D角度计算工具
│   └── coordinate_utils.py          # 新建：坐标系转换工具
│
├── data/
│   ├── checkpoints/                 # 模型权重
│   └── database.db                  # SQLite数据库
│
├── tests/                           # 新建：测试用例
│   ├── test_pose_3d.py
│   └── test_kinematic.py
│
├── requirements.txt                 # 更新：新增依赖
├── main.py                          # 主入口
└── README.md                        # 更新：新文档
```

---

## 🔧 关键模块设计规格

### 模块1: pose_estimator_3d.py

```python
"""
3D姿态估计模块
封装 MMPose 2D + MotionBERT 3D 流程
"""

class PoseEstimator3D:
    """
    核心类：3D姿态估计器

    处理流程:
    1. 使用MMPose提取2D关键点（17个COCO格式）
    2. 使用MotionBERT将2D序列提升为3D

    优化策略（针对4GB显存）:
    - 输入降采样至15fps
    - 使用FP16推理
    - 分段处理（每段243帧≈16秒@15fps）
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.detector_2d = None  # MMPose 2D检测器
        self.lifter_3d = None    # MotionBERT 2D→3D

    def process_video(self, video_path: str) -> Dict:
        """
        处理整个视频

        Returns:
            {
                'poses_2d': np.ndarray,  # (T, 17, 2)
                'poses_3d': np.ndarray,  # (T, 17, 3)
                'fps': float,
                'frame_count': int,
                'confidence': np.ndarray  # (T, 17)
            }
        """
        pass

    def _init_2d_detector(self):
        """初始化MMPose 2D检测器（RTMPose推荐，速度快）"""
        pass

    def _init_3d_lifter(self):
        """初始化MotionBERT 3D提升器"""
        pass

    def _optimize_for_low_memory(self):
        """4GB显存优化策略"""
        # - torch.cuda.empty_cache()
        # - 使用torch.no_grad()
        # - FP16推理
        # - 分段处理
        pass

# 关键点定义（COCO 17点 → 与MotionBERT兼容）
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# 关键点索引（用于运动学分析）
JOINT_INDICES = {
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
    'left_shoulder': 5, 'right_shoulder': 6
}
```

### 模块2: kinematic_analyzer_3d.py

```python
"""
3D运动学分析模块（核心）
基于真实3D坐标计算所有生物力学指标
"""

class KinematicAnalyzer3D:
    """
    3D运动学分析器

    核心优势（相比2D）:
    1. 膝关节角度无透视失真
    2. 落地检测基于真实z坐标
    3. 步态对称性基于3D轨迹
    4. COM垂直振幅基于真实高度
    """

    def analyze(self, poses_3d: np.ndarray, fps: float) -> Dict:
        """
        完整3D运动学分析

        Args:
            poses_3d: (T, 17, 3) 3D关键点序列
            fps: 帧率

        Returns:
            {
                'knee_angles': {...},      # 3D膝关节角度
                'landing_detection': {...}, # 3D落地检测
                'gait_symmetry': {...},    # 步态对称性
                'vertical_oscillation': {...}, # COM垂直振幅
                'cadence': {...},          # 步频
                'trunk_lean': {...},       # 躯干前倾
                'hip_extension': {...},    # 髋关节伸展
            }
        """
        pass

    # ============ 3D膝关节角度 ============
    def calculate_knee_angle_3d(self, hip: np.ndarray, knee: np.ndarray,
                                 ankle: np.ndarray) -> float:
        """
        计算3D膝关节角度

        优势: 不受相机视角影响，直接计算真实角度

        公式:
            vec1 = hip - knee
            vec2 = ankle - knee
            angle = arccos(dot(vec1, vec2) / (|vec1| * |vec2|))

        返回: 角度（度），范围0-180
        """
        vec1 = hip - knee
        vec2 = ankle - knee
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # ============ 3D落地检测 ============
    def detect_landing_3d(self, poses_3d: np.ndarray, fps: float) -> List[Dict]:
        """
        基于3D的落地检测（核心改进）

        原理:
        1. 提取foot_z（脚踝z坐标）
        2. 计算z速度: v_z = d(foot_z)/dt
        3. 落地时刻: foot_z达到最低点 且 v_z从负变正

        优势:
        - z坐标直接表示高度，比2D的y_norm更准确
        - 不受相机角度影响
        - 速度变化更可靠地标识着地时刻

        Returns:
            [
                {
                    'frame': int,           # 落地帧
                    'foot': 'left'|'right', # 哪只脚
                    'foot_z': float,        # 脚踝z坐标
                    'velocity_z': float,    # z方向速度
                    'knee_angle': float,    # 落地时膝角
                    'confidence': float     # 置信度
                },
                ...
            ]
        """
        pass

    # ============ 步态对称性（3D） ============
    def analyze_gait_symmetry_3d(self, poses_3d: np.ndarray) -> Dict:
        """
        3D步态对称性分析

        指标:
        1. 左右膝角对称性（相关系数）
        2. 左右步幅对称性
        3. 左右触地时间对称性
        4. 骨盆左右位移（正面视角核心指标）

        Returns:
            {
                'knee_symmetry': float,      # 0-1
                'stride_symmetry': float,
                'contact_time_symmetry': float,
                'pelvic_displacement': float, # 骨盆横向位移（mm或归一化）
                'overall_symmetry': float
            }
        """
        pass

    # ============ 膝内扣/外翻检测（正面核心） ============
    def detect_knee_valgus_3d(self, poses_3d: np.ndarray) -> Dict:
        """
        检测膝内扣(valgus)和膝外翻(varus)

        原理:
        在3D空间中，计算膝关节相对于髋-踝连线的横向偏移

        valgus: 膝关节向内偏移（X轴负向）
        varus: 膝关节向外偏移（X轴正向）

        Returns:
            {
                'left_knee': {
                    'valgus_angle': float,  # 度
                    'max_valgus': float,
                    'issue': 'normal'|'valgus'|'varus',
                    'severity': 'none'|'mild'|'moderate'|'severe'
                },
                'right_knee': {...}
            }
        """
        pass

    # ============ COM垂直振幅 ============
    def calculate_vertical_oscillation_3d(self, poses_3d: np.ndarray) -> Dict:
        """
        计算身体质心(COM)的垂直振幅

        原理:
        1. 估算COM位置（髋关节中点近似）
        2. 提取COM的z坐标时间序列
        3. 计算振幅 = max(z) - min(z)
        4. 归一化为身高百分比

        优势: z坐标直接是高度，无需躯干归一化

        Returns:
            {
                'amplitude_cm': float,       # 绝对振幅（厘米）
                'amplitude_percent': float,  # 身高百分比
                'frequency': float,          # 振荡频率
                'rating': {...}
            }
        """
        pass
```

### 模块3: landing_detector_3d.py（独立模块）

```python
"""
3D落地检测器（独立模块）
这是从2D迁移到3D后最核心的改进
"""

class LandingDetector3D:
    """
    基于3D坐标的落地检测

    核心算法:
    1. 提取左右脚踝的z坐标序列
    2. 对z序列进行轻度平滑（Savitzky-Golay）
    3. 计算z速度（一阶导数）
    4. 检测落地时刻：z达到局部最小值 且 z速度过零
    5. 生物力学约束验证：落地时膝角应在145-175°范围
    """

    # 生物力学硬约束（从原项目保留）
    MIN_LANDING_KNEE_ANGLE = 145.0
    MAX_LANDING_KNEE_ANGLE = 175.0

    def detect(self, poses_3d: np.ndarray, fps: float) -> Dict:
        """
        检测所有落地事件

        Returns:
            {
                'valid_landings': [
                    {
                        'frame': int,
                        'foot': 'left'|'right',
                        'knee_angle': float,
                        'foot_z': float,
                        'confidence': float
                    }
                ],
                'rejected_landings': [
                    {
                        'frame': int,
                        'rejection_reason': str,  # 'angle_too_low'|'angle_too_high'|...
                        'actual_angle': float
                    }
                ],
                'ground_contact_time_ms': float,
                'flight_time_ms': float
            }
        """
        pass

    def _detect_foot_contact(self, foot_z: np.ndarray, fps: float) -> List[int]:
        """
        检测单脚触地帧

        算法:
        1. 平滑z序列
        2. 计算z速度
        3. 找z的局部最小值（谷值）
        4. 验证谷值处速度接近零或从负变正
        """
        pass

    def _validate_with_biomechanics(self, frame: int, poses_3d: np.ndarray,
                                     foot: str) -> Dict:
        """
        生物力学约束验证

        检查:
        1. 膝角是否在合理范围
        2. 膝关节是否处于伸展趋势
        """
        pass
```

### 模块4: quality_evaluator.py（重构）

```python
"""
质量评价模块（保留框架，更新指标）

三维度评分体系:
- 稳定性 (Stability): 35%
- 效率 (Efficiency): 35%
- 跑姿 (Form): 30%
"""

class QualityEvaluator:

    WEIGHTS = {
        'stability': 0.35,
        'efficiency': 0.35,
        'form': 0.30
    }

    def evaluate(self, kinematic_3d: Dict, temporal_analysis: Dict = None) -> Dict:
        """
        综合评估（适配3D指标）

        Args:
            kinematic_3d: 3D运动学分析结果
            temporal_analysis: 深度学习时序分析结果（可选）

        Returns:
            {
                'total_score': float,  # 0-100
                'rating': str,         # 精英/优秀/良好/一般/待改进
                'dimension_scores': {
                    'stability': float,
                    'efficiency': float,
                    'form': float
                },
                'strengths': [...],
                'weaknesses': [...],
                'suggestions': [...]
            }
        """
        pass

    def _evaluate_stability_3d(self, kinematic_3d: Dict) -> float:
        """
        稳定性评估（3D版本）

        指标:
        - 骨盆横向位移（3D新增）
        - 躯干晃动（COM轨迹标准差）
        - 头部稳定性
        - 步态对称性（3D新增）
        """
        pass

    def _evaluate_efficiency_3d(self, kinematic_3d: Dict) -> float:
        """
        效率评估（3D版本）

        指标:
        - COM垂直振幅（3D直接测量）
        - 步频
        - 触地时间
        """
        pass

    def _evaluate_form_3d(self, kinematic_3d: Dict) -> float:
        """
        跑姿评估（3D版本）

        指标:
        - 膝关节角度（3D真实角度）
        - 膝内扣/外翻程度（3D新增）
        - 躯干前倾角（3D更准确）
        - 髋关节伸展（3D新增）
        """
        pass
```

---

## 📊 3D vs 2D 指标对比

| 指标 | 2D实现 | 3D实现 | 改进 |
|------|--------|--------|------|
| **膝关节角度** | 2D投影角度，受视角影响 | 3D真实角度，无失真 | ⭐⭐⭐ |
| **落地检测** | ankle_y峰值，噪声大 | foot_z + z速度，更可靠 | ⭐⭐⭐ |
| **垂直振幅** | 需要躯干归一化 | 直接z坐标差值 | ⭐⭐ |
| **膝内扣/外翻** | 无法准确检测 | 3D横向偏移直接计算 | ⭐⭐⭐ |
| **步态对称性** | 左右y坐标对比 | 3D轨迹完整对比 | ⭐⭐ |
| **骨盆位移** | 无法检测 | 3D x坐标变化 | ⭐⭐⭐ |
| **躯干前倾** | 2D投影估算 | 3D矢量角度 | ⭐⭐ |

---

## 🔌 依赖更新

### requirements.txt

```
# 核心依赖
torch>=2.0.0
torchvision>=0.15.0

# MMPose（2D姿态估计）
mmcv>=2.0.0
mmpose>=1.0.0
mmdet>=3.0.0

# MotionBERT（3D提升）
# 需要从GitHub安装: https://github.com/Walter0807/MotionBERT
einops>=0.6.0
timm>=0.9.0

# 信号处理
scipy>=1.10.0
numpy>=1.24.0

# 可视化
matplotlib>=3.7.0
plotly>=5.14.0  # 3D可视化

# Web界面
streamlit>=1.22.0
flask>=2.3.0
flask-cors>=4.0.0

# AI分析
zhipuai>=2.0.0

# 数据库
# sqlite3 (Python内置)

# 视频处理
opencv-python>=4.7.0

# 其他
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0
```

---

## ⚙️ 配置文件更新

### config.py 新增配置

```python
# ============ MMPose 配置 ============
MMPOSE_CONFIG = {
    # 2D检测器
    '2d_detector': {
        'type': 'RTMPose',  # 推荐：速度快，精度够用
        'config': 'rtmpose-m_8xb256-420e_coco-256x192.py',
        'checkpoint': 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth',
        'device': 'cuda:0'
    },

    # 人体检测器（用于裁剪人体区域）
    'person_detector': {
        'type': 'RTMDet',
        'config': 'rtmdet_m_8xb32-100e_coco-obj365-person.py',
        'checkpoint': 'rtmdet_m_8xb32-100e_coco-obj365-person.pth'
    }
}

# ============ MotionBERT 配置 ============
MOTIONBERT_CONFIG = {
    'model_type': 'MotionBERT',
    'checkpoint': 'data/checkpoints/motionbert_pretrained.pth',
    'config': {
        'num_frames': 243,      # 时序窗口大小
        'num_joints': 17,       # COCO关键点数
        'in_channels': 2,       # 2D输入
        'out_channels': 3,      # 3D输出
        'embed_dim': 512,
        'depth': 5,
        'num_heads': 8
    },
    # 显存优化
    'optimization': {
        'use_fp16': True,       # 半精度推理
        'batch_size': 1,        # 小批量
        'max_frames': 243,      # 分段处理
        'overlap': 27           # 段间重叠
    }
}

# ============ 3D运动学配置 ============
KINEMATIC_3D_CONFIG = {
    # 落地检测
    'landing_detection': {
        'min_knee_angle': 145.0,  # 最小合法落地膝角
        'max_knee_angle': 175.0,  # 最大合法落地膝角
        'z_velocity_threshold': 0.05,  # z速度阈值
        'min_landing_interval_ms': 200  # 最小落地间隔
    },

    # 膝内扣/外翻
    'knee_valgus': {
        'normal_range': (-5, 5),      # 正常范围（度）
        'mild_range': (-10, 10),
        'moderate_range': (-15, 15)
    },

    # 垂直振幅
    'vertical_oscillation': {
        'excellent_max': 0.03,    # 3% 身高
        'good_max': 0.05,
        'fair_max': 0.08
    }
}
```

---

## 🎯 开发优先级

### Phase 1: 核心框架（必须完成）
1. ✅ MMPose 2D 集成
2. ✅ MotionBERT 3D 集成
3. ✅ 3D膝关节角度计算
4. ✅ 3D落地检测

### Phase 2: 完整分析（重要）
5. ✅ 步频计算（基于3D落地检测）
6. ✅ COM垂直振幅
7. ✅ 质量评分系统适配
8. ✅ UI更新

### Phase 3: 高级功能（加分项）
9. ⬜ 膝内扣/外翻检测
10. ⬜ 骨盆位移分析
11. ⬜ 3D可视化
12. ⬜ 步态对称性详细分析

---

## 📝 论文相关

### 论文标题
《基于深度学习的跑步动作视频解析与技术质量评价系统的设计与实现》

### 技术亮点（可写入论文）

1. **深度学习驱动的3D姿态估计**
   - 使用MotionBERT实现2D→3D提升
   - 解决单目视频的深度估计问题

2. **基于3D坐标的生物力学分析**
   - 消除2D投影的透视失真
   - 膝关节角度准确度提升
   - 落地检测可靠性提升

3. **多维度技术质量评价体系**
   - 稳定性/效率/跑姿三维度
   - 可解释的评分逻辑
   - AI辅助报告生成

4. **工程化系统设计**
   - 模块化架构
   - 低显存优化策略
   - Web界面展示

---

## ⚠️ 注意事项

### 显存优化（4GB限制）
```python
# 必须使用的优化策略:
1. torch.cuda.empty_cache()  # 及时清理显存
2. with torch.no_grad():     # 推理时禁用梯度
3. model.half()              # FP16推理
4. 分段处理: 243帧/段，重叠27帧
5. 降采样: 30fps → 15fps
```

### MotionBERT 安装
```bash
# 需要从源码安装
git clone https://github.com/Walter0807/MotionBERT.git
cd MotionBERT
pip install -e .

# 下载预训练权重
# 放到 data/checkpoints/motionbert_pretrained.pth
```

### 关键点格式转换
```
MediaPipe (33点) → COCO (17点) → MotionBERT

需要实现关键点映射，或直接使用MMPose输出的COCO格式
```

---

## 🚀 开始开发

请按以下顺序开发:

1. **环境搭建**: 安装MMPose + MotionBERT
2. **pose_estimator_3d.py**: 封装2D+3D流程
3. **kinematic_analyzer_3d.py**: 实现3D运动学分析
4. **landing_detector_3d.py**: 实现3D落地检测
5. **quality_evaluator.py**: 适配3D指标
6. **streamlit_app.py**: 更新UI
7. **测试验证**: 与2D结果对比

---

## 📎 参考资源

- MMPose文档: https://mmpose.readthedocs.io/
- MotionBERT: https://github.com/Walter0807/MotionBERT
- 跑步生物力学参考: 膝角155-175°、步频180+、垂直振幅<5%

---

**文档版本**: v1.0
**创建时间**: 2024年
**用途**: 新Claude Code会话的迁移指导
