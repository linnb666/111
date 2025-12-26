import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
CHECKPOINT_DIR = DATA_DIR / 'checkpoints'

# 创建必要目录
for dir_path in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR,
                 OUTPUT_DIR / 'videos', OUTPUT_DIR / 'visualizations']:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据库配置
DATABASE_PATH = DATA_DIR / 'database.db'

# 视频处理配置
VIDEO_CONFIG = {
    'target_width': 640,
    'target_height': 480,
    'fps': 30,
    'supported_formats': ['.mp4', '.avi', '.mov', '.mkv']
}

# ================== 姿态估计配置 ==================

# MediaPipe Pose配置
POSE_CONFIG = {
    'backend': 'mediapipe',  # 'mediapipe' 或 'mmpose'
    'model_complexity': 1,  # 0, 1, 2 (复杂度递增)
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'static_image_mode': False
}

# MMPose配置（预留）
MMPOSE_CONFIG = {
    'det_model': 'rtmdet',
    'det_checkpoint': '',  # 检测模型权重路径
    'pose_model': 'rtmpose',
    'pose_checkpoint': '',  # 姿态模型权重路径
    'device': 'cuda:0'  # 'cuda:0' 或 'cpu'
}

# ================== 视角检测配置 ==================

VIEW_DETECTION_CONFIG = {
    # 视角判断阈值
    'side_view_threshold': 0.4,      # 肩宽/髋宽比值阈值，低于此值判定为侧面
    'frontal_view_threshold': 0.7,   # 高于此值判定为正面
    'ear_visibility_threshold': 0.3,  # 耳朵可见性阈值
    'nose_offset_threshold': 0.15,    # 鼻子偏移阈值

    # 混合视角判断
    'mixed_view_ratio': 0.3,  # 如果侧面帧占比超过此值但不到0.7，判定为混合视角

    # 置信度阈值
    'min_confidence': 0.5,  # 关键点最低置信度

    # 分析策略
    'analysis_strategies': {
        'side': ['knee_angle', 'vertical_oscillation', 'trunk_lean', 'arm_swing'],
        'front': ['shoulder_symmetry', 'hip_alignment', 'knee_valgus', 'foot_strike'],
        'back': ['shoulder_symmetry', 'hip_alignment', 'heel_whip'],
        'mixed': ['knee_angle', 'vertical_oscillation', 'shoulder_symmetry']
    }
}

# ================== 运动学分析配置 ==================

KINEMATIC_CONFIG = {
    'smooth_window': 5,        # 平滑窗口大小
    'min_step_duration': 0.2,  # 最小步态周期(秒)
    'max_step_duration': 1.5,  # 最大步态周期(秒)

    # 躯干归一化配置
    'trunk_normalization': {
        'enabled': True,
        'fallback_ratio': 0.3,  # 当无法计算躯干长度时，使用图像高度的比例作为参考
        'min_trunk_length': 0.1,  # 最小躯干长度（归一化坐标）
        'smoothing_window': 3     # 躯干长度平滑窗口
    },

    # 相位检测配置
    'phase_detection': {
        'enabled': True,
        'ground_contact_threshold': 0.02,  # 触地判断的Y坐标变化阈值
        'flight_threshold': 0.05,          # 腾空判断的Y坐标阈值
        'min_phase_frames': 3              # 最小相位持续帧数
    },

    # 垂直振幅配置（基于躯干长度归一化）
    'vertical_amplitude': {
        'excellent_max': 0.06,   # 优秀：≤6%躯干长度
        'good_max': 0.10,        # 良好：≤10%躯干长度
        'fair_max': 0.15,        # 一般：≤15%躯干长度
        'poor_min': 0.15         # 较差：>15%躯干长度
    },

    # 膝关节角度配置（分阶段）
    'knee_angle': {
        'ground_contact': {
            'optimal_range': (155, 175),  # 触地阶段最优范围
            'acceptable_range': (145, 180)
        },
        'flight': {
            'optimal_range': (90, 120),   # 腾空阶段最优范围
            'acceptable_range': (80, 140)
        },
        'transition': {
            'optimal_range': (120, 155),  # 过渡阶段最优范围
            'acceptable_range': (100, 165)
        }
    }
}

# ================== 深度学习模型配置 ==================

MODEL_CONFIG = {
    'input_dim': 33 * 2,  # MediaPipe 33个关键点 * 2D坐标
    'hidden_dim': 64,
    'num_layers': 2,
    'output_dim': 3,  # 触地/腾空/过渡
    'dropout': 0.3,
    'sequence_length': 30,  # 时间序列长度
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50
}

# ================== 技术质量评价配置 ==================

# 评价维度权重（删除节奏一致性）
QUALITY_WEIGHTS = {
    'stability': 0.35,     # 动作稳定性
    'efficiency': 0.35,    # 动作效率
    'form': 0.30,          # 跑姿标准度
}

# 评价等级阈值
QUALITY_THRESHOLDS = {
    'excellent': 85,
    'good': 70,
    'fair': 55,
    'poor': 0
}

# 步频标准（不可修改！用户要求保持180-200范围）
CADENCE_THRESHOLDS = {
    'optimal_min': 180,    # 最优步频下限
    'optimal_max': 200,    # 最优步频上限
    'acceptable_min': 160, # 可接受步频下限
    'acceptable_max': 220  # 可接受步频上限
}

# ================== AI分析配置 ==================

# 智谱AI API密钥（用户提供）
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY', '79a902c70ed7420094d2e49d24d48128.OFDigksLwuslKmlp')

AI_CONFIG = {
    'enabled': True,  # 启用AI分析
    'provider': 'zhipu',  # 默认使用智谱AI
    'api_key': ZHIPU_API_KEY,

    # 提供商配置
    'providers': {
        'openai': {
            'enabled': False,
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'api_base': os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
            'model': 'gpt-4-turbo-preview',
            'vision_model': 'gpt-4-vision-preview',
            'max_tokens': 1000,
            'temperature': 0.7
        },
        'anthropic': {
            'enabled': False,
            'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
            'model': 'claude-3-sonnet-20240229',
            'vision_model': 'claude-3-sonnet-20240229',
            'max_tokens': 1000,
            'temperature': 0.7
        },
        'qwen': {
            'enabled': False,
            'api_key': os.getenv('DASHSCOPE_API_KEY', ''),
            'model': 'qwen-turbo',
            'vision_model': 'qwen-vl-plus',
            'max_tokens': 1000,
            'temperature': 0.7
        },
        'zhipu': {
            'enabled': True,
            'api_key': ZHIPU_API_KEY,
            'api_base': 'https://open.bigmodel.cn/api/paas/v4',
            'model': 'glm-4',
            'vision_model': 'glm-4v',
            'max_tokens': 4000,  # 增加token限制，避免文本截断
            'temperature': 0.7
        },
        'local': {
            'enabled': True,  # 本地规则引擎始终可用作后备
            'model': 'rule_engine'
        }
    }
}

# ================== Flask API配置 ==================

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}

# ================== Streamlit配置 ==================

STREAMLIT_CONFIG = {
    'page_title': '跑步动作分析系统',
    'page_icon': '🏃',
    'layout': 'wide'
}

# ================== 日志配置 ==================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(OUTPUT_DIR / 'analysis.log')
}
