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

# MediaPipe Pose配置
POSE_CONFIG = {
    'model_complexity': 1,  # 0, 1, 2 (复杂度递增)
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'static_image_mode': False
}

# 运动学分析配置
KINEMATIC_CONFIG = {
    'smooth_window': 5,  # 平滑窗口大小
    'min_step_duration': 0.2,  # 最小步态周期(秒)
    'max_step_duration': 1.5   # 最大步态周期(秒)
}

# 深度学习模型配置
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

# 技术质量评价权重配置
QUALITY_WEIGHTS = {
    'stability': 0.3,      # 动作稳定性
    'efficiency': 0.3,     # 动作效率
    'form': 0.2,          # 跑姿标准度
    'rhythm': 0.2         # 节奏一致性
}

# 技术质量阈值配置
QUALITY_THRESHOLDS = {
    'excellent': 85,
    'good': 70,
    'fair': 55,
    'poor': 0
}

# AI分析配置（预留接口）
AI_CONFIG = {
    'enabled': False,  # 是否启用AI分析
    'provider': 'openai',  # 'openai', 'anthropic', 'qwen'
    'api_key': os.getenv('AI_API_KEY', ''),
    'model': 'gpt-3.5-turbo',
    'max_tokens': 500,
    'temperature': 0.7
}

# Flask API配置
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}

# Streamlit配置
STREAMLIT_CONFIG = {
    'page_title': '跑步动作分析系统',
    'page_icon': '🏃',
    'layout': 'wide'
}