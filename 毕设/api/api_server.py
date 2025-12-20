from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import API_CONFIG, OUTPUT_DIR
from modules.video_processor import VideoProcessor
from modules.pose_estimator import PoseEstimator
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.temporal_model import TemporalModelAnalyzer
from modules.quality_evaluator import QualityEvaluator
from modules.ai_analyzer import AIAnalyzer
from modules.database import DatabaseManager

app = Flask(__name__)
CORS(app)  # 允许跨域

# 初始化模块
db_manager = DatabaseManager()
ai_analyzer = AIAnalyzer()


@app.route('/')
def index():
    """API根路径"""
    return jsonify({
        'message': '跑步动作分析系统 API',
        'version': '1.0.0',
        'endpoints': {
            'analyze': '/api/analyze_video',
            'result': '/api/result/<id>',
            'history': '/api/history',
            'statistics': '/api/statistics'
        }
    })


@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    """
    分析视频接口
    Request: multipart/form-data with 'video' file
    Response: JSON with analysis results
    """
    try:
        # 检查文件
        if 'video' not in request.files:
            return jsonify({'error': '未上传视频文件'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        # 保存临时文件
        temp_path = OUTPUT_DIR / 'temp_video.mp4'
        video_file.save(temp_path)

        # 执行分析
        results = run_complete_analysis(str(temp_path))

        # 保存到数据库
        record_id = db_manager.save_analysis(results)
        results['record_id'] = record_id

        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()

        return jsonify({
            'success': True,
            'record_id': record_id,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/result/<int:record_id>', methods=['GET'])
def get_result(record_id):
    """获取分析结果"""
    try:
        record = db_manager.get_analysis_by_id(record_id)

        if not record:
            return jsonify({'error': '记录不存在'}), 404

        return jsonify({
            'success': True,
            'record': record
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """获取历史记录"""
    try:
        limit = request.args.get('limit', 10, type=int)
        records = db_manager.get_recent_analyses(limit)

        return jsonify({
            'success': True,
            'count': len(records),
            'records': records
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """获取统计信息"""
    try:
        stats = db_manager.get_statistics()

        return jsonify({
            'success': True,
            'statistics': stats
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def run_complete_analysis(video_path: str) -> dict:
    """运行完整分析流程"""
    print(f"开始分析视频: {video_path}")

    # 1. 视频预处理
    print("1️⃣ 视频预处理...")
    processor = VideoProcessor(video_path)
    video_info = processor.get_video_info()
    frames, fps = processor.extract_frames(target_fps=30, max_frames=300)

    # 2. 姿态估计
    print("2️⃣ 姿态估计...")
    estimator = PoseEstimator()
    keypoints_sequence = estimator.process_frames(frames)

    # 3. 运动学分析
    print("3️⃣ 运动学分析...")
    kinematic_analyzer = KinematicAnalyzer()
    kinematic_results = kinematic_analyzer.analyze_sequence(keypoints_sequence, fps)

    # 4. 深度学习分析
    print("4️⃣ 深度学习分析...")
    temporal_analyzer = TemporalModelAnalyzer()
    temporal_results = temporal_analyzer.analyze(keypoints_sequence)

    # 5. 质量评价
    print("5️⃣ 技术质量评价...")
    quality_evaluator = QualityEvaluator()
    quality_results = quality_evaluator.evaluate(kinematic_results, temporal_results)

    # 6. AI文本生成
    print("6️⃣ AI文本分析...")
    results_for_ai = {
        'quality_evaluation': quality_results,
        'kinematic_analysis': kinematic_results,
        'temporal_analysis': temporal_results
    }
    ai_text = ai_analyzer.generate_analysis_report(results_for_ai)

    # 整合结果
    complete_results = {
        'video_info': video_info,
        'kinematic_analysis': kinematic_results,
        'temporal_analysis': temporal_results,
        'quality_evaluation': quality_results,
        'ai_analysis': ai_text
    }

    # 清理资源
    processor.release()
    estimator.close()

    print("✅ 分析完成!")
    return complete_results


if __name__ == '__main__':
    print("=" * 60)
    print("跑步动作分析系统 - API服务器")
    print("=" * 60)
    print(f"服务地址: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print("=" * 60)

    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )