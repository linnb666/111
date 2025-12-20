import sys
import argparse
from pathlib import Path

from config.config import OUTPUT_DIR
from modules.video_processor import VideoProcessor
from modules.pose_estimator import PoseEstimator
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.temporal_model import TemporalModelAnalyzer
from modules.quality_evaluator import QualityEvaluator
from modules.ai_analyzer import AIAnalyzer
from modules.database import DatabaseManager
from utils.visualization import create_comparison_video, plot_angle_curves


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='跑步动作分析系统')
    parser.add_argument('video_path', type=str, help='视频文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--save-db', action='store_true', help='保存到数据库')

    args = parser.parse_args()

    # 验证视频文件
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"错误: 视频文件不存在 - {video_path}")
        sys.exit(1)

    # 设置输出目录
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("基于深度学习的跑步动作视频解析与技术质量评价系统")
    print("=" * 80)
    print(f"视频文件: {video_path.name}")
    print("=" * 80)

    try:
        # 执行分析
        results = run_analysis_pipeline(str(video_path), output_dir, args.visualize)

        # 打印结果
        print_results(results)

        # 保存到数据库
        if args.save_db:
            db = DatabaseManager()
            record_id = db.save_analysis(results)
            print(f"\n💾 分析结果已保存到数据库 (ID: {record_id})")

        print("\n" + "=" * 80)
        print("✅ 分析完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_analysis_pipeline(video_path: str, output_dir: Path, visualize: bool = False):
    """运行完整分析流程"""

    # 1. 视频预处理
    print("\n1️⃣ 视频输入与预处理...")
    processor = VideoProcessor(video_path)
    video_info = processor.get_video_info()
    print(f"   分辨率: {video_info['width']}x{video_info['height']}")
    print(f"   帧率: {video_info['fps']:.2f} FPS")
    print(f"   时长: {video_info['duration']:.2f} 秒")

    frames, fps = processor.extract_frames(target_fps=30, max_frames=300)
    print(f"   提取帧数: {len(frames)}")

    # 2. 姿态估计
    print("\n2️⃣ 人体姿态估计（MediaPipe Pose）...")
    estimator = PoseEstimator()
    keypoints_sequence = estimator.process_frames(frames)

    detected_count = sum(1 for kp in keypoints_sequence if kp['detected'])
    print(f"   检测成功: {detected_count}/{len(keypoints_sequence)} 帧")

    # 可视化姿态
    if visualize and detected_count > 0:
        print("   生成姿态可视化...")
        pose_frames = []
        for i, kp in enumerate(keypoints_sequence[:10]):  # 仅前10帧
            pose_frame = estimator.visualize_pose(frames[i], kp)
            pose_frames.append(pose_frame)

        # 保存第一帧
        import cv2
        cv2.imwrite(str(output_dir / 'pose_sample.jpg'), pose_frames[0])

    # 3. 运动学特征解析
    print("\n3️⃣ 运动学特征解析...")
    kinematic_analyzer = KinematicAnalyzer()
    kinematic_results = kinematic_analyzer.analyze_sequence(keypoints_sequence, fps)

    print(f"   步频: {kinematic_results['cadence']['cadence']:.1f} 步/分")
    print(f"   步数: {kinematic_results['cadence']['step_count']}")
    print(f"   垂直振幅: {kinematic_results['vertical_motion']['amplitude']:.2f}")

    # 可视化角度曲线
    if visualize and 'angles' in kinematic_results:
        print("   生成角度曲线图...")
        plot_angle_curves(kinematic_results['angles'],
                          str(output_dir / 'angle_curves.png'))

    # 4. 时序深度学习分析
    print("\n4️⃣ 时序深度学习分析（LSTM/CNN）...")
    temporal_analyzer = TemporalModelAnalyzer()
    temporal_results = temporal_analyzer.analyze(keypoints_sequence)

    print(f"   AI质量评分: {temporal_results['quality_score']:.2f}")
    print(f"   AI稳定性: {temporal_results['stability_score']:.2f}")

    phase_dist = temporal_results['phase_distribution']
    print(f"   阶段分布: 触地{phase_dist['ground_contact'] * 100:.1f}% | "
          f"腾空{phase_dist['flight'] * 100:.1f}% | "
          f"过渡{phase_dist['transition'] * 100:.1f}%")

    # 5. 跑步技术质量评价
    print("\n5️⃣ 跑步技术质量评价...")
    quality_evaluator = QualityEvaluator()
    quality_results = quality_evaluator.evaluate(kinematic_results, temporal_results)

    print(f"   总体评分: {quality_results['total_score']:.2f}/100")
    print(f"   评级: {quality_results['rating']}")

    # 6. AI文本分析
    print("\n6️⃣ AI文本分析与润色...")
    ai_analyzer = AIAnalyzer()
    results_for_ai = {
        'quality_evaluation': quality_results,
        'kinematic_analysis': kinematic_results,
        'temporal_analysis': temporal_results
    }
    ai_text = ai_analyzer.generate_analysis_report(results_for_ai)

    # 保存AI报告
    with open(output_dir / 'ai_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(ai_text)
    print(f"   AI报告已保存: {output_dir / 'ai_analysis_report.txt'}")

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

    return complete_results


def print_results(results: dict):
    """打印分析结果"""
    quality = results['quality_evaluation']

    print("\n" + "=" * 80)
    print("📊 分析结果汇总")
    print("=" * 80)

    print(f"\n🎯 总体评价")
    print(f"   技术质量评分: {quality['total_score']:.2f}/100")
    print(f"   评级: {quality['rating']}")

    print(f"\n📈 各维度得分")
    dims = quality['dimension_scores']
    print(f"   稳定性: {dims['stability']:.2f}")
    print(f"   效率: {dims['efficiency']:.2f}")
    print(f"   跑姿: {dims['form']:.2f}")
    print(f"   节奏: {dims['rhythm']:.2f}")

    print(f"\n✅ 优势")
    for strength in quality['strengths']:
        print(f"   • {strength}")

    print(f"\n⚠️  薄弱项")
    for weakness in quality['weaknesses']:
        print(f"   • {weakness}")

    print(f"\n💡 改进建议")
    for suggestion in quality['suggestions']:
        print(f"   • {suggestion}")


if __name__ == '__main__':
    main()