import sys
import argparse
from pathlib import Path

from config.config import OUTPUT_DIR, POSE_CONFIG
from modules.video_processor import VideoProcessor
from modules.pose_estimator import create_pose_estimator
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.temporal_model import TemporalModelAnalyzer
from modules.quality_evaluator import QualityEvaluator
from modules.ai_analyzer import AIAnalyzer
from modules.database import DatabaseManager
from modules.view_detector import ViewAngleDetector, AdaptiveAnalyzer
from utils.visualization import create_comparison_video, plot_angle_curves


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='跑步动作分析系统')
    parser.add_argument('video_path', type=str, help='视频文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--save-db', action='store_true', help='保存到数据库')
    parser.add_argument('--view', type=str, choices=['auto', 'side', 'front', 'back'],
                        default='auto', help='视频视角 (auto=自动检测)')

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
    print(f"姿态估计后端: {POSE_CONFIG['backend'].upper()}")
    print(f"视角模式: {args.view}")
    print("=" * 80)

    try:
        # 执行分析
        results = run_analysis_pipeline(
            str(video_path), output_dir, args.visualize,
            view_mode=args.view
        )

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


def run_analysis_pipeline(video_path: str, output_dir: Path, visualize: bool = False,
                          view_mode: str = 'auto'):
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
    print("\n2️⃣ 人体姿态估计...")
    estimator = create_pose_estimator(POSE_CONFIG['backend'], POSE_CONFIG)
    keypoints_sequence = estimator.process_frames(frames)

    detected_count = sum(1 for kp in keypoints_sequence if kp['detected'])
    print(f"   检测成功: {detected_count}/{len(keypoints_sequence)} 帧 ({detected_count/len(keypoints_sequence)*100:.1f}%)")

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

    # 3. 视角检测
    print("\n3️⃣ 视角检测...")
    if view_mode == 'auto':
        view_detector = ViewAngleDetector()
        view_result = view_detector.detect_view_angle(keypoints_sequence)
        detected_view = view_result['primary_view']  # 修复: 使用正确的键名
        view_confidence = view_result['confidence']
        print(f"   检测视角: {get_view_name(detected_view)}")
        print(f"   置信度: {view_confidence*100:.1f}%")
        print(f"   分析策略: {get_strategy_description(detected_view)}")
    else:
        detected_view = view_mode
        view_confidence = 1.0
        print(f"   使用手动指定视角: {get_view_name(detected_view)}")

    # 4. 运动学特征解析（使用自适应分析器）
    print("\n4️⃣ 运动学特征解析...")
    adaptive_analyzer = AdaptiveAnalyzer()
    kinematic_results = adaptive_analyzer.analyze(
        keypoints_sequence, fps,
        view_angle=detected_view
    )

    # 基础指标输出
    cadence_data = kinematic_results['cadence']
    print(f"   步频: {cadence_data['cadence']:.1f} 步/分")
    print(f"   检测步数: {cadence_data['step_count']} 步 (视频时长 {cadence_data['duration']:.1f} 秒)")
    if cadence_data.get('confidence', 0) > 0:
        print(f"   步频置信度: {cadence_data['confidence']*100:.1f}%")

    # 垂直振幅（归一化）
    vertical_motion = kinematic_results.get('vertical_motion', {})
    # 优先使用归一化振幅（amplitude_normalized 是相对躯干长度的百分比）
    if 'amplitude_normalized' in vertical_motion:
        amplitude_pct = vertical_motion['amplitude_normalized']
        print(f"   垂直振幅: {amplitude_pct:.2f}% (躯干长度)")
        rating = vertical_motion.get('amplitude_rating', {})
        if rating:
            print(f"   振幅评级: {rating.get('description', '')}")
    elif vertical_motion.get('amplitude', 0) > 0:
        print(f"   垂直振幅: {vertical_motion['amplitude']:.4f} (归一化坐标)")
    else:
        print(f"   垂直振幅: 数据不足")

    # 膝关节角度分析（侧面视角）
    if detected_view in ['side', 'mixed']:
        angles = kinematic_results.get('angles', {})
        knee_angles = angles.get('knee', {})
        if 'phase_analysis' in knee_angles:
            print("   膝关节角度（分阶段）:")
            phase_analysis = knee_angles['phase_analysis']
            for phase_name, phase_data in phase_analysis.items():
                phase_cn = {'ground_contact': '触地', 'flight': '腾空', 'transition': '过渡'}.get(phase_name, phase_name)
                if phase_data.get('count', 0) > 0:
                    print(f"      {phase_cn}: {phase_data['mean']:.1f}° (范围: {phase_data['min']:.1f}°-{phase_data['max']:.1f}°)")

    # 可视化角度曲线
    if visualize and 'angles' in kinematic_results:
        print("   生成角度曲线图...")
        try:
            plot_angle_curves(kinematic_results['angles'],
                              str(output_dir / 'angle_curves.png'))
        except Exception as e:
            print(f"   警告: 无法生成角度曲线图 - {e}")

    # 5. 时序深度学习分析
    print("\n5️⃣ 时序深度学习分析（LSTM/CNN）...")
    temporal_analyzer = TemporalModelAnalyzer()
    temporal_results = temporal_analyzer.analyze(keypoints_sequence)

    print(f"   AI质量评分: {temporal_results['quality_score']:.2f}")
    print(f"   AI稳定性: {temporal_results['stability_score']:.2f}")

    phase_dist = temporal_results['phase_distribution']
    print(f"   阶段分布: 触地{phase_dist['ground_contact'] * 100:.1f}% | "
          f"腾空{phase_dist['flight'] * 100:.1f}% | "
          f"过渡{phase_dist['transition'] * 100:.1f}%")

    # 6. 跑步技术质量评价
    print("\n6️⃣ 跑步技术质量评价...")
    quality_evaluator = QualityEvaluator()
    quality_results = quality_evaluator.evaluate(
        kinematic_results, temporal_results,
        view_angle=detected_view
    )

    print(f"   总体评分: {quality_results['total_score']:.2f}/100")
    print(f"   评级: {quality_results['rating']}")

    # 7. AI文本分析
    print("\n7️⃣ AI文本分析与报告生成...")
    ai_analyzer = AIAnalyzer()
    results_for_ai = {
        'quality_evaluation': quality_results,
        'kinematic_analysis': kinematic_results,
        'temporal_analysis': temporal_results,
        'view_angle': detected_view
    }
    ai_text = ai_analyzer.generate_analysis_report(results_for_ai)

    # 保存AI报告
    with open(output_dir / 'ai_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(ai_text)
    print(f"   AI报告已保存: {output_dir / 'ai_analysis_report.txt'}")

    # 整合结果
    complete_results = {
        'video_info': video_info,
        'view_angle': detected_view,
        'view_confidence': view_confidence,
        'kinematic_analysis': kinematic_results,
        'temporal_analysis': temporal_results,
        'quality_evaluation': quality_results,
        'ai_analysis': ai_text
    }

    # 清理资源
    processor.release()
    estimator.close()

    return complete_results


def get_view_name(view: str) -> str:
    """获取视角中文名称"""
    names = {
        'side': '侧面视角',
        'front': '正面视角',
        'back': '背面视角',
        'mixed': '混合视角'
    }
    return names.get(view, view)


def get_strategy_description(view: str) -> str:
    """获取分析策略描述"""
    strategies = {
        'side': '膝关节角度 + 垂直振幅 + 躯干前倾',
        'front': '身体对称性 + 髋部稳定性 + 膝外翻检测',
        'back': '身体对称性 + 髋部稳定性 + 足跟外翻检测',
        'mixed': '综合分析（侧面+正面指标）'
    }
    return strategies.get(view, '标准分析')


def print_results(results: dict):
    """打印分析结果"""
    quality = results['quality_evaluation']
    view_angle = results.get('view_angle', 'unknown')

    print("\n" + "=" * 80)
    print("📊 分析结果汇总")
    print("=" * 80)

    print(f"\n📐 视角信息")
    print(f"   检测视角: {get_view_name(view_angle)}")
    print(f"   置信度: {results.get('view_confidence', 0)*100:.1f}%")

    print(f"\n🎯 总体评价")
    print(f"   技术质量评分: {quality['total_score']:.2f}/100")
    print(f"   评级: {quality['rating']}")

    print(f"\n📈 各维度得分")
    dims = quality.get('dimension_scores', {})
    print(f"   稳定性: {dims.get('stability', 0):.2f}")
    print(f"   效率: {dims.get('efficiency', 0):.2f}")
    print(f"   跑姿: {dims.get('form', 0):.2f}")
    print(f"   节奏: {dims.get('rhythm', 0):.2f}")

    if quality.get('strengths'):
        print(f"\n✅ 优势")
        for strength in quality['strengths']:
            print(f"   • {strength}")

    if quality.get('weaknesses'):
        print(f"\n⚠️  薄弱项")
        for weakness in quality['weaknesses']:
            print(f"   • {weakness}")

    if quality.get('suggestions'):
        print(f"\n💡 改进建议")
        for suggestion in quality['suggestions']:
            print(f"   • {suggestion}")


if __name__ == '__main__':
    main()
