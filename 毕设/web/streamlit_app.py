import streamlit as st
import sys
from pathlib import Path
import cv2
import tempfile
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import STREAMLIT_CONFIG, POSE_CONFIG, VIEW_DETECTION_CONFIG
from modules.video_processor import VideoProcessor
from modules.pose_estimator import create_pose_estimator
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.temporal_model import TemporalModelAnalyzer
from modules.quality_evaluator import QualityEvaluator
from modules.ai_analyzer import AIAnalyzer
from modules.database import DatabaseManager
from modules.view_detector import ViewAngleDetector, AdaptiveAnalyzer

# 页面配置
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout']
)


# 初始化组件
@st.cache_resource
def init_components():
    """初始化系统组件"""
    return {
        'db': DatabaseManager(),
        'ai': AIAnalyzer()
    }


components = init_components()


def main():
    """主界面"""
    st.title("🏃 跑步动作分析系统")
    st.markdown("*基于深度学习的跑步动作视频解析与技术质量评价*")
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.header("📋 导航")
        page = st.radio(
            "选择功能",
            ["视频分析", "历史记录", "系统统计", "系统设置"]
        )

        st.markdown("---")
        st.info("💡 上传跑步视频，获取专业技术分析")

        # 显示系统信息
        st.markdown("---")
        st.caption("系统信息")
        st.caption(f"姿态估计: {POSE_CONFIG['backend'].upper()}")

    # 主内容区
    if page == "视频分析":
        video_analysis_page()
    elif page == "历史记录":
        history_page()
    elif page == "系统统计":
        statistics_page()
    elif page == "系统设置":
        settings_page()


def video_analysis_page():
    """视频分析页面"""
    st.header("📹 视频分析")

    # 分析选项
    with st.expander("⚙️ 分析选项", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            auto_detect_view = st.checkbox("自动检测视角", value=True,
                                           help="自动识别视频是侧面、正面还是混合视角")
        with col2:
            manual_view = st.selectbox(
                "手动指定视角",
                ["自动", "侧面", "正面", "背面"],
                disabled=auto_detect_view
            )

    # 文件上传
    uploaded_file = st.file_uploader(
        "上传跑步视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="支持常见视频格式，建议使用侧面或正面拍摄的视频"
    )

    if uploaded_file is not None:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # 显示视频信息
        st.video(uploaded_file)

        # 分析按钮
        if st.button("🔍 开始分析", type="primary"):
            view_override = None if auto_detect_view else {
                "自动": None, "侧面": "side", "正面": "front", "背面": "back"
            }.get(manual_view)
            analyze_video(video_path, view_override)


def analyze_video(video_path: str, view_override: str = None):
    """执行视频分析"""
    try:
        # 进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1. 视频预处理
        status_text.text("1️⃣ 视频预处理中...")
        progress_bar.progress(5)
        processor = VideoProcessor(video_path)
        video_info = processor.get_video_info()
        frames, fps = processor.extract_frames(target_fps=30, max_frames=300)

        # 显示视频信息
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("分辨率", f"{video_info['width']}x{video_info['height']}")
        col2.metric("帧率", f"{video_info['fps']:.1f} FPS")
        col3.metric("时长", f"{video_info['duration']:.1f} 秒")
        col4.metric("提取帧数", f"{len(frames)}")

        # 2. 姿态估计
        status_text.text("2️⃣ 姿态估计中...")
        progress_bar.progress(20)
        estimator = create_pose_estimator(POSE_CONFIG['backend'], POSE_CONFIG)
        keypoints_sequence = estimator.process_frames(frames)

        detected_count = sum(1 for kp in keypoints_sequence if kp['detected'])
        st.info(f"✓ 姿态检测成功: {detected_count}/{len(keypoints_sequence)} 帧 ({detected_count/len(keypoints_sequence)*100:.1f}%)")

        # 3. 视角检测
        status_text.text("3️⃣ 视角检测中...")
        progress_bar.progress(30)

        if view_override:
            detected_view = view_override
            view_confidence = 1.0
            st.info(f"📐 使用手动指定视角: {get_view_name(detected_view)}")
        else:
            view_detector = ViewAngleDetector()
            view_result = view_detector.detect_view_angle(keypoints_sequence)
            detected_view = view_result['primary_view']  # 修复: 使用正确的键名
            view_confidence = view_result['confidence']

            # 显示视角检测结果
            view_col1, view_col2, view_col3 = st.columns(3)
            view_col1.metric("检测视角", get_view_name(detected_view))
            view_col2.metric("置信度", f"{view_confidence*100:.1f}%")
            view_col3.metric("分析策略", get_strategy_name(detected_view))

        # 生成姿态识别内容
        status_text.text("3️⃣ 生成姿态识别视频与关键帧...")
        progress_bar.progress(40)

        # 尝试生成视频
        try:
            pose_video_path = generate_pose_video(frames, keypoints_sequence, fps, estimator)
            st.subheader("🦴 姿态识别视频")
            st.video(pose_video_path)
        except Exception as video_err:
            st.warning(f"视频生成失败: {video_err}，将显示关键帧图像")

        # 提取并显示关键帧（无论视频是否成功都显示）
        keyframe_data = extract_keyframes_with_poses(frames, keypoints_sequence, fps, estimator, num_keyframes=6)
        if keyframe_data:
            st.subheader("🖼️ 关键帧姿态分析")

            # 每行显示3张关键帧
            for row_start in range(0, len(keyframe_data), 3):
                cols = st.columns(3)
                for i, kf in enumerate(keyframe_data[row_start:row_start+3]):
                    with cols[i]:
                        st.image(kf['path'], caption=f"时间: {kf['time_sec']:.2f}s",
                                 use_container_width=True)
                        if not kf['detected']:
                            st.caption("⚠️ 未检测到姿态")

        # 4. 运动学分析（使用自适应分析器）
        status_text.text("4️⃣ 运动学分析中...")
        progress_bar.progress(55)

        adaptive_analyzer = AdaptiveAnalyzer()
        kinematic_results = adaptive_analyzer.analyze(
            keypoints_sequence, fps,
            view_angle=detected_view if not view_override else view_override
        )

        # 5. 深度学习分析
        status_text.text("5️⃣ 深度学习分析中...")
        progress_bar.progress(70)
        temporal_analyzer = TemporalModelAnalyzer()
        temporal_results = temporal_analyzer.analyze(keypoints_sequence)

        # 6. 质量评价
        status_text.text("6️⃣ 技术质量评价中...")
        progress_bar.progress(85)
        quality_evaluator = QualityEvaluator()
        quality_results = quality_evaluator.evaluate(
            kinematic_results, temporal_results,
            view_angle=detected_view
        )

        # 7. AI文本生成
        status_text.text("7️⃣ AI文本分析中...")
        progress_bar.progress(90)
        results_for_ai = {
            'quality_evaluation': quality_results,
            'kinematic_analysis': kinematic_results,
            'temporal_analysis': temporal_results,
            'view_angle': detected_view
        }
        ai_text = components['ai'].generate_analysis_report(results_for_ai)

        # 8. 多模态时间段分析（如果有关键帧数据）
        time_segment_analysis = ""
        if keyframe_data and len(keyframe_data) > 0:
            status_text.text("8️⃣ 多模态时间段分析中...")
            progress_bar.progress(95)
            try:
                time_segment_analysis = components['ai'].analyze_time_segments(
                    keyframe_data, kinematic_results
                )
            except Exception as e:
                st.warning(f"时间段分析失败: {e}")
                time_segment_analysis = ""

        # 完成
        progress_bar.progress(100)
        status_text.text("✅ 分析完成!")

        # 显示结果
        st.markdown("---")
        display_results(quality_results, kinematic_results, temporal_results, ai_text, detected_view, time_segment_analysis)

        # 保存到数据库
        complete_results = {
            'video_info': video_info,
            'kinematic_analysis': kinematic_results,
            'temporal_analysis': temporal_results,
            'quality_evaluation': quality_results,
            'ai_analysis': ai_text,
            'view_angle': detected_view
        }
        record_id = components['db'].save_analysis(complete_results)
        st.success(f"分析结果已保存 (ID: {record_id})")

        # 清理资源
        processor.release()
        estimator.close()

    except Exception as e:
        st.error(f"分析过程出错: {e}")
        import traceback
        st.code(traceback.format_exc())


def get_view_name(view: str) -> str:
    """获取视角中文名称"""
    names = {
        'side': '侧面视角',
        'front': '正面视角',
        'back': '背面视角',
        'mixed': '混合视角'
    }
    return names.get(view, view)


def get_strategy_name(view: str) -> str:
    """获取分析策略名称"""
    strategies = {
        'side': '膝角+振幅+躯干',
        'front': '对称性+髋部+膝外翻',
        'back': '对称性+足跟',
        'mixed': '综合分析'
    }
    return strategies.get(view, '标准分析')


def generate_pose_video(frames, keypoints_sequence, fps, estimator):
    """将姿态骨架绘制到每一帧并生成视频"""
    import tempfile
    import os

    # 使用临时文件避免路径问题
    output_dir = Path("output/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用唯一的文件名
    import time
    timestamp = int(time.time())
    output_path = output_dir / f"pose_visualization_{timestamp}.mp4"

    h, w = frames[0].shape[:2]
    fps_int = max(1, int(round(fps)))

    # 尝试多种编码格式
    codecs = [
        ('avc1', '.mp4'),  # H.264 - 最兼容
        ('mp4v', '.mp4'),  # MPEG-4
        ('XVID', '.avi'),  # XVID
    ]

    writer = None
    final_path = None

    for codec, ext in codecs:
        test_path = output_dir / f"pose_visualization_{timestamp}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(test_path),
            fourcc,
            fps_int,
            (w, h)
        )
        if writer.isOpened():
            final_path = test_path
            break
        writer.release()

    if not writer or not writer.isOpened():
        raise RuntimeError("❌ VideoWriter 打开失败，尝试了多种编码格式")

    for frame, kp in zip(frames, keypoints_sequence):
        if kp.get("detected", False):
            vis_frame = estimator.visualize_pose(frame, kp)
        else:
            vis_frame = frame.copy()

        writer.write(vis_frame)

    writer.release()

    return str(final_path)


def extract_keyframes_with_poses(frames, keypoints_sequence, fps, estimator, num_keyframes=6):
    """提取关键帧并绘制姿态骨架"""
    import time

    output_dir = Path("output/keyframes")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = len(frames)
    if total_frames == 0:
        return []

    # 计算关键帧索引（均匀分布）
    if total_frames <= num_keyframes:
        indices = list(range(total_frames))
    else:
        indices = [int(i * (total_frames - 1) / (num_keyframes - 1)) for i in range(num_keyframes)]

    keyframe_paths = []
    timestamp = int(time.time())

    for i, idx in enumerate(indices):
        frame = frames[idx]
        kp = keypoints_sequence[idx]

        if kp.get("detected", False):
            vis_frame = estimator.visualize_pose(frame.copy(), kp)
        else:
            vis_frame = frame.copy()
            # 在未检测到姿态的帧上添加提示
            cv2.putText(vis_frame, "No pose detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 添加时间戳
        time_sec = idx / fps
        cv2.putText(vis_frame, f"Time: {time_sec:.2f}s", (10, vis_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 保存关键帧
        keyframe_path = output_dir / f"keyframe_{timestamp}_{i}.jpg"
        cv2.imwrite(str(keyframe_path), vis_frame)
        keyframe_paths.append({
            'path': str(keyframe_path),
            'frame_idx': idx,
            'time_sec': time_sec,
            'detected': kp.get("detected", False)
        })

    return keyframe_paths


def display_results(quality, kinematic, temporal, ai_text, view_angle='side', time_segment_analysis=''):
    """显示分析结果"""
    st.header("📊 分析结果")

    # 总体评分
    st.subheader("🎯 总体评价")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        score = quality['total_score']
        st.metric("技术质量评分", f"{score:.1f}/100",
                  delta=f"{quality['rating']}")

    with col2:
        st.metric("分析视角", get_view_name(view_angle))

    with col3:
        st.markdown(f"**评级:** {quality['rating']}")
        if quality.get('strengths'):
            st.markdown(f"**优势:** {', '.join(quality['strengths'][:3])}")
        if quality.get('weaknesses'):
            st.markdown(f"**薄弱项:** {', '.join(quality['weaknesses'][:3])}")

    # 各维度得分
    st.subheader("📈 各维度表现")
    cols = st.columns(4)
    dimensions = quality.get('dimension_scores', {})

    cols[0].metric("稳定性", f"{dimensions.get('stability', 0):.1f}")
    cols[1].metric("效率", f"{dimensions.get('efficiency', 0):.1f}")
    cols[2].metric("跑姿", f"{dimensions.get('form', 0):.1f}")
    cols[3].metric("节奏", f"{dimensions.get('rhythm', 0):.1f}")

    # 运动学指标 - 根据视角显示不同信息
    st.subheader("🔬 运动学指标")

    # 基础指标（所有视角都显示）
    cadence_data = kinematic.get('cadence', {})
    col1, col2, col3 = st.columns(3)
    col1.metric("步频", f"{cadence_data.get('cadence', 0):.1f} 步/分",
                delta=cadence_data.get('rating', {}).get('description', ''))
    col2.metric("检测步数", f"{cadence_data.get('step_count', 0)} 步",
                help=f"视频时长 {cadence_data.get('duration', 0):.1f} 秒")

    # 垂直振幅 - 使用归一化值
    vertical_motion = kinematic.get('vertical_motion', {})
    if 'amplitude_normalized' in vertical_motion:
        amplitude_pct = vertical_motion['amplitude_normalized']
        rating_info = vertical_motion.get('amplitude_rating', {})
        col3.metric("垂直振幅", f"{amplitude_pct:.1f}% 躯干",
                    delta=rating_info.get('description', ''),
                    help="相对于躯干长度的垂直振幅百分比")
    elif vertical_motion.get('amplitude', 0) > 0:
        col3.metric("垂直振幅", f"{vertical_motion['amplitude']:.4f}",
                    help="归一化坐标下的振幅")
    else:
        col3.metric("垂直振幅", "数据不足")

    # 膝关节角度分析（侧面视角重点）
    if view_angle in ['side', 'mixed']:
        angles = kinematic.get('angles', {})

        # phase_analysis 直接在 angles 下，不是在 angles['knee'] 下
        if 'phase_analysis' in angles:
            st.subheader("🦵 膝关节角度分析（分阶段）")
            phase_analysis = angles['phase_analysis']

            phase_cols = st.columns(3)

            # 触地阶段
            gc = phase_analysis.get('ground_contact', {})
            with phase_cols[0]:
                st.markdown("**触地阶段**")
                st.metric("平均角度", f"{gc.get('mean', 0):.1f}°")
                st.caption(f"范围: {gc.get('min', 0):.1f}° - {gc.get('max', 0):.1f}°")
                st.caption(f"帧数: {gc.get('count', 0)}")

            # 腾空阶段
            fl = phase_analysis.get('flight', {})
            with phase_cols[1]:
                st.markdown("**腾空阶段**")
                st.metric("平均角度", f"{fl.get('mean', 0):.1f}°")
                st.caption(f"范围: {fl.get('min', 0):.1f}° - {fl.get('max', 0):.1f}°")
                st.caption(f"帧数: {fl.get('count', 0)}")

            # 过渡阶段
            tr = phase_analysis.get('transition', {})
            with phase_cols[2]:
                st.markdown("**过渡阶段**")
                st.metric("平均角度", f"{tr.get('mean', 0):.1f}°")
                st.caption(f"范围: {tr.get('min', 0):.1f}° - {tr.get('max', 0):.1f}°")
                st.caption(f"帧数: {tr.get('count', 0)}")

    # 对称性分析（正面/背面视角重点）
    if view_angle in ['front', 'back', 'mixed']:
        symmetry = kinematic.get('symmetry', {})
        if symmetry:
            st.subheader("⚖️ 对称性分析")
            sym_cols = st.columns(3)

            sym_cols[0].metric("肩部对称性",
                               f"{symmetry.get('shoulder_symmetry', 0)*100:.1f}%")
            sym_cols[1].metric("髋部对称性",
                               f"{symmetry.get('hip_symmetry', 0)*100:.1f}%")
            sym_cols[2].metric("整体对称性",
                               f"{symmetry.get('overall_symmetry', 0)*100:.1f}%")

    # 深度学习结果
    st.subheader("🤖 深度学习分析")
    col1, col2 = st.columns(2)

    col1.metric("AI质量评分", f"{temporal.get('quality_score', 0):.1f}")
    col2.metric("AI稳定性", f"{temporal.get('stability_score', 0):.1f}")

    phase_dist = temporal.get('phase_distribution', {})
    if phase_dist:
        st.markdown(f"**阶段分布:** 触地 {phase_dist.get('ground_contact', 0) * 100:.1f}% | "
                    f"腾空 {phase_dist.get('flight', 0) * 100:.1f}% | "
                    f"过渡 {phase_dist.get('transition', 0) * 100:.1f}%")

    # 改进建议
    if quality.get('suggestions'):
        st.subheader("💡 改进建议")
        for i, suggestion in enumerate(quality['suggestions'], 1):
            st.markdown(f"{i}. {suggestion}")

    # AI分析文本
    st.subheader("📝 AI深度分析")
    st.markdown(ai_text)

    # 时间段问题分析（多模态）
    if time_segment_analysis:
        st.subheader("🔍 多模态时间段分析")
        st.markdown(time_segment_analysis)


def history_page():
    """历史记录页面"""
    st.header("📜 历史记录")

    records = components['db'].get_recent_analyses(20)

    if not records:
        st.info("暂无历史记录")
        return

    for record in records:
        with st.expander(f"📹 {record['video_filename']} - {record['analysis_date']}"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("评分", f"{record['total_score']:.1f}")
            col2.metric("评级", record['rating'])
            col3.metric("时长", f"{record['video_duration']:.1f}秒")
            col4.metric("步频", f"{record['cadence']:.1f}")


def statistics_page():
    """统计页面"""
    st.header("📊 系统统计")

    stats = components['db'].get_statistics()

    col1, col2 = st.columns(2)
    col1.metric("总分析次数", stats['total_analyses'])
    col2.metric("平均评分", f"{stats['average_score']:.1f}")

    st.subheader("评级分布")
    for rating, count in stats['rating_distribution'].items():
        st.markdown(f"**{rating}:** {count} 次")


def settings_page():
    """系统设置页面"""
    st.header("⚙️ 系统设置")

    st.subheader("姿态估计设置")
    st.info(f"当前后端: {POSE_CONFIG['backend'].upper()}")
    st.caption("如需切换姿态估计后端，请修改配置文件 config/config.py")

    st.subheader("视角检测设置")
    with st.expander("查看当前配置"):
        st.json(VIEW_DETECTION_CONFIG)

    st.subheader("AI分析设置")
    st.caption("支持的AI提供商: OpenAI, Anthropic, 通义千问, 智谱AI")
    st.caption("如需启用AI分析，请在环境变量中配置相应的API密钥")

    # 显示环境变量状态
    import os
    providers = {
        'OpenAI': 'OPENAI_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY',
        '通义千问': 'DASHSCOPE_API_KEY',
        '智谱AI': 'ZHIPU_API_KEY'
    }

    st.markdown("**API密钥状态:**")
    for name, env_var in providers.items():
        status = "✅ 已配置" if os.getenv(env_var) else "❌ 未配置"
        st.markdown(f"- {name}: {status}")


if __name__ == '__main__':
    main()
