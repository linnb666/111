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

    # 视角选择（必须手动选择）
    st.info("📐 请根据您的视频拍摄角度选择正确的视角")
    view_angle = st.radio(
        "选择视频拍摄视角",
        ["侧面视角", "正面视角"],
        horizontal=True,
        help="侧面视角：从跑者侧面拍摄，适合分析膝关节角度、垂直振幅、躯干前倾。\n正面视角：从跑者正前方拍摄，适合分析左右对称性、下肢力线。"
    )

    selected_view = "side" if view_angle == "侧面视角" else "front"

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
            analyze_video(video_path, selected_view)


def analyze_video(video_path: str, selected_view: str = 'side'):
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

        # 3. 使用用户选择的视角
        status_text.text("3️⃣ 确认分析视角...")
        progress_bar.progress(30)
        detected_view = selected_view
        st.info(f"📐 使用视角: {get_view_name(detected_view)} - {get_strategy_name(detected_view)}")

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
                                 use_container_width=True)  # 使用容器宽度
                        if not kf['detected']:
                            st.caption("⚠️ 未检测到姿态")

        # 4. 运动学分析（使用自适应分析器）
        status_text.text("4️⃣ 运动学分析中...")
        progress_bar.progress(55)

        adaptive_analyzer = AdaptiveAnalyzer()
        kinematic_results = adaptive_analyzer.analyze(
            keypoints_sequence, fps,
            view_angle=detected_view
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

        # 7. 生成本地规则引擎报告（始终生成）
        status_text.text("7️⃣ 生成本地分析报告...")
        progress_bar.progress(95)
        results_for_report = {
            'quality_evaluation': quality_results,
            'kinematic_analysis': kinematic_results,
            'temporal_analysis': temporal_results,
            'view_angle': detected_view
        }
        # 使用本地规则引擎生成报告
        local_report = components['ai'].local_engine.generate_analysis_report(results_for_report)

        # 完成
        progress_bar.progress(100)
        status_text.text("✅ 分析完成!")

        # 显示结果
        st.markdown("---")
        display_results(quality_results, kinematic_results, temporal_results, local_report, detected_view, results_for_report)

        # 保存到数据库
        complete_results = {
            'video_info': video_info,
            'kinematic_analysis': kinematic_results,
            'temporal_analysis': temporal_results,
            'quality_evaluation': quality_results,
            'ai_analysis': local_report,
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
        'front': '正面视角'
    }
    return names.get(view, view)


def get_strategy_name(view: str) -> str:
    """获取分析策略名称"""
    strategies = {
        'side': '膝角+振幅+躯干前倾',
        'front': '对称性+下肢力线+肩部晃动'
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


def display_results(quality, kinematic, temporal, local_report, view_angle='side', results_for_ai=None):
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
    cols = st.columns(3)
    dimensions = quality.get('dimension_scores', {})

    cols[0].metric("稳定性", f"{dimensions.get('stability', 0):.1f}")
    cols[1].metric("效率", f"{dimensions.get('efficiency', 0):.1f}")
    cols[2].metric("跑姿", f"{dimensions.get('form', 0):.1f}")

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

    # 触地时间显示
    gait_cycle = kinematic.get('gait_cycle', {})
    phase_duration = gait_cycle.get('phase_duration_ms', {})
    if phase_duration:
        st.subheader("⏱️ 步态时间")
        time_cols = st.columns(3)
        ground_contact_ms = phase_duration.get('ground_contact', 0)
        flight_ms = phase_duration.get('flight', 0)

        # 触地时间评级
        if ground_contact_ms > 0:
            if ground_contact_ms < 210:
                gc_rating = "精英"
            elif ground_contact_ms < 240:
                gc_rating = "优秀"
            elif ground_contact_ms < 270:
                gc_rating = "良好"
            elif ground_contact_ms < 300:
                gc_rating = "一般"
            else:
                gc_rating = "较差"
            time_cols[0].metric("触地时间", f"{ground_contact_ms:.1f} ms", delta=gc_rating)
        else:
            time_cols[0].metric("触地时间", "数据不足")

        time_cols[1].metric("腾空时间", f"{flight_ms:.1f} ms" if flight_ms > 0 else "数据不足")

        cycle_ms = gait_cycle.get('avg_cycle_duration_ms', 0)
        time_cols[2].metric("步态周期", f"{cycle_ms:.1f} ms" if cycle_ms > 0 else "数据不足")

    # 膝关节角度分析（侧面视角重点）
    if view_angle == 'side':
        angles = kinematic.get('angles', {})

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

    # 正面视角分析（移除对称性，保留下肢力线和肩部稳定）
    if view_angle == 'front':
        # 下肢力线分析
        lower_limb = kinematic.get('lower_limb_alignment', {})
        if lower_limb:
            st.subheader("🦿 下肢力线分析")
            limb_cols = st.columns(2)

            left_leg = lower_limb.get('left_leg', {})
            right_leg = lower_limb.get('right_leg', {})

            with limb_cols[0]:
                st.markdown("**左腿**")
                st.metric("偏移角度", f"{left_leg.get('mean', 0):.1f}°")
                st.caption(f"问题: {left_leg.get('issue', 'unknown')}")

            with limb_cols[1]:
                st.markdown("**右腿**")
                st.metric("偏移角度", f"{right_leg.get('mean', 0):.1f}°")
                st.caption(f"问题: {right_leg.get('issue', 'unknown')}")

        # 肩部稳定性分析（正面视角重点）
        stability = kinematic.get('stability', {})
        if stability and 'shoulder_sway' in stability:
            st.subheader("💪 肩部稳定性")
            st.metric("肩部稳定评分", f"{stability.get('shoulder_sway', 0):.1f}/100")

        # 横向稳定性
        lateral = kinematic.get('lateral_stability', {})
        if lateral:
            st.subheader("↔️ 横向稳定性")
            lat_cols = st.columns(3)
            lat_cols[0].metric("髋部横摆", f"{lateral.get('hip_sway', 0):.2f}%")
            lat_cols[1].metric("肩部横摆", f"{lateral.get('shoulder_sway', 0):.2f}%")
            lat_cols[2].metric("稳定评分", f"{lateral.get('stability_score', 0):.1f}")

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

    # 本地分析报告（始终显示）
    st.subheader("📝 本地分析报告")
    st.markdown(local_report)

    # AI大模型分析按钮（用户自行决定是否使用）
    st.markdown("---")
    st.subheader("🤖 AI智能分析（可选）")
    st.info("点击下方按钮使用智谱AI大模型对数据进行深度分析和总结建议。")

    if st.button("🚀 启动AI智能分析", type="secondary"):
        if results_for_ai:
            with st.spinner("正在调用智谱AI进行深度分析..."):
                try:
                    ai_response = components['ai'].generate_analysis_report(results_for_ai)
                    st.subheader("🧠 AI深度分析结果")
                    st.markdown(ai_response)
                except Exception as e:
                    st.error(f"AI分析失败: {e}")


def history_page():
    """历史记录页面"""
    st.header("📜 历史记录")

    # 管理选项
    with st.expander("🛠️ 管理选项", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空所有记录", type="secondary"):
                count = components['db'].delete_all_analyses()
                st.success(f"已删除 {count} 条记录")
                st.rerun()

        with col2:
            if st.button("🧹 清理临时文件", type="secondary"):
                cleanup_temp_files()
                st.success("临时文件清理完成")

    # 获取记录
    records = components['db'].get_recent_analyses(50)

    if not records:
        st.info("暂无历史记录")
        return

    st.markdown(f"共 **{len(records)}** 条记录")

    for record in records:
        record_id = record.get('id', 0)
        with st.expander(f"📹 {record['video_filename']} - {record['analysis_date']} (ID: {record_id})"):
            # 基本信息
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("评分", f"{record['total_score']:.1f}")
            col2.metric("评级", record['rating'])
            col3.metric("时长", f"{record['video_duration']:.1f}秒")
            col4.metric("步频", f"{record['cadence']:.1f}")

            # 各维度得分
            st.markdown("**各维度得分:**")
            dim_cols = st.columns(3)
            dim_cols[0].metric("稳定性", f"{record.get('stability_score', 0):.1f}")
            dim_cols[1].metric("效率", f"{record.get('efficiency_score', 0):.1f}")
            dim_cols[2].metric("跑姿", f"{record.get('form_score', 0):.1f}")

            # 深度学习结果
            st.markdown("**深度学习分析:**")
            dl_cols = st.columns(2)
            dl_cols[0].metric("AI质量评分", f"{record.get('dl_quality_score', 0):.1f}")
            dl_cols[1].metric("AI稳定性", f"{record.get('dl_stability_score', 0):.1f}")

            # AI分析文本
            ai_text = record.get('ai_analysis_text', '')
            if ai_text:
                with st.container():
                    st.markdown("**AI分析报告:**")
                    st.markdown(ai_text)

            # 操作按钮
            st.markdown("---")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button(f"📊 查看完整数据", key=f"view_{record_id}"):
                    full_results = components['db'].get_full_results(record_id)
                    if full_results:
                        st.json(full_results)
                    else:
                        st.warning("完整数据不可用")

            with btn_col2:
                if st.button(f"🗑️ 删除记录", key=f"delete_{record_id}"):
                    if components['db'].delete_analysis(record_id):
                        st.success("记录已删除")
                        st.rerun()
                    else:
                        st.error("删除失败")


def cleanup_temp_files():
    """清理临时文件"""
    import shutil
    from pathlib import Path

    cleanup_dirs = [
        Path("output/videos"),
        Path("output/keyframes"),
        Path("output/visualizations")
    ]

    total_cleaned = 0
    for dir_path in cleanup_dirs:
        if dir_path.exists():
            for file in dir_path.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                        total_cleaned += 1
                except Exception:
                    pass

    return total_cleaned


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

    st.subheader("视角设置")
    st.markdown("""
    **支持的视角:**
    - **侧面视角**: 分析膝关节角度、垂直振幅、躯干前倾、手臂摆动
    - **正面视角**: 分析左右对称性、下肢力线、肩部晃动
    """)

    st.subheader("AI分析设置")
    st.markdown("**使用智谱AI (glm-4.6模型)**")
    st.caption("需要安装zai库：pip install zai")

    # 显示智谱AI状态
    import os
    from config.config import AI_CONFIG
    api_key = AI_CONFIG.get('api_key', '')

    if api_key:
        st.success("智谱AI已配置")
        st.caption(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    else:
        st.warning("智谱AI未配置，将使用本地规则引擎")
        st.caption("请在 config/config.py 中配置 ZHIPU_API_KEY")

    st.subheader("评价维度")
    st.markdown("""
    **技术质量评分维度（已移除节奏一致性）:**
    | 维度 | 权重 | 说明 |
    |------|------|------|
    | 动作稳定性 | 35% | 躯干稳定、头部稳定 |
    | 跑步效率 | 35% | 垂直振幅、步频 |
    | 跑姿标准度 | 30% | 膝关节角度、躯干前倾 |
    """)


if __name__ == '__main__':
    main()
