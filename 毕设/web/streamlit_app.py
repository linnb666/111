import streamlit as st
import sys
from pathlib import Path
import cv2
import tempfile
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import STREAMLIT_CONFIG
from modules.video_processor import VideoProcessor
from modules.pose_estimator import PoseEstimator
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.temporal_model import TemporalModelAnalyzer
from modules.quality_evaluator import QualityEvaluator
from modules.ai_analyzer import AIAnalyzer
from modules.database import DatabaseManager

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
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.header("📋 导航")
        page = st.radio(
            "选择功能",
            ["视频分析", "历史记录", "系统统计"]
        )

        st.markdown("---")
        st.info("💡 上传跑步视频，获取专业技术分析")

    # 主内容区
    if page == "视频分析":
        video_analysis_page()
    elif page == "历史记录":
        history_page()
    elif page == "系统统计":
        statistics_page()


def video_analysis_page():
    """视频分析页面"""
    st.header("📹 视频分析")

    # 文件上传
    uploaded_file = st.file_uploader(
        "上传跑步视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="支持常见视频格式"
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
            analyze_video(video_path)


def analyze_video(video_path: str):
    """执行视频分析"""
    try:
        # 进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1. 视频预处理
        status_text.text("1️⃣ 视频预处理中...")
        progress_bar.progress(10)
        processor = VideoProcessor(video_path)
        video_info = processor.get_video_info()
        frames, fps = processor.extract_frames(target_fps=30, max_frames=300)

        # 显示视频信息
        col1, col2, col3 = st.columns(3)
        col1.metric("分辨率", f"{video_info['width']}x{video_info['height']}")
        col2.metric("帧率", f"{video_info['fps']:.1f} FPS")
        col3.metric("时长", f"{video_info['duration']:.1f} 秒")

        # 2. 姿态估计
        status_text.text("2️⃣ 姿态估计中...")
        progress_bar.progress(30)
        estimator = PoseEstimator()
        keypoints_sequence = estimator.process_frames(frames)

        detected_count = sum(1 for kp in keypoints_sequence if kp['detected'])
        st.info(f"✓ 检测到 {detected_count}/{len(keypoints_sequence)} 帧")

        # ⭐ 姿态视频生成
        status_text.text("2️⃣ 生成姿态识别视频...")
        pose_video_path = generate_pose_video(frames, keypoints_sequence, fps)

        st.subheader("🦴 姿态识别（火柴人）视频")
        st.video(pose_video_path)

        # 3. 运动学分析
        status_text.text("3️⃣ 运动学分析中...")
        progress_bar.progress(50)
        kinematic_analyzer = KinematicAnalyzer()
        kinematic_results = kinematic_analyzer.analyze_sequence(keypoints_sequence, fps)

        # 4. 深度学习分析
        status_text.text("4️⃣ 深度学习分析中...")
        progress_bar.progress(70)
        temporal_analyzer = TemporalModelAnalyzer()
        temporal_results = temporal_analyzer.analyze(keypoints_sequence)

        # 5. 质量评价
        status_text.text("5️⃣ 技术质量评价中...")
        progress_bar.progress(85)
        quality_evaluator = QualityEvaluator()
        quality_results = quality_evaluator.evaluate(kinematic_results, temporal_results)

        # 6. AI文本生成
        status_text.text("6️⃣ AI文本分析中...")
        progress_bar.progress(95)
        results_for_ai = {
            'quality_evaluation': quality_results,
            'kinematic_analysis': kinematic_results,
            'temporal_analysis': temporal_results
        }
        ai_text = components['ai'].generate_analysis_report(results_for_ai)

        # 完成
        progress_bar.progress(100)
        status_text.text("✅ 分析完成!")

        # 显示结果
        st.markdown("---")
        display_results(quality_results, kinematic_results, temporal_results, ai_text)

        # 保存到数据库
        complete_results = {
            'video_info': video_info,
            'kinematic_analysis': kinematic_results,
            'temporal_analysis': temporal_results,
            'quality_evaluation': quality_results,
            'ai_analysis': ai_text
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

def generate_pose_video(frames, keypoints_sequence, fps):
    """
    将姿态骨架绘制到每一帧并生成视频（稳定版）
    """
    from pathlib import Path

    output_dir = Path("output/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "pose_visualization.mp4"

    h, w, _ = frames[0].shape
    fps = int(round(fps))  # ⭐ 关键：必须是 int

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (w, h)
    )

    if not writer.isOpened():
        raise RuntimeError("❌ VideoWriter 打开失败，无法生成视频")

    estimator = PoseEstimator()

    for frame, kp in zip(frames, keypoints_sequence):
        if kp.get("detected", False):
            vis_frame = estimator.visualize_pose(frame, kp)
        else:
            vis_frame = frame

        writer.write(vis_frame)

    writer.release()
    estimator.close()

    return str(output_path)


def display_results(quality, kinematic, temporal, ai_text):
    """显示分析结果"""
    st.header("📊 分析结果")

    # 总体评分
    st.subheader("🎯 总体评价")
    col1, col2 = st.columns([1, 2])

    with col1:
        score = quality['total_score']
        st.metric("技术质量评分", f"{score:.1f}",
                  delta=f"{quality['rating']}")

    with col2:
        st.markdown(f"**评级:** {quality['rating']}")
        st.markdown(f"**优势:** {', '.join(quality['strengths'])}")
        st.markdown(f"**薄弱项:** {', '.join(quality['weaknesses'])}")

    # 各维度得分
    st.subheader("📈 各维度表现")
    cols = st.columns(4)
    dimensions = quality['dimension_scores']

    cols[0].metric("稳定性", f"{dimensions['stability']:.1f}")
    cols[1].metric("效率", f"{dimensions['efficiency']:.1f}")
    cols[2].metric("跑姿", f"{dimensions['form']:.1f}")
    cols[3].metric("节奏", f"{dimensions['rhythm']:.1f}")

    # 运动学指标
    st.subheader("🔬 运动学指标")
    col1, col2, col3 = st.columns(3)

    col1.metric("步频", f"{kinematic['cadence']['cadence']:.1f} 步/分")
    col2.metric("步数", f"{kinematic['cadence']['step_count']}")
    col3.metric("垂直振幅", f"{kinematic['vertical_motion']['amplitude']:.1f} px")

    # 深度学习结果
    st.subheader("🤖 深度学习分析")
    col1, col2 = st.columns(2)

    col1.metric("AI质量评分", f"{temporal['quality_score']:.1f}")
    col2.metric("AI稳定性", f"{temporal['stability_score']:.1f}")

    phase_dist = temporal['phase_distribution']
    st.markdown(f"**阶段分布:** 触地 {phase_dist['ground_contact'] * 100:.1f}% | "
                f"腾空 {phase_dist['flight'] * 100:.1f}% | "
                f"过渡 {phase_dist['transition'] * 100:.1f}%")

    # 改进建议
    st.subheader("💡 改进建议")
    for i, suggestion in enumerate(quality['suggestions'], 1):
        st.markdown(f"{i}. {suggestion}")

    # AI分析文本
    st.subheader("📝 AI深度分析")
    st.markdown(ai_text)


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


if __name__ == '__main__':
    main()
