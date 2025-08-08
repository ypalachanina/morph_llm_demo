import streamlit as st
import streamlit.components.v1 as components
from audiorecorder import audiorecorder
from streamlit_webrtc import webrtc_streamer
from utils.webrtc_utils import FrameCaptureProcessor
from utils.storage_utils import list_azure_videos, get_video_url
from utils.audio_utils import text_to_speech, audio_to_base64
from utils.cv_utils import capture_frame, parse_timestamp, image_to_bytes
from utils.llm_utils import get_audio_description


def init_session(session):
    defaults = {
        "video_name": None,
        "video_url": None,
        "mode": "camera",
        "current_frame": None,
        "run_yolo": False
    }
    for k, v in defaults.items():
        if k not in session:
            session[k] = v


def display_sidebar(session):
    st.sidebar.markdown("## Input Source")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("📸 Live Camera", type="primary" if session["mode"] == 'camera' else "secondary"):
            session["mode"] = "camera"
            st.rerun()

    with col2:
        if st.button("🎞️ Video from Library", type="primary" if session["mode"] == 'video' else "secondary"):
            session["mode"] = "video"
            st.rerun()

    st.sidebar.markdown("## Response Language:")
    language = st.sidebar.selectbox("Select Language", ["English", "Nederlands", "Vlaams"], label_visibility="collapsed")
    session["language"] = language
    session["model_name"] = "gemini-2.5-flash"
    if session["mode"] == "camera":
        run_yolo = st.sidebar.checkbox("Show YOLO Bounding Boxes", value=session["run_yolo"])
        if run_yolo != session["run_yolo"]:
            session["run_yolo"] = run_yolo
            st.rerun()


def camera_mode(session):
    st.markdown("## 📸 Live Camera Input")
    run_yolo = session["run_yolo"]
    webrtc_ctx = webrtc_streamer(
        key="camera_streamer",
        video_processor_factory=lambda: FrameCaptureProcessor(run_yolo=run_yolo),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        process_audio(session, webrtc_ctx.video_processor)
    else:
        st.warning("Please start the camera.")


def video_mode(session):
    videos = list_azure_videos()
    if not videos:
        st.error("No videos found in Azure storage")
        return

    st.sidebar.markdown("## Select Video from Azure")
    default = "Choose video..."
    options = [default] + list(videos.keys())

    current_video = session.get("video_name", default)
    index = options.index(current_video) if current_video in options else 0
    video_name = st.sidebar.selectbox("Available Videos:", options, index=index)

    if video_name != default:
        if video_name != session.get("video_name"):
            session["video_name"] = video_name
            video = videos[video_name]
            st.sidebar.markdown(f"## Selected {video['desc']}")
            video_url = get_video_url(video["name"])
            if video_url:
                session["video_url"] = video_url
            else:
                st.error("Failed to load video")
                session["video_url"] = None

        if session.get("video_url"):
            st.sidebar.video(session["video_url"])
            st.markdown("## 🎞️️ Video Input")
            col_label, col_input = st.columns([1, 1])
            with col_label:
                st.write("**Timestamp:**")
            with col_input:
                tms = st.text_input("Enter Timestamp", value="00:10", key="timestamp", label_visibility="collapsed")
            if tms.strip():
                session["tms"] = parse_timestamp(tms)
                process_audio(session)


def process_audio(session, video_processor=None):
    col_label_audio, col_input_audio = st.columns([1, 1])
    with col_label_audio:
        st.write("**Ask Question:**")
    with col_input_audio:
        audio = audiorecorder("🎙️ Start recording", "🔴 Stop recording", key="audio")
    if len(audio) > 0:
        image = None
        if session["mode"] == "video":
            image = capture_frame(session.get("video_url"), session.get("tms"))
            if image is None:
                st.error("Failed to capture frame from video at the specified timestamp")
                return
        elif session["mode"] == "camera":
            if video_processor is None:
                st.error("Camera processor is not active. Please start the camera.")
                return
            image = video_processor.get_latest_frame()
            if image is None:
                st.error("Could not capture a frame from the camera.")
                return
        img_bytes = image_to_bytes(image)

        col_img, col_audio = st.columns(2)
        with col_img:
            st.image(image)
        with col_audio:
            st.audio(audio.export().read())

        with st.spinner("Preparing reply..."):
            audio_base64 = audio_to_base64(audio)
            desc = get_audio_description(img_bytes, audio_base64, session["model_name"], session["language"])
            st.markdown(desc)
            if desc:
                response_bytes, response_base64 = text_to_speech(desc, session["language"])
                if response_base64:
                    st.audio(response_bytes, format="audio/wav", start_time=0)
                    audio_html = f"""
                            <audio autoplay>
                              <source src="data:audio/wav;base64,{response_base64}" type="audio/wav" />
                              Your browser does not support the audio element.
                            </audio>
                            """
                    components.html(audio_html, height=1)
