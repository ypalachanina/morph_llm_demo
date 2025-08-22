import concurrent.futures
import streamlit as st
import streamlit.components.v1 as components
from audiorecorder import audiorecorder
from streamlit_webrtc import webrtc_streamer
from utils.secrets_utils import get_secrets
from utils.webrtc_utils import FrameCaptureProcessor
from utils.storage_utils import StorageClient
from utils.audio_utils import text_to_speech, audio_to_base64
from utils.cv_utils import create_yolo_model, capture_frame, parse_timestamp, image_to_bytes
from utils.llm_utils import LLM


MODEL_WEIGHTS = {
    "yolo_model": "yolo11n.pt",
    "yoloe_model": "yoloe-11l-seg.pt",
    "clip_model": "mobileclip_blt.ts"
}


class StreamlitUI:
    def __init__(self, session, host):
        self.session = session
        self.host = host
        self.video_processor = None

        defaults = {
            "video_name": None,
            "video_url": None,
            "mode": "camera",
            "current_frame": None,
            "show_bb": False
        }
        for k, v in defaults.items():
            if k not in self.session:
                self.session[k] = v

        self.session["host"] = self.host
        self.session["secrets"] = get_secrets(self.host)

        self.session["storage_client"] = StorageClient(self.session["secrets"])
        self.session["storage_client"].load_model_weights(MODEL_WEIGHTS)
        self.session["yolo_model"] = create_yolo_model(MODEL_WEIGHTS)

    def display_sidebar(self):
        st.sidebar.markdown("## Input Source")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("📸 Live Camera", type="primary" if self.session["mode"] == 'camera' else "secondary"):
                self.session["mode"] = "camera"
                st.rerun()

        with col2:
            if st.button("🎞️ Video from Library", type="primary" if self.session["mode"] == 'video' else "secondary"):
                self.session["mode"] = "video"
                st.rerun()

        st.sidebar.markdown("## Response Language:")
        language = st.sidebar.selectbox("Select Language", ["English", "Nederlands", "Vlaams"],
                                        label_visibility="collapsed")
        self.session["language"] = language
        self.session["model_name"] = "gemini-2.5-flash"
        self.session["LLM"] = LLM(self.session)
        if self.session["mode"] == "camera":
            show_bb = st.sidebar.checkbox("Show YOLO Bounding Boxes", value=self.session["show_bb"])
            if show_bb != self.session["show_bb"]:
                self.session["show_bb"] = show_bb
                st.rerun()

    def start_app(self):
        if self.session["mode"] == 'video':
            self.video_mode()
        elif self.session["mode"] == 'camera':
            self.camera_mode()

    def camera_mode(self):
        yolo_model = self.session["yolo_model"]
        show_bb = self.session["show_bb"]

        st.markdown("## 📸 Live Camera Input")
        webrtc_ctx = webrtc_streamer(
            key="camera_streamer",
            video_processor_factory=lambda: FrameCaptureProcessor(yolo_model, show_bb),
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            async_processing=True,
        )
        self.video_processor = webrtc_ctx.video_processor
        if webrtc_ctx.state.playing and self.video_processor:
            self.process_audio_and_image()
        else:
            st.warning("Please start the camera.")

    def video_mode(self):
        videos = self.session["storage_client"].list_azure_videos()
        if not videos:
            st.error("No videos found in Azure storage")
            return

        st.sidebar.markdown("## Select Video from Azure")
        default = "Choose video..."
        options = [default] + list(videos.keys())

        current_video = self.session.get("video_name", default)
        index = options.index(current_video) if current_video in options else 0
        video_name = st.sidebar.selectbox("Available Videos:", options, index=index)

        if video_name != default:
            if video_name != self.session.get("video_name"):
                self.session["video_name"] = video_name
                video = videos[video_name]
                st.sidebar.markdown(f"## Selected {video['desc']}")
                video_url = self.session["storage_client"].get_video_url(video["name"])
                if video_url:
                    self.session["video_url"] = video_url
                else:
                    st.error("Failed to load video")
                    self.session["video_url"] = None

            if self.session.get("video_url"):
                st.sidebar.video(self.session["video_url"])
                st.markdown("## 🎞️️ Video Input")
                col_label, col_input = st.columns([1, 1])
                with col_label:
                    st.write("**Timestamp:**")
                with col_input:
                    tms = st.text_input("Enter Timestamp", value="00:10", key="timestamp", label_visibility="collapsed")
                if tms.strip():
                    self.session["tms"] = parse_timestamp(tms)
                    self.process_audio_and_image()

    def get_image(self):
        image = None
        if self.session["mode"] == "video":
            image = capture_frame(self.session.get("video_url"), self.session.get("tms"))
        elif self.session["mode"] == "camera":
            if self.video_processor is None:
                st.error("Camera processor is not active. Please start the camera.")
                return
            image = self.video_processor.get_latest_frame()
            if image is None:
                st.error("Could not capture a frame from the camera.")
                return
        img_bytes = image_to_bytes(image)
        return image, img_bytes

    def search_objects(self, llm_model, yolo_model, image, audio_base64):
        is_list, objects, resp = llm_model.search_audio(audio_base64)
        seg_res = yolo_model.run_yoloe(image, objects) if objects else None
        return is_list, objects, resp, seg_res

    def process_parallel(self, audio, image, img_bytes):
        output = {}
        audio_base64 = audio_to_base64(audio)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            llm_model = self.session["LLM"]
            yolo_model = self.session["yolo_model"]
            search_future = executor.submit(
                self.search_objects,
                llm_model, yolo_model, image, audio_base64
            )
            description_future = executor.submit(
                llm_model.get_image_audio_description,
                img_bytes, audio_base64
            )
            try:
                is_list, objects, resp, seg_res = search_future.result()
                if not is_list:
                    output["warning"] = f"Output is not list! Response: {resp}"
                output["objects"] = objects
                output["segmentation"] = seg_res
                output["desc"] = description_future.result()
            except Exception as e:
                output["error"] = f"Error during LLM processing: {e}"
        return output

    def process_audio_and_image(self) -> None:
        col_label_audio, col_input_audio = st.columns([1, 1])
        with col_label_audio:
            st.write("**Ask Question:**")
        with col_input_audio:
            audio = audiorecorder("🎙️ Start recording", "🔴 Stop recording", key="audio")
        if len(audio) > 0:
            image, img_bytes = self.get_image()
            col_img, col_audio = st.columns(2)
            with col_img:
                st.image(image)
            with col_audio:
                st.audio(audio.export().read())
            with st.spinner("Processing audio and image..."):
                output = self.process_parallel(audio, image, img_bytes)
                if "error" in output:
                    st.error(output["error"])
                    return
                if "warning" in output:
                    st.warning(output["warning"])
                objects, seg_res = output["objects"], output["segmentation"]
                desc = output["desc"]
                st.markdown(f"Objects: {objects}")
                self.video_processor.set_segmentation_results(seg_res)
                # self.video_processor.set_display_text(str(objects))
                st.markdown(desc)
                response_bytes, response_base64 = text_to_speech(self.session, output['desc'])
                if response_base64:
                    st.audio(response_bytes, format="audio/wav", start_time=0)
                    audio_html = f"""
                        <audio autoplay>
                          <source src="data:audio/wav;base64,{response_base64}" type="audio/wav" />
                          Your browser does not support the audio element.
                        </audio>
                        """
                    components.html(audio_html, height=1)
