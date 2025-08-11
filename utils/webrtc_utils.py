import av
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase
import threading


class FrameCaptureProcessor(VideoProcessorBase):
    def __init__(self, run_yolo=True):
        super().__init__()
        self.latest_frame = None
        self.lock = threading.Lock()
        self.run_yolo = run_yolo
        if self.run_yolo:
            self.yolo_model = YOLO('yolo11n.pt')
            self.tracker = "bytetrack.yaml"
            self.conf = 0.5
            self.img_size = 320
            self.counter = -1
            self.skip_frames = 5

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        if self.run_yolo:
            self.counter += 1
            if self.counter % self.skip_frames == 0:
                yolo_results = self.yolo_model.track(
                    img[..., ::-1],
                    imgsz=self.img_size,
                    conf=self.conf,
                    tracker=self.tracker,
                    persist=True,
                    verbose=False
                )
                frame_np = yolo_results[0].plot()[..., ::-1]
                frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                with self.lock:
                    self.latest_frame = frame
            return self.latest_frame
        else:
            with self.lock:
                self.latest_frame = img
            return frame

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy()
