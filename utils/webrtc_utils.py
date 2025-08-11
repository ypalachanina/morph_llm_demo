import av
from streamlit_webrtc import VideoProcessorBase
import threading


class FrameCaptureProcessor(VideoProcessorBase):
    def __init__(self, yolo_model):
        super().__init__()
        self.latest_frame = None
        self.lock = threading.Lock()
        self.yolo_model = yolo_model
        if self.yolo_model:
            self.counter = 0
            self.skip_frames = 10
            self.latest_boxes = None

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        with self.lock:
            self.latest_frame = img.copy()
        if self.yolo_model:
            img_bgr = img[..., ::-1].copy()
            self.counter += 1
            if self.counter % self.skip_frames == 0:
                yolo_results = self.yolo_model.track(img_bgr)
                self.latest_boxes = yolo_results[0]
            if self.latest_boxes is not None:
                frame_np = self.yolo_model.draw_boxes(img_bgr, self.latest_boxes)
                frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        return frame

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy()
