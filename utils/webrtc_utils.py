import av
import threading
import time
from streamlit_webrtc import VideoProcessorBase


class FrameCaptureProcessor(VideoProcessorBase):
    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_boxes = None

        self.processing_thread = None
        self.stop_event = threading.Event()

        if self.yolo_model:
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.start()

    def _processing_loop(self):
        while not self.stop_event.is_set():
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            if frame_to_process is not None:
                try:
                    yolo_results = self.yolo_model.track(frame_to_process)
                    with self.lock:
                        self.latest_boxes = yolo_results[0]
                except Exception as e:
                    print(f"YOLO processing error: {e}")
                    with self.lock:
                        self.latest_boxes = None
                time.sleep(0.1)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.latest_frame = img.copy()
            boxes = self.latest_boxes
        if self.yolo_model and boxes is not None:
            frame_with_boxes = self.yolo_model.draw_boxes(img, boxes)
            frame = av.VideoFrame.from_ndarray(frame_with_boxes, format="bgr24")
        return frame

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def release(self):
        """
        This method is called when the stream is stopped.
        It's used for cleanup.
        """
        if self.processing_thread and self.processing_thread.is_alive():
            # Signal the processing thread to stop
            self.stop_event.set()
            # Wait for the thread to finish
            self.processing_thread.join()

        # Reset resources
        with self.lock:
            self.latest_frame = None
            self.latest_boxes = None
