import av
import threading
import time
import streamlit as st
from streamlit_webrtc import VideoProcessorBase
import cv2


class FrameCaptureProcessor(VideoProcessorBase):
    def __init__(self, yolo_model, show_bb):
        super().__init__()
        self.yolo_model = yolo_model
        self.show_bb = show_bb
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_boxes = None

        self.seg_results = None
        self.seg_timestamp = None
        self.seg_duration = 10

        self.processing_thread = None
        self.stop_event = threading.Event()
        if self.show_bb:
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
                    st.error(f"YOLO processing error: {e}")
                    with self.lock:
                        self.latest_boxes = None
            time.sleep(0.1)

    def set_segmentation_results(self, seg_results):
        with self.lock:
            self.seg_results = seg_results
            self.seg_timestamp = time.time()

    def _draw_text_overlay(self, img, text):
        img_with_text = img.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        bg_color = (255, 255, 255)
        text_color = (0, 0, 0)
        padding = 10

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        x = padding
        y = padding + text_height

        x1, y1 = x - padding, y - text_height - padding
        x2, y2 = x + text_width + padding, y + padding

        cv2.rectangle(img_with_text, (x1, y1), (x2, y2), bg_color, -1)
        cv2.putText(img_with_text, text, (x, y), font, font_scale, text_color, font_thickness)
        return img

    def _draw_segmentation(self, img, segmentation_results):
        img = self.yolo_model.draw_segmentation_on_image(img, segmentation_results)
        return img

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        with self.lock:
            self.latest_frame = img.copy()
            boxes = self.latest_boxes
            seg_results = self.seg_results
            seg_time = self.seg_timestamp
        if self.show_bb and boxes is not None:
            img = self.yolo_model.draw_boxes(img, boxes)
        if seg_results and seg_time:
            elapsed_time = time.time() - seg_time
            if elapsed_time <= self.seg_duration:
                img = self._draw_segmentation(img, seg_results)
            else:
                with self.lock:
                    self.seg_results = None
                    self.seg_timestamp = None
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        return frame

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def release(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.processing_thread.join()

        with self.lock:
            self.latest_frame = None
            self.latest_boxes = None
