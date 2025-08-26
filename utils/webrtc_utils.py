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
        self.seg_classes = None
        self.latest_seg_results = None
        self.seg_timestamp = None
        self.seg_duration = 20  # seconds

        self.processing_thread = None
        self.stop_event = threading.Event()
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()

    def _processing_loop(self):
        while not self.stop_event.is_set():
            frame_to_process = None
            local_show_bb = False
            local_seg_classes = None

            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
                local_show_bb = self.show_bb
                local_seg_classes = self.seg_classes

            if frame_to_process is not None:
                try:
                    if local_show_bb:
                        yolo_results = self.yolo_model.track(frame_to_process)
                        with self.lock:
                            self.latest_boxes = yolo_results[0]
                    else:
                        with self.lock:
                            self.latest_boxes = None

                    if local_seg_classes:
                        segmentation_results = self.yolo_model.run_yoloe(frame_to_process, local_seg_classes)
                        with self.lock:
                            self.latest_seg_results = segmentation_results
                    else:
                        with self.lock:
                            self.latest_seg_results = None
                except Exception as e:
                    st.error(f"YOLO processing error: {e}")
                    with self.lock:
                        self.latest_boxes = None
                        self.latest_seg_results = None
            time.sleep(0.01)

    def set_seg_classes(self, seg_classes):
        with self.lock:
            self.seg_classes = seg_classes
            self.latest_seg_results = None
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
            seg_results = self.latest_seg_results
            seg_classes = self.seg_classes
            seg_time = self.seg_timestamp

        if self.show_bb and boxes is not None:
            img = self.yolo_model.draw_boxes(img, boxes)

        if seg_classes and seg_results is not None and seg_time is not None:
            elapsed_time = time.time() - seg_time
            if elapsed_time <= self.seg_duration:
                img = self._draw_segmentation(img, seg_results)
            else:
                with self.lock:
                    self.seg_classes = None
                    self.latest_seg_results = None
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
            self.latest_seg_results = None