import cv2
import av
from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import VideoProcessorBase
import threading


class FrameCaptureProcessor(VideoProcessorBase):
    def __init__(self, run_yolo=True):
        super().__init__()
        self.latest_frame = None
        self.lock = threading.Lock()
        self.run_yolo = run_yolo
        if self.run_yolo:
            self.yolo_model = YOLO('yolo11m.pt')
            self.tracker = "bytetrack.yaml"
            self.conf = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        with self.lock:
            self.latest_frame = img
        if self.run_yolo:
            yolo_results = self.yolo_model.track(
                img[..., ::-1],
                conf=0.5,
                tracker=self.tracker,
                persist=True,
                verbose=False
            )
            frame_bb = yolo_results[0].plot()[..., ::-1]
            return av.VideoFrame.from_ndarray(frame_bb, format="rgb24")
        return frame

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame
