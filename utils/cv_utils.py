import cv2
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from ultralytics import YOLO
import streamlit as st


class YOLOModel:
    def __init__(self, model_name='yolo11n.pt'):
        self.yolo_model = YOLO(model_name)
        self.tracker = "bytetrack.yaml"
        self.conf = 0.4
        self.img_size = 320
        self.classes = self.yolo_model.names
        self.colors = self.generate_colors()

    def generate_colors(self):
        num_classes = len(self.classes)
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr))
        return colors

    def track(self, img_bgr):
        return self.yolo_model.track(
            img_bgr,
            imgsz=self.img_size,
            conf=self.conf,
            tracker=self.tracker,
            persist=True,
            verbose=False
        )

    def draw_boxes(self, img, results):
        # img = img.copy()[..., ::-1]
        for box in results.boxes:
            # show bounding boxes
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            cls_name = self.yolo_model.names[cls]
            color = self.colors[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # show labels
            brightness = sum(color) / 3  # use white text on dark colors, black on light colors
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
            label = f'{cls_name} {conf:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_height - 4), (x1 + label_width, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        return img[..., ::-1]


@st.cache_resource
def load_yolo_model(model_name='yolo11n.pt'):
    return YOLOModel(model_name)


def capture_frame(video_path, timestamp_s):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_no = int(timestamp_s * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame at {timestamp_s:.2f}s.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def image_to_bytes(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def parse_timestamp(tms):
    minutes = int(tms.split(':')[0])
    secs = int(tms.split(':')[1])
    ts = minutes * 60 + secs
    return ts
