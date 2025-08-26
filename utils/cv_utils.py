import cv2
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from ultralytics import YOLO, YOLOE
import streamlit as st


class YOLOModel:
    def __init__(self, weights):
        self.yolo_model = YOLO(weights["yolo_model"])
        self.yoloe_model_name = weights["yoloe_model"]
        self.tracker = "bytetrack.yaml"
        self.conf = 0.25
        self.yoloe_thr = 0.25
        self.imgsz = 640
        self.classes = self.yolo_model.names
        self.colors = self.generate_colors()

    def generate_colors(self, num_classes=None):
        color_pairs = []
        if num_classes is not None and num_classes <= 8:
            colors = [
                (0, 255, 255),  # Cyan
                (255, 255, 0),  # Yellow
                (128, 0, 128),  # Purple
                (255, 165, 0),  # Orange
                (255, 0, 255),  # Magenta
                (0, 255, 0),  # Green
                (0, 0, 255),  # Blue
                (255, 0, 0),  # Red
            ]
        else:
            num_classes = len(self.classes)
            colors = []
            for i in range(num_classes):
                hue = int(180 * i / num_classes)
                hsv = np.uint8([[[hue, 255, 255]]])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                color = tuple(int(c) for c in bgr)
                colors.append(color)
        for color in colors:
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
            color_pairs.append((color, text_color))
        return color_pairs

    def track(self, img_bgr):
        kwargs = dict(
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False
        )
        if self.tracker:
            kwargs["tracker"] = self.tracker
            kwargs["persist"] = True
            return self.yolo_model.track(img_bgr, **kwargs)
        else:
            return self.yolo_model.predict(img_bgr, **kwargs)

    def draw_boxes(self, img, results):
        for box in results.boxes:
            # show bounding boxes
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            cls_name = self.yolo_model.names[cls]
            color, text_color = self.colors[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # show labels
            label = f'{cls_name} {conf:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1), (x1 + label_width, y1 + label_height + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        return img

    def run_yoloe(self, img, class_names):
        if not class_names:
            return None
        yoloe_model = YOLOE(self.yoloe_model_name)
        yoloe_model.set_classes(class_names, yoloe_model.get_text_pe(class_names))
        results = yoloe_model.predict(img, imgsz=self.imgsz)
        colors = self.generate_colors(len(class_names))
        segmentation_data = []
        for r in results:
            if r.boxes is not None and r.masks is not None:
                boxes = r.boxes
                masks = r.masks
                for i in range(len(boxes)):
                    seg_info = {
                        "class_id": int(boxes.cls[i].cpu().numpy()),
                        "class_name": class_names[int(boxes.cls[i].cpu().numpy())],
                        "confidence": float(boxes.conf[i].cpu().numpy()),
                        "bbox": boxes.xyxy[i].cpu().numpy().astype(int).tolist(),
                        "mask": masks.data[i].cpu().numpy(),
                        "mask_area": np.sum(masks.data[i].cpu().numpy() > self.yoloe_thr)
                    }
                    seg_info["color"] = colors[seg_info["class_id"]]
                    segmentation_data.append(seg_info)
        return segmentation_data

    def draw_segmentation_on_image(self, img, results):
        h, w = img.shape[:2]
        for r in results:
            mask = r['mask']
            mask = cv2.resize(mask, (w, h))
            mask = cv2.GaussianBlur(mask, (31, 31), 0)

            x1, y1, x2, y2 = r['bbox']
            label = r['class_name']
            color, text_color = r['color']
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            mask_bool = mask > self.yoloe_thr
            mask_color = np.full_like(img[mask_bool], color)
            img[mask_bool] = cv2.addWeighted(img[mask_bool], 0.5, mask_color, 0.5, 0)

            border_mask = (mask_bool).astype(np.uint8) * 255
            contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 255, 255), 3)

            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1), (x1 + label_width, y1 + label_height + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        return img


@st.cache_resource
def create_yolo_model(weights):
    return YOLOModel(weights)


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
