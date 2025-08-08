import cv2
from PIL import Image
import base64
from io import BytesIO
import numpy as np


def capture_frame(video_path, timestamp_s) -> Image.Image:
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
