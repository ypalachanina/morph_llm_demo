import av
import threading
import time
from streamlit_webrtc import VideoProcessorBase


class FrameCaptureProcessor(VideoProcessorBase):
    """
    A video processor that captures frames, runs YOLO inference in a separate thread,
    and draws the results back onto the video stream.
    """

    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model

        # A lock to ensure thread-safe access to shared resources
        self.lock = threading.Lock()

        # Shared resources between the main (recv) and processing threads
        self.latest_frame = None
        self.latest_boxes = None

        # Attributes for the processing thread
        self.processing_thread = None
        self.stop_event = threading.Event()

        # Start the background processing thread if a model is provided
        if self.yolo_model:
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.start()

    def _processing_loop(self):
        """
        The main loop for the background processing thread.
        This is where the heavy YOLO inference happens.
        """
        while not self.stop_event.is_set():
            frame_to_process = None

            # Safely get the latest frame for processing
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()

            if frame_to_process is not None:
                try:
                    # Perform YOLO tracking. Note that the input is already BGR
                    # because we will draw on a BGR copy.
                    yolo_results = self.yolo_model.track(frame_to_process)

                    # Safely update the latest bounding boxes
                    with self.lock:
                        self.latest_boxes = yolo_results[0]
                except Exception as e:
                    # Handle any errors during YOLO processing
                    print(f"YOLO processing error: {e}")
                    with self.lock:
                        self.latest_boxes = None

            # Control the processing frequency to avoid pegging the CPU.
            # Adjust this value based on your model's performance and desired FPS.
            time.sleep(0.1)  # Process at roughly 10 FPS

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        This method is called for each frame from the webcam.
        It's responsible for receiving, processing, and returning frames.
        To keep the stream smooth, it should execute as quickly as possible.
        """
        # Convert the frame to a NumPy array in BGR format for OpenCV functions
        img_bgr = frame.to_ndarray(format="bgr24")

        # Update the latest_frame for the processing thread
        with self.lock:
            self.latest_frame = img_bgr.copy()
            # Get the most recent bounding boxes
            boxes = self.latest_boxes

        # Draw the bounding boxes on the frame if they exist
        if self.yolo_model and boxes is not None:
            # The draw_boxes function should operate on and return a BGR image
            frame_with_boxes = self.yolo_model.draw_boxes(img_bgr, boxes)
            # Convert the modified NumPy array back to a VideoFrame
            return av.VideoFrame.from_ndarray(frame_with_boxes, format="bgr24")
        else:
            # If no model or no boxes, return the original frame
            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    def get_latest_frame(self):
        """
        A utility function to get the latest raw frame for other purposes (e.g., saving a snapshot).
        """
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
