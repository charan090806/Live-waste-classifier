import numpy as np
import cv2

def process_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodes the uploaded image bytes into a numpy array (BGR format for OpenCV/YOLO)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image
