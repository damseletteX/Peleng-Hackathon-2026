import cv2
from pathlib import Path

# Get the directory where get_fps.py is located, then append the filename
VIDEO_PATH = Path(__file__).parent / "cam1.mp4"

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise IOError(f"Could not open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")

cap.release()