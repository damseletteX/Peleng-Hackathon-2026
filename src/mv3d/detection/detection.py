import argparse
import csv
from pathlib import Path

import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model("../../configs/cam1.mp4", show=True, conf=0.5, save=True)

TRUCK_CLASS_ID = 7  # Class ID for "truck" in the YOLO model

def detect_truck(video_path: str, 
    ):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened(): 
        raise IOError(f"Could not open video file: {video_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truck Detection using YOLOv8")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    args = parser.parse_args()

    detect_truck(args.video_path)