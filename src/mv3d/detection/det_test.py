from pathlib import Path

import cv2
from ultralytics import YOLO

model = YOLO("yolo26s.pt")

VIDEO_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "cam1.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=1280,
        conf=0.25,
        verbose=False
    )

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])

            # COCO class 7 = truck
            if cls_id != 7:
                continue

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (int(center_x), int(center_y)),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"Truck {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Truck Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()