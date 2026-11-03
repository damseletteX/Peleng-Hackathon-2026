from __future__ import annotations

import cv2
import numpy as np

from mv3d.detection.detector import Detection


def draw_detection(
    image_bgr: np.ndarray,
    detection: Detection,
    xyz: np.ndarray | None = None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    x1, y1, x2, y2 = (round(v) for v in detection.bbox)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 255), 2)

    cx, cy = (round(v) for v in detection.center)
    cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)

    caption = f"{detection.label} {detection.score:.2f}"
    if xyz is not None:
        caption += f" | X={xyz[0]:.0f} Y={xyz[1]:.0f} Z={xyz[2]:.0f}"

    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_y = max(y1 - 8, th + 8)
    cv2.rectangle(
        canvas,
        (x1, text_y - th - 6),
        (x1 + tw + 8, text_y + 4),
        (0, 180, 255),
        -1,
    )
    cv2.putText(
        canvas,
        caption,
        (x1 + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas
