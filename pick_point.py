import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="cam1_frame150.png")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Не удалось открыть: {args.image}")

    window = "Click truck center; Esc to exit"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"u={x}, v={y}", flush=True)

            preview = image.copy()
            cv2.drawMarker(
                preview,
                (x, y),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                20,
                2,
            )
            cv2.imshow(window, preview)

    cv2.setMouseCallback(window, on_click)
    cv2.imshow(window, image)

    while True:
        if cv2.waitKey(20) & 0xFF == 27:
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
