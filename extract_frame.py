import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="configs/cam1.mp4")
    parser.add_argument("--frame", type=int,default=0)
    args = parser.parse_args()

    if args.frame < 0:
        raise SystemExit("Номер кадра должен быть неотрицательным.")

    cap = cv2.VideoCapture(args.video)

    try:
        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть видео.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, image = cap.read()

        if not ok:
            raise RuntimeError(f"Не удалось прочитать кадр {args.frame}.")

        output = f"cam1_frame{args.frame}.png"

        if not cv2.imwrite(output, image):
            raise RuntimeError("Не удалось сохранить изображение.")

        print(f"Сохранено: {output}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
