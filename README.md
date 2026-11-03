# Peleng Hackathon 2026

3D-трек грузовика по нескольким калиброванным камерам. Детектор — RF-DETR Medium, класс COCO truck. Точка считается в системе координат выданных таблиц, сантиметры.

Камеры, ролики и фокус: `configs/cameras.yml`. Поза каждой камеры читается на каждом кадре из `configs/calibration/`.

```bash
uv sync
uv run python -m mv3d csv
uv run python -m mv3d video
uv run python -m mv3d track
uv run python -m mv3d score outputs/track_<время>
```

`csv` пишет только таблицы и поэтому быстрее: нет копирования кадров и кодирования роликов. `video` пишет только размеченные `camN.mp4`. `track` пишет и то и другое. Детектор задаётся флагом `--model`. По умолчанию RF-DETR Medium. Её размеры: `nano`, `small`, `medium`, `large`. YOLO подключается тем же флагом, три варианта: `yolov8n`, `yolo26m`, `yolo11x`.

В режиме с CSV в `outputs/track_<время>/` появляются `cam1.csv`…`cam5.csv`, общий `track.csv`, `errors.csv` и `coordinates_time.png`. `score` считает MSE по уже готовому `track.csv`, без детектора.
