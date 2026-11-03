# Peleng Hackathon 2026

3D-трек грузовика по нескольким калиброванным камерам. Детектор — RF-DETR Nano, класс truck. Точка считается в системе координат выданных таблиц, сантиметры.

Камеры, ролики и фокус: `configs/cameras.yml`. Поза каждой камеры читается на каждом кадре из `configs/calibration/`.

```bash
uv sync
uv run python -m mv3d train
uv run python -m mv3d csv
uv run python -m mv3d video
uv run python -m mv3d track
uv run python -m mv3d score outputs/track_<время>
```

`csv` пишет только таблицы и поэтому быстрее: нет копирования кадров и кодирования роликов. `video` пишет только размеченные `camN.mp4`. `track` пишет и то и другое.

`train` размечает кадры предобученным nano. Дальше пять фолдов: каждый раз одна камера целиком уходит в проверку, на остальных четырёх кадр поворачивается, отражается, кропается и чуть меняется по цвету. В трекинг попадает фолд с лучшим mAP, файл `weights/rfdetr-nano-truck.pth`. Пока его нет, берётся предобученный COCO nano.

В режиме с CSV в `outputs/track_<время>/` появляются `cam1.csv`…`cam5.csv`, общий `track.csv`, `errors.csv` и `coordinates_time.png`. `score` считает MSE по уже готовому `track.csv`, без детектора.
