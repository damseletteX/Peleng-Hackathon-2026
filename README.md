# Peleng Hackathon 2026

3D-трек грузовика по нескольким калиброванным камерам. Детектор — RF-DETR Medium, класс COCO truck. Точка считается в системе координат выданных таблиц, сантиметры.

Камеры, ролики и фокус: `configs/cameras.yml`. Поза каждой камеры читается на каждом кадре из `configs/calibration/`.

```bash
uv sync
uv run python -m mv3d track
uv run python -m mv3d score outputs/track_<время>
uv run python -m mv3d detect configs/cam1.mp4 --camera cam1 --frame 150
```

`track` пишет в `outputs/track_<время>/` файлы `cam1.csv`…`cam5.csv`, общий `track.csv`, `errors.csv` и `coordinates_time.png`. `score` считает MSE по уже готовому `track.csv`, без детектора.
