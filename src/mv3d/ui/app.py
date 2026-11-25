from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mv3d.ui.session import (
    DEFAULT_CAMERAS,
    CameraEntry,
    load_camera_entries,
    save_camera_list,
    startup_cameras,
    write_run_config,
)

_CONFIG = Path("configs/cameras.yml")
_DEFAULT_GT = Path("configs/calibration/object_transform_SM_AA_track.csv")
_FRAME_LINE = re.compile(
    r"^frame (\d+): (?:нет точки, views=(\d+)|X=([-\d.]+) Y=([-\d.]+) Z=([-\d.]+).*)"
)
_STYLE = """
QMainWindow, QWidget { background-color: #121212; color: #e0e0e0;
    font-family: "Helvetica Neue", "Segoe UI"; font-size: 13px; }
QGroupBox {
    background-color: #1a1a1a; border: 1px solid #2c2c2c; border-radius: 12px;
    margin-top: 14px; padding: 22px 12px 12px 12px; font-weight: bold;
    font-size: 14px; color: #ffffff;
}
QGroupBox::title { subcontrol-position: top left; subcontrol-origin: margin; left: 14px; top: 4px; }
QPushButton {
    background-color: #262626; color: #ffffff; border: none; border-radius: 8px;
    padding: 10px 16px; font-weight: 600; min-height: 24px;
}
QPushButton:hover { background-color: #333333; }
QPushButton:disabled { background-color: #1c1c1c; color: #666666; }
QPushButton#accentButton { background-color: #ccff00; color: #121212; }
QPushButton#accentButton:hover { background-color: #b3e600; }
QPushButton#accentButton:disabled { background-color: #3d4a1a; color: #6d7848; }
QPushButton#secondaryButton { background-color: #3a3a3a; color: #ffffff; }
QPushButton#secondaryButton:hover { background-color: #4a4a4a; }
QTableWidget {
    background-color: #181818; border: 1px solid #2c2c2c; border-radius: 8px;
    gridline-color: #2c2c2c; color: #e0e0e0;
}
QTableWidget::item:selected { background-color: #2e3b1c; color: #ccff00; }
QHeaderView::section {
    background-color: #202020; color: #aaaaaa; padding: 8px 6px; border: none;
    border-bottom: 1px solid #2c2c2c; border-right: 1px solid #2c2c2c; font-weight: bold;
}
QSlider::groove:horizontal { height: 6px; background: #2c2c2c; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #ccff00; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #ffffff; width: 14px; margin: -4px 0; border-radius: 7px;
}
QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit {
    background-color: #202020; border: 1px solid #333; border-radius: 6px;
    padding: 4px; color: white;
}
QCheckBox { spacing: 8px; font-weight: 500; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #555; background: #202020; }
QCheckBox::indicator:checked { background: #ccff00; border: 1px solid #ccff00; }
QTabWidget::pane { border: 1px solid #2c2c2c; border-radius: 8px; background: #141414; }
QTabBar::tab {
    background: #1a1a1a; color: #aaaaaa; padding: 8px 16px; border-top-left-radius: 8px;
    border-top-right-radius: 8px; margin-right: 4px;
}
QTabBar::tab:selected { background: #262626; color: #ccff00; }
QProgressBar {
    background: #2c2c2c; border: none; border-radius: 4px; height: 8px; text-align: center; color: transparent;
}
QProgressBar::chunk { background: #ccff00; border-radius: 4px; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: #1a1a1a; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #4a4a4a; min-height: 28px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #ccff00; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


class _Stream:
    def __init__(self, camera_id: str, file_path: str):
        self.camera_id = camera_id
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened() or int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0:
            self.cap.release()
            raise ValueError(f"Файл не читается: {file_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if self.fps <= 0:
            self.fps = 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._last = -1

    def frame(self, index: int) -> np.ndarray | None:
        if not 0 <= index < self.total_frames:
            return None
        if index == self._last + 1:
            ok, image = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = self.cap.read()
        if not ok:
            return None
        self._last = index
        return image

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()


class _Job(QThread):
    logged = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        config: Path,
        frames: int,
        csv: bool,
        plots: bool,
        video: bool,
        gt_path: str,
    ):
        super().__init__()
        self.config = config
        self.frames = frames
        self.csv = csv
        self.plots = plots
        self.video = video
        self.gt_path = gt_path
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        from mv3d.tracking.pipeline import run_tracking

        previous = sys.stdout
        sys.stdout = _Log(self.logged)
        try:
            output = run_tracking(
                config_path=self.config,
                frames=self.frames,
                output_root="outputs",
                write_csv=self.csv,
                write_video=self.video,
                write_plots=self.plots,
                should_stop=lambda: self._stop,
                gt_path=self.gt_path,
            )
            self.finished_ok.emit(str(output))
        except SystemExit as exc:
            self.failed.emit(str(exc.code or exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc) or "Прогон прерван")
        finally:
            sys.stdout.flush()
            sys.stdout = previous


class _Log:
    def __init__(self, signal):
        self.signal = signal
        self.buf = ""

    def write(self, text: str) -> int:
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.signal.emit(line)
        return len(text)

    def flush(self) -> None:
        if self.buf.strip():
            self.signal.emit(self.buf.strip())
        self.buf = ""


class TrackerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.streams: dict[str, _Stream] = {}
        self.labels: dict[str, QLabel] = {}
        self.catalog = _catalog()
        self.frame_idx = 0
        self.playing = False
        self.job: _Job | None = None
        self.last_output: Path | None = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._advance)
        self.setWindowTitle("3D Tracker — Peleng 2026")
        self.resize(1360, 860)
        self.setStyleSheet(_STYLE)
        self._build()

    def _build(self) -> None:
        root = QWidget()
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_col = QVBoxLayout(left)
        left_col.setContentsMargins(0, 0, 8, 0)

        files = QGroupBox("Список камер")
        files_col = QVBoxLayout(files)
        self.catalog_table = QTableWidget(0, 4)
        self.catalog_table.setHorizontalHeaderLabels(["Имя", "Калибровка", "Фокус, мм", ""])
        self.catalog_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.catalog_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.catalog_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.catalog_table.verticalHeader().setVisible(False)
        self.catalog_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.catalog_table.setMinimumHeight(280)
        files_col.addWidget(self.catalog_table, stretch=1)
        catalog_actions = QHBoxLayout()
        add_camera = QPushButton("Добавить")
        add_camera.clicked.connect(self._add_camera)
        remove_camera = QPushButton("Удалить")
        remove_camera.clicked.connect(self._remove_camera)
        save_list = QPushButton("Сохранить в конфиг")
        save_list.setObjectName("accentButton")
        save_list.clicked.connect(self._save_catalog)
        reset_list = QPushButton("Исходные 5")
        reset_list.clicked.connect(self._reset_default)
        catalog_actions.addWidget(add_camera)
        catalog_actions.addWidget(remove_camera)
        catalog_actions.addWidget(reset_list)
        catalog_actions.addWidget(save_list)
        files_col.addLayout(catalog_actions)
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Сколько снять"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, max(len(self.catalog), 1))
        self.count_spin.setValue(max(len(self.catalog), 1))
        count_row.addWidget(self.count_spin)
        create = QPushButton("Таблица видео")
        create.clicked.connect(lambda: self._create_table(save=True))
        count_row.addWidget(create)
        files_col.addLayout(count_row)
        hint = QLabel(
            "Список хранится в configs/cameras.yml: имя, калибровка и фокус. "
            "Ниже к каждой камере крепится видео с любым именем файла. "
            "Все выбранные ролики попадут в один track.csv."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8d8d8d; font-weight: 400;")
        files_col.addWidget(hint)
        left_col.addWidget(files, stretch=3)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Камера", "Файл", "", "FPS", "Кадры"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(180)
        left_col.addWidget(self.table, stretch=1)

        outputs = QGroupBox("Что посчитать")
        out_col = QVBoxLayout(outputs)
        self.csv_box = QCheckBox("CSV: track.csv, camN.csv, errors.csv")
        self.plot_box = QCheckBox("Графики X, Y, Z от кадра и вид сверху")
        self.video_box = QCheckBox("Видео с баундинг-боксом")
        for box in (self.csv_box, self.plot_box, self.video_box):
            box.setChecked(True)
            out_col.addWidget(box)
        frames_row = QHBoxLayout()
        frames_row.addWidget(QLabel("Кадров, 0 = все"))
        self.frames = QSpinBox()
        self.frames.setRange(0, 100_000)
        self.frames.setValue(0)
        frames_row.addWidget(self.frames)
        out_col.addLayout(frames_row)
        gt_row = QHBoxLayout()
        gt_row.addWidget(QLabel("Эталон"))
        self.gt_edit = QLineEdit(str(_DEFAULT_GT) if _DEFAULT_GT.is_file() else "")
        self.gt_edit.setPlaceholderText("Необязательно: CSV с правильными X Y Z")
        gt_row.addWidget(self.gt_edit, stretch=1)
        pick_gt = QPushButton("CSV")
        pick_gt.clicked.connect(self._pick_gt)
        clear_gt = QPushButton("Убрать")
        clear_gt.clicked.connect(self.gt_edit.clear)
        gt_row.addWidget(pick_gt)
        gt_row.addWidget(clear_gt)
        out_col.addLayout(gt_row)
        actions = QHBoxLayout()
        self.run_btn = QPushButton("Запустить")
        self.run_btn.setObjectName("accentButton")
        self.run_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.open_btn = QPushButton("Открыть папку")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_folder)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.stop_btn)
        actions.addWidget(self.open_btn)
        out_col.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        out_col.addWidget(self.progress)
        left_col.addWidget(outputs)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Здесь будет ход прогона, кадр за кадром.")
        self.log.setMinimumHeight(260)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        left_col.addWidget(self.log)

        player = QGroupBox("Плеер")
        player_col = QVBoxLayout(player)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        player_col.addWidget(self.slider)
        play_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.frame_spin = QSpinBox()
        self.frame_spin.valueChanged.connect(self._on_spin)
        play_row.addWidget(self.play_btn)
        play_row.addWidget(QLabel("Кадр"))
        play_row.addWidget(self.frame_spin)
        player_col.addLayout(play_row)
        left_col.addWidget(player)

        self.tabs = QTabWidget()
        preview = QWidget()
        self.grid = QGridLayout(preview)
        preview_scroll = _scroll(preview)
        self.tabs.addTab(preview_scroll, "Камеры")
        graph = QWidget()
        graph_col = QVBoxLayout(graph)
        self.time_plot = QLabel("График X, Y, Z появится после прогона.")
        self.time_plot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_plot.setMinimumHeight(280)
        self.time_plot.setStyleSheet("background: #181818; border-radius: 8px; color: #888;")
        self.top_plot = QLabel()
        self.top_plot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top_plot.setMinimumHeight(180)
        self.top_plot.setStyleSheet("background: #181818; border-radius: 8px;")
        graph_col.addWidget(self.time_plot, stretch=2)
        graph_col.addWidget(self.top_plot, stretch=1)
        self.tabs.addTab(graph, "Графики")
        self.coord_page = QWidget()
        coord_col = QVBoxLayout(self.coord_page)
        self.coord_readout = QLabel("3D-координаты появятся по ходу прогона.")
        self.coord_readout.setWordWrap(True)
        self.coord_readout.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #ccff00; padding: 8px 4px;"
        )
        coord_col.addWidget(self.coord_readout)
        self.metrics = QLabel("MSE и RMSE появятся, если выбран эталон.")
        self.metrics.setWordWrap(True)
        self.metrics.setStyleSheet("color: #e0e0e0; padding: 0 4px 8px 4px;")
        coord_col.addWidget(self.metrics)
        self.coords = QTableWidget(0, 5)
        self.coords.setHorizontalHeaderLabels(["Кадр", "X", "Y", "Z", "Ошибка, см"])
        self.coords.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.coords.verticalHeader().setVisible(False)
        self.coords.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.coords.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.coords.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        coord_col.addWidget(self.coords)
        self.tabs.addTab(self.coord_page, "3D координаты")
        self._saw_coordinate = False
        self.space_page = QWidget()
        space_col = QVBoxLayout(self.space_page)
        caption = QLabel("Общий путь грузовика и следы каждой камеры в одном пространстве.")
        caption.setStyleSheet("color: #ccff00; font-weight: 600;")
        space_col.addWidget(caption)
        self.space_view = QLabel("Траектория в пространстве появится после прогона.")
        self.space_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.space_view.setMinimumHeight(420)
        self.space_view.setStyleSheet("background: #121212; border-radius: 8px; color: #888;")
        space_col.addWidget(self.space_view, stretch=1)
        space_row = QHBoxLayout()
        self.space_play = QPushButton("Движение")
        self.space_play.clicked.connect(self._toggle_space)
        self.space_slider = QSlider(Qt.Orientation.Horizontal)
        self.space_slider.valueChanged.connect(self._on_space_slider)
        space_row.addWidget(self.space_play)
        space_row.addWidget(self.space_slider, stretch=1)
        space_col.addLayout(space_row)
        self.tabs.addTab(self.space_page, "Пространство")
        self.space_timer = QTimer(self)
        self.space_timer.timeout.connect(self._advance_space)
        self._space_fused: list = []
        self._space_cameras: dict = {}

        self.left_scroll = _scroll(left)
        split.addWidget(self.left_scroll)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(split)
        self.setCentralWidget(root)
        self._fill_catalog(startup_cameras(_CONFIG))
        self._create_table(save=False)

    def _fill_catalog(self, entries: list[CameraEntry] | None = None) -> None:
        if entries is None:
            entries = load_camera_entries(_CONFIG)
        self.catalog_table.setRowCount(0)
        for entry in entries:
            self._append_catalog_row(entry.camera_id, entry.calibration, entry.focal_mm)
        self._sync_count_limit()
        self.count_spin.setValue(max(len(entries), 1))
        self._startup_videos = {entry.camera_id: entry.video for entry in entries}

    def _reset_default(self) -> None:
        self._fill_catalog(list(DEFAULT_CAMERAS))
        self._create_table(save=False)
        self._note("Исходный тест: cam1–cam5.")

    def _append_catalog_row(self, camera_id: str, calibration: str, focal_mm: float) -> None:
        row = self.catalog_table.rowCount()
        self.catalog_table.insertRow(row)
        name = QLineEdit(camera_id)
        self.catalog_table.setCellWidget(row, 0, name)
        label = QTableWidgetItem(Path(calibration).name if calibration else "не выбрана")
        label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
        label.setData(Qt.ItemDataRole.UserRole, calibration)
        label.setToolTip(calibration)
        self.catalog_table.setItem(row, 1, label)
        focal = QDoubleSpinBox()
        focal.setRange(1.0, 500.0)
        focal.setDecimals(1)
        focal.setValue(focal_mm)
        self.catalog_table.setCellWidget(row, 2, focal)
        button = QPushButton("Файл")
        button.clicked.connect(lambda _checked=False, index=row: self._pick_calibration(index))
        self.catalog_table.setCellWidget(row, 3, button)

    def _catalog_entries(self) -> list[CameraEntry]:
        entries: list[CameraEntry] = []
        for row in range(self.catalog_table.rowCount()):
            name = self.catalog_table.cellWidget(row, 0).text().strip()
            calibration = self.catalog_table.item(row, 1).data(Qt.ItemDataRole.UserRole) or ""
            focal = float(self.catalog_table.cellWidget(row, 2).value())
            entries.append(CameraEntry(name, str(calibration), focal))
        return entries

    def _sync_count_limit(self) -> None:
        count = max(self.catalog_table.rowCount(), 1)
        self.count_spin.setMaximum(count)
        if self.catalog_table.rowCount() and self.count_spin.value() > self.catalog_table.rowCount():
            self.count_spin.setValue(self.catalog_table.rowCount())
        self.catalog = [entry.camera_id for entry in self._catalog_entries()]

    def _add_camera(self) -> None:
        used = {entry.camera_id for entry in self._catalog_entries()}
        number = 1
        while f"cam{number}" in used:
            number += 1
        self._append_catalog_row(f"cam{number}", "", 35.0)
        self._sync_count_limit()

    def _remove_camera(self) -> None:
        selected = self.catalog_table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "Список", "Выберите камеру в списке.")
            return
        self.catalog_table.removeRow(selected[0].row())
        self._sync_count_limit()

    def _pick_calibration(self, row: int) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Калибровка", "configs/calibration", "CSV (*.csv)"
        )
        if not path:
            return
        item = self.catalog_table.item(row, 1)
        item.setText(Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)

    def _save_catalog(self) -> bool:
        try:
            entries = self._catalog_entries()
            save_camera_list(_CONFIG, entries)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            QMessageBox.warning(self, "Конфиг", str(exc))
            return False
        self.catalog = [entry.camera_id for entry in entries]
        self._sync_count_limit()
        self._note(f"Список сохранён: {', '.join(self.catalog)}")
        return True

    def _create_table(self, save: bool = True) -> None:
        if save and not self._save_catalog():
            return
        kept: dict[str, str] = dict(getattr(self, "_startup_videos", {}))
        for row in range(self.table.rowCount()):
            camera = self.table.item(row, 0).text()
            path = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
            if path:
                kept[camera] = str(path)
        self._release_streams()
        self.table.setRowCount(0)
        for camera in self.catalog[: self.count_spin.value()]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            title = QTableWidgetItem(camera)
            title.setFlags(title.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, title)
            file_item = QTableWidgetItem("не выбрано")
            file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, file_item)
            button = QPushButton("Выбрать")
            button.clicked.connect(lambda _checked=False, index=row: self._pick(index))
            self.table.setCellWidget(row, 2, button)
            for col in (3, 4):
                blank = QTableWidgetItem("—")
                blank.setFlags(blank.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, blank)
            video = kept.get(camera)
            if video and Path(video).is_file():
                self._attach_video(row, camera, video)
        self._startup_videos = {}
        self._sync_timeline()
        self._layout_grid()
        self._show_frames()

    def _pick(self, row: int) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Видео камеры", "", "Видео (*.mp4 *.avi *.mkv *.mov)"
        )
        if not path:
            return
        self._attach_video(row, self.table.item(row, 0).text(), path)
        self._sync_timeline()
        self._layout_grid()
        self._show_frames()

    def _attach_video(self, row: int, camera: str, path: str) -> None:
        previous = self.streams.pop(camera, None)
        if previous is not None:
            previous.release()
        try:
            stream = _Stream(camera, path)
        except ValueError as exc:
            QMessageBox.critical(self, "Видео", str(exc))
            return
        self.streams[camera] = stream
        file_item = self.table.item(row, 1)
        file_item.setText(Path(path).name)
        file_item.setData(Qt.ItemDataRole.UserRole, path)
        self.table.item(row, 3).setText(f"{stream.fps:.1f}")
        self.table.item(row, 4).setText(str(stream.total_frames))

    def _release_streams(self) -> None:
        for stream in self.streams.values():
            stream.release()
        self.streams.clear()

    def _max_frame(self) -> int:
        if not self.streams:
            return 0
        return max(stream.total_frames for stream in self.streams.values()) - 1

    def _sync_timeline(self) -> None:
        last = max(self._max_frame(), 0)
        self.slider.setMaximum(last)
        self.frame_spin.setMaximum(last)
        if self.frame_idx > last:
            self.frame_idx = 0

    def _layout_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.labels.clear()
        for index, key in enumerate(self.streams):
            label = QLabel(key)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(320, 180)
            label.setStyleSheet(
                "background-color: #181818; color: #888; border: 1px solid #282828; border-radius: 8px;"
            )
            self.grid.addWidget(label, index // 2, index % 2)
            self.labels[key] = label

    def _show_frames(self) -> None:
        if not self.streams:
            return
        for key, stream in self.streams.items():
            label = self.labels.get(key)
            if label is None:
                continue
            image = stream.frame(min(self.frame_idx, stream.total_frames - 1))
            if image is None:
                label.setText(key)
                continue
            rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            height, width, channels = rgb.shape
            picture = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
            label.setPixmap(
                QPixmap.fromImage(picture).scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _on_slider(self, value: int) -> None:
        self.frame_idx = value
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value)
        self.frame_spin.blockSignals(False)
        self._show_frames()

    def _on_spin(self, value: int) -> None:
        self.frame_idx = value
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self._show_frames()

    def _toggle_play(self) -> None:
        if not self.streams:
            return
        self.playing = not self.playing
        if self.playing:
            self.timer.start(33)
            self.play_btn.setText("Pause")
        else:
            self.timer.stop()
            self.play_btn.setText("Play")

    def _advance(self) -> None:
        if self.frame_idx >= self._max_frame():
            self._toggle_play()
            return
        self.slider.setValue(self.frame_idx + 1)

    def _assignments(self) -> dict[str, Path]:
        if self.table.rowCount() == 0:
            raise ValueError("Сначала создайте таблицу камер.")
        entries = self._catalog_entries()
        if self.table.rowCount() > len(entries):
            raise ValueError("Таблица видео длиннее списка. Создайте её заново.")
        chosen: dict[str, Path] = {}
        missing: list[str] = []
        for row in range(self.table.rowCount()):
            camera = entries[row].camera_id
            self.table.item(row, 0).setText(camera)
            raw = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
            if not raw:
                missing.append(camera)
                continue
            chosen[camera] = Path(raw)
        if missing:
            raise ValueError(f"Не выбрано видео: {', '.join(missing)}")
        return chosen

    def _start(self) -> None:
        if not (self.csv_box.isChecked() or self.plot_box.isChecked() or self.video_box.isChecked()):
            QMessageBox.warning(self, "Прогон", "Выберите CSV, графики или видео.")
            return
        self.log.clear()
        self.coords.setRowCount(0)
        self.coord_readout.setText("3D-координаты появятся по ходу прогона.")
        self.metrics.setText("MSE и RMSE появятся, если выбран эталон.")
        self._saw_coordinate = False
        self.left_scroll.ensureWidgetVisible(self.log)
        if not self._save_catalog():
            return
        try:
            assignments = self._assignments()
        except ValueError as exc:
            QMessageBox.warning(self, "Прогон", str(exc))
            return
        dest = Path("outputs") / "_ui_cameras.yml"
        try:
            write_run_config(_CONFIG, assignments, dest)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            QMessageBox.critical(self, "Конфиг", str(exc))
            return
        self._note(f"В общий CSV: {', '.join(assignments)}")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.show()
        self.job = _Job(
            dest,
            self.frames.value(),
            self.csv_box.isChecked(),
            self.plot_box.isChecked(),
            self.video_box.isChecked(),
            self.gt_edit.text().strip(),
        )
        self.job.logged.connect(self._note)
        self.job.finished_ok.connect(self._done)
        self.job.failed.connect(self._fail)
        self.job.start()

    def _stop(self) -> None:
        if self.job is not None:
            self.job.stop()
            self._note("Остановка после текущего кадра.")

    def _done(self, folder: str) -> None:
        self.last_output = Path(folder)
        self._finish_job()
        self.open_btn.setEnabled(True)
        self._show_plots(self.last_output)
        self._load_track(self.last_output)
        self._show_space(self.last_output)
        self.tabs.setCurrentWidget(self.space_page)
        self._note(f"Готово: {folder}")

    def _fail(self, message: str) -> None:
        self._finish_job()
        self._note(message)
        QMessageBox.critical(self, "Прогон", message)

    def _finish_job(self) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.hide()

    def _show_space(self, folder: Path) -> None:
        from mv3d.ui.scene import load_tracks

        self.space_timer.stop()
        self.space_play.setText("Движение")
        self._space_fused, self._space_cameras = load_tracks(folder)
        last = max(len(self._space_fused) - 1, 0)
        self.space_slider.blockSignals(True)
        self.space_slider.setMaximum(last)
        self.space_slider.setValue(last)
        self.space_slider.blockSignals(False)
        picture = self._paint_space()
        if picture is not None:
            path = folder / "space.png"
            cv2.imwrite(str(path), picture)

    def _paint_space(self) -> np.ndarray | None:
        from mv3d.ui.scene import render_space

        if len(self._space_fused) < 2:
            self.space_view.setText("Мало точек для сцены.")
            self.space_view.setPixmap(QPixmap())
            return None
        width = max(self.space_view.width(), 960)
        height = max(self.space_view.height(), 640)
        image = render_space(
            self._space_fused,
            self._space_cameras,
            width=width,
            height=height,
            cursor=self.space_slider.value(),
        )
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        view_h, view_w, channels = rgb.shape
        frame = QImage(rgb.data, view_w, view_h, channels * view_w, QImage.Format.Format_RGB888).copy()
        self.space_view.setPixmap(
            QPixmap.fromImage(frame).scaled(
                self.space_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        return image

    def _on_space_slider(self, _value: int) -> None:
        self._paint_space()

    def _toggle_space(self) -> None:
        if len(self._space_fused) < 2:
            return
        if self.space_timer.isActive():
            self.space_timer.stop()
            self.space_play.setText("Движение")
            return
        if self.space_slider.value() >= self.space_slider.maximum():
            self.space_slider.setValue(0)
        self.space_timer.start(40)
        self.space_play.setText("Пауза")

    def _advance_space(self) -> None:
        if self.space_slider.value() >= self.space_slider.maximum():
            self.space_timer.stop()
            self.space_play.setText("Движение")
            return
        self.space_slider.setValue(self.space_slider.value() + 1)

    def _show_plots(self, folder: Path) -> None:
        self._set_plot(self.time_plot, folder / "coordinates_time.png", "Нет графика координат.")
        self._set_plot(self.top_plot, folder / "trajectory_topdown.png", "")

    def _set_plot(self, label: QLabel, path: Path, empty: str) -> None:
        if not path.is_file():
            label.setText(empty)
            label.setPixmap(QPixmap())
            return
        picture = QPixmap(str(path))
        width = max(label.width(), 860)
        height = max(label.height(), 220)
        label.setPixmap(
            picture.scaled(
                width,
                height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _open_folder(self) -> None:
        if self.last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output.resolve())))

    def _pick_gt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Эталон координат", "configs/calibration", "CSV (*.csv)"
        )
        if path:
            self.gt_edit.setText(path)

    def _note(self, text: str) -> None:
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        if text.startswith("MSE"):
            current = self.metrics.text()
            if current.startswith("MSE и RMSE"):
                current = ""
            self.metrics.setText(text if not current else f"{current}\n{text}")
        self._capture_coordinate(text)

    def _capture_coordinate(self, text: str) -> None:
        match = _FRAME_LINE.match(text.strip())
        if match is None:
            return
        frame = int(match.group(1))
        if match.group(2) is not None:
            cells = ("", "", "", "")
        else:
            cells = (match.group(3), match.group(4), match.group(5), "")
            self.coord_readout.setText(
                f"Кадр {frame}    X={cells[0]}   Y={cells[1]}   Z={cells[2]}"
            )
        self._append_coord_row(frame, *cells)
        self.coords.scrollToBottom()
        if not self._saw_coordinate and cells[0]:
            self._saw_coordinate = True
            self.tabs.setCurrentWidget(self.coord_page)

    def _append_coord_row(self, frame: int, x: str, y: str, z: str, error: str) -> None:
        row = self.coords.rowCount()
        self.coords.insertRow(row)
        for column, value in enumerate((str(frame), x, y, z, error)):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.coords.setItem(row, column, item)

    def _load_track(self, folder: Path) -> None:
        track = folder / "track.csv"
        if not track.is_file():
            return
        errors: dict[int, str] = {}
        error_file = folder / "errors.csv"
        if error_file.is_file():
            for line in error_file.read_text(encoding="utf-8").splitlines()[1:]:
                parts = [item.strip() for item in line.split(";")]
                if len(parts) >= 5 and parts[0]:
                    errors[int(float(parts[0]))] = parts[4]
        self.coords.setRowCount(0)
        last: tuple[int, list[str]] | None = None
        for line in track.read_text(encoding="utf-8").splitlines()[1:]:
            parts = [item.strip() for item in line.split(";")]
            if not parts or not parts[0]:
                continue
            frame = int(float(parts[0]))
            xyz = parts[1:4] if len(parts) >= 4 else ["", "", ""]
            while len(xyz) < 3:
                xyz.append("")
            self._append_coord_row(frame, xyz[0], xyz[1], xyz[2], errors.get(frame, ""))
            if any(xyz):
                last = (frame, xyz)
        if last is not None:
            frame, xyz = last
            self.coord_readout.setText(
                f"Кадр {frame}    X={xyz[0]}   Y={xyz[1]}   Z={xyz[2]}"
            )
        self.coords.scrollToBottom()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.last_output is not None:
            self._show_plots(self.last_output)
            self._paint_space()

    def closeEvent(self, event) -> None:
        if self.job is not None and self.job.isRunning():
            self.job.stop()
            self.job.wait(3000)
        self._release_streams()
        super().closeEvent(event)


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return area


def _catalog() -> list[str]:
    try:
        found = [entry.camera_id for entry in load_camera_entries(_CONFIG)]
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError):
        found = []
    return found or ["cam1", "cam2", "cam3", "cam4", "cam5"]


def main() -> int:
    app = QApplication(sys.argv)
    window = TrackerWindow()
    window.show()
    return app.exec()
