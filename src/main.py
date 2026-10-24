import json
import math
import queue
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import sounddevice as sd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QFont,
    QFontDatabase,
    QImageReader,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VISEME_DIR = PROJECT_ROOT / "assets" / "visemes"
CUSTOM_VISEME_DIR = PROJECT_ROOT / "assets" / "custom_visemes"
MAPPING_PATH = Path(__file__).resolve().with_name("viseme_mapping.json")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class InferenceWorker(QThread):
    prediction_ready = Signal(str, float, object)
    input_level_changed = Signal(int)
    error_occurred = Signal(str)

    def __init__(
        self,
        device_index: int | None,
        noise_gate_dbfs: int,
    ) -> None:
        super().__init__()
        self.device_index = device_index
        self.noise_gate_dbfs = noise_gate_dbfs
        self._noise_gate_lock = threading.Lock()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def set_noise_gate_dbfs(self, threshold: int) -> None:
        with self._noise_gate_lock:
            self.noise_gate_dbfs = threshold

    def run(self) -> None:
        try:
            from src.inference import (
                HOP_LENGTH,
                SAMPLE_RATE,
                RollingMelCache,
                predict_latest_phoneme_from_features,
            )

            with MAPPING_PATH.open(encoding="utf-8") as mapping_file:
                viseme_mapping = json.load(mapping_file)

            audio_queue: queue.Queue = queue.Queue(maxsize=25)

            def audio_callback(indata, *_) -> None:
                samples = indata[:, 0].copy()
                rms = float((samples * samples).mean() ** 0.5)
                level_dbfs = 20 * math.log10(max(rms, 1e-6))
                self.input_level_changed.emit(
                    max(0, min(100, round((level_dbfs + 60) * 100 / 60)))
                )
                with self._noise_gate_lock:
                    noise_gate_dbfs = self.noise_gate_dbfs
                if level_dbfs < noise_gate_dbfs:
                    samples.fill(0)
                try:
                    audio_queue.put_nowait(samples)
                except queue.Full:
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    audio_queue.put_nowait(samples)

            cache = RollingMelCache()
            with sd.InputStream(
                device=self.device_index,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=2 * HOP_LENGTH,
                callback=audio_callback,
            ):
                while not self._stop_event.is_set():
                    try:
                        samples = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    features = cache.update(samples)
                    phoneme, confidence = (
                        predict_latest_phoneme_from_features(features)
                    )
                    if phoneme == "[UNK]":
                        viseme = None
                    else:
                        viseme = viseme_mapping.get(
                            phoneme,
                            viseme_mapping.get(
                                phoneme.strip("[]"),
                                "NEUTRAL",
                            ),
                        )
                    self.prediction_ready.emit(
                        phoneme,
                        confidence,
                        viseme,
                    )
        except Exception as exc:
            if not self._stop_event.is_set():
                self.error_occurred.emit(str(exc))


class ObsPreviewWindow(QLabel):
    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("loquela OBS Preview")
        self.setWindowFlags(
            Qt.Window
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMinMaxButtonsHint
        )
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: transparent; color: #181818;")
        self.setText("Waiting for viseme")
        self._source_pixmap = QPixmap()

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closed.emit()
        super().closeEvent(event)

    def _update_scaled_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            self.setPixmap(QPixmap())
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                self.contentsRect().size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class MainWindow(QMainWindow):
    def __init__(self, viseme_mapping: dict[str, str]) -> None:
        super().__init__()
        self.viseme_mapping = viseme_mapping
        self.viseme_names = sorted(set(viseme_mapping.values()) | {"NEUTRAL"})
        self.worker: InferenceWorker | None = None
        self.current_viseme: str | None = None
        self.source_pixmap = QPixmap()
        self.viseme_image_labels: dict[str, QLabel] = {}
        self.viseme_cards: dict[str, QWidget] = {}
        self.viseme_edit_buttons: dict[str, QPushButton] = {}
        self._close_requested = False

        self.setWindowTitle("loquela")
        self.resize(1180, 850)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(28, 24, 28, 24)
        root_layout.setSpacing(18)

        title = QLabel("loquela")
        title.setObjectName("wordmark")
        font_path = PROJECT_ROOT / "assets" / "fonts" / "Michroma-Regular.ttf"
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if not font_families:
            raise RuntimeError(f"Could not load Michroma font: {font_path}")
        wordmark_font = QFont(font_families[0])
        wordmark_font.setWeight(QFont.Weight.Bold)
        wordmark_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        title.setFont(wordmark_font)

        subtitle = QLabel(
            "Real time lip syncing, use default visemes or choose your own"
        )
        subtitle.setObjectName("subtitle")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        workspace_row = QHBoxLayout()
        workspace_row.setSpacing(20)

        viseme_panel = QWidget()
        viseme_panel.setMinimumWidth(500)
        viseme_layout = QVBoxLayout(viseme_panel)
        viseme_layout.setContentsMargins(0, 0, 0, 0)
        viseme_layout.setSpacing(10)

        viseme_heading = QLabel("Current visemes")
        viseme_heading.setObjectName("sectionTitle")
        viseme_layout.addWidget(viseme_heading)

        self.custom_enabled_check = QCheckBox("Enable custom viseme images")
        self.custom_enabled_check.toggled.connect(self._custom_mode_changed)
        viseme_layout.addWidget(self.custom_enabled_check)

        self.viseme_grid = QGridLayout()
        self.viseme_grid.setContentsMargins(0, 0, 0, 0)
        self.viseme_grid.setHorizontalSpacing(10)
        self.viseme_grid.setVerticalSpacing(10)
        for column in range(4):
            self.viseme_grid.setColumnStretch(column, 1)
        viseme_layout.addLayout(self.viseme_grid)

        for index, viseme in enumerate(self.viseme_names):
            card = QWidget()
            card.setObjectName("visemeCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(5)

            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setFixedSize(120, 64)
            image_label.setStyleSheet("background-color: #ffffff;")
            card_layout.addWidget(image_label, alignment=Qt.AlignHCenter)

            name_label = QLabel(viseme)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setObjectName("visemeName")
            card_layout.addWidget(name_label)

            edit_button = QPushButton("Choose image…")
            edit_button.setObjectName("visemeEdit")
            edit_button.setEnabled(False)
            edit_button.clicked.connect(
                lambda *_, name=viseme: self._import_viseme_image(
                    name
                )
            )
            card_layout.addWidget(edit_button)

            self.viseme_cards[viseme] = card
            self.viseme_image_labels[viseme] = image_label
            self.viseme_edit_buttons[viseme] = edit_button
            self.viseme_grid.addWidget(card, index // 4, index % 4)

        workspace_row.addWidget(viseme_panel)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(10)
        preview_heading = QLabel("Preview")
        preview_heading.setObjectName("sectionTitle")
        preview_layout.addWidget(preview_heading)

        self.preview_label = QLabel("Start listening to preview visemes")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(440, 200)
        self.preview_label.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #454545; "
            "border-radius: 12px; color: #181818;"
        )
        preview_layout.addWidget(self.preview_label, 1)
        workspace_row.addWidget(preview_panel, 1)
        root_layout.addLayout(workspace_row, 1)

        self.prediction_label = QLabel("Phoneme: --   Confidence: --   Viseme: --")
        self.prediction_label.setObjectName("prediction")
        root_layout.addWidget(self.prediction_label)

        audio_row = QHBoxLayout()
        audio_row.addWidget(QLabel("Microphone"))
        self.microphone_combo = QComboBox()
        self.microphone_combo.setMinimumWidth(300)
        audio_row.addWidget(self.microphone_combo, 1)
        self.refresh_devices_button = QPushButton("Refresh")
        self.refresh_devices_button.clicked.connect(self._refresh_devices)
        audio_row.addWidget(self.refresh_devices_button)
        root_layout.addLayout(audio_row)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Mic level"))
        self.input_level_bar = QProgressBar()
        self.input_level_bar.setObjectName("inputLevel")
        self.input_level_bar.setRange(0, 100)
        self.input_level_bar.setValue(0)
        self.input_level_bar.setTextVisible(False)
        self.input_level_bar.setFixedHeight(18)
        self.input_level_bar.setAccessibleName("Microphone input level")
        meter_control = QWidget()
        meter_control.setMinimumHeight(28)
        meter_layout = QGridLayout(meter_control)
        meter_layout.setContentsMargins(0, 0, 0, 0)
        meter_layout.setSpacing(0)
        meter_layout.addWidget(self.input_level_bar, 0, 0)
        input_row.addWidget(meter_control, 1)
        self.noise_gate_slider = QSlider(Qt.Horizontal)
        self.noise_gate_slider.setObjectName("noiseGate")
        self.noise_gate_slider.setRange(-60, 0)
        self.noise_gate_slider.setValue(-45)
        self.noise_gate_slider.setAccessibleName("Noise gate threshold")
        self.noise_gate_slider.setToolTip(
            "Audio quieter than this threshold is muted."
        )
        self.noise_gate_slider.valueChanged.connect(
            self._noise_gate_changed
        )
        meter_layout.addWidget(self.noise_gate_slider, 0, 0)
        input_row.addWidget(QLabel("Noise gate"))
        self.noise_gate_value = QLabel()
        input_row.addWidget(self.noise_gate_value)
        input_row.addWidget(QLabel("Quieter input is muted"))
        root_layout.addLayout(input_row)

        self._noise_gate_changed(self.noise_gate_slider.value())

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start listening")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start_listening)
        button_row.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_listening)
        button_row.addWidget(self.stop_button)

        self.obs_button = QPushButton("Open OBS preview")
        self.obs_button.setCheckable(True)
        self.obs_button.toggled.connect(self._toggle_obs_preview)
        button_row.addWidget(self.obs_button)
        button_row.addStretch()
        root_layout.addLayout(button_row)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("subtitle")
        root_layout.addWidget(self.status_label)

        self.obs_preview = ObsPreviewWindow()
        self.obs_preview.closed.connect(self._obs_preview_closed)

        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #181818;
                color: #e5e5e5;
                font-size: 14px;
            }
            QLabel#wordmark {
                color: #ffffff;
                font-size: 50px;
            }
            QLabel#subtitle {
                color: #aaaaaa;
            }
            QLabel#prediction {
                font-size: 18px;
                font-weight: 600;
                padding: 4px 0;
            }
            QLabel#sectionTitle {
                font-size: 17px;
                font-weight: 600;
            }
            QLabel#visemeName {
                font-weight: 600;
            }
            QProgressBar#inputLevel {
                min-height: 16px;
                border: 1px solid #414141;
                border-radius: 5px;
                background-color: #292929;
            }
            QProgressBar#inputLevel::chunk {
                border-radius: 4px;
                background-color: #43d17b;
            }
            QSlider#noiseGate {
                min-height: 28px;
                background-color: transparent;
            }
            QSlider#noiseGate::groove:horizontal {
                height: 18px;
                background-color: transparent;
            }
            QSlider#noiseGate::handle:horizontal {
                width: 14px;
                margin: 0;
                border: 1px solid #d0d0d0;
                border-radius: 7px;
                background-color: #f0f0f0;
            }
            QSlider#noiseGate::sub-page:horizontal {
                border-radius: 2px;
                background-color: #f39c12;
            }
            QSlider#noiseGate::add-page:horizontal {
                background-color: transparent;
            }
            QWidget#visemeCard {
                border: 1px solid #414141;
                border-radius: 8px;
                background-color: #222222;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #666666;
                border-radius: 3px;
                background-color: #292929;
            }
            QCheckBox::indicator:hover {
                border-color: #aaaaaa;
                background-color: #393939;
            }
            QCheckBox::indicator:checked {
                border-color: #b0b0b0;
                background-color: #858585;
            }
            QComboBox, QPushButton {
                min-height: 34px;
                padding: 2px 10px;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                background-color: #2b2b2b;
            }
            QPushButton {
                color: #e5e5e5;
            }
            QPushButton#visemeEdit {
                min-height: 20px;
                max-height: 22px;
                padding: 0 3px;
                font-size: 10px;
            }
            QPushButton:hover, QComboBox:hover {
                border-color: #777777;
                background-color: #414141;
            }
            QPushButton:pressed {
                border-color: #909090;
                background-color: #555555;
            }
            QPushButton:checked {
                border-color: #a0a0a0;
                background-color: #555555;
                color: #ffffff;
            }
            QPushButton:checked:hover {
                background-color: #666666;
            }
            QPushButton:disabled {
                border-color: #363636;
                background-color: #222222;
                color: #777777;
            }
            QPushButton#primary {
                background-color: #555555;
                border-color: #777777;
                font-weight: 600;
            }
            QPushButton#primary:hover {
                background-color: #6b6b6b;
                border-color: #999999;
            }
            QPushButton#primary:pressed {
                background-color: #3d3d3d;
                border-color: #aaaaaa;
            }
            QPushButton#primary:disabled {
                background-color: #292929;
                border-color: #3b3b3b;
                color: #777777;
            }
            QComboBox QAbstractItemView {
                background-color: #252525;
                color: #e5e5e5;
                selection-background-color: #555555;
                selection-color: #ffffff;
            }
            """
        )
        self._refresh_devices()
        self._refresh_viseme_grid()
        self._refresh_preview()

    def _refresh_devices(self) -> None:
        selected_device_name = self.microphone_combo.currentText()
        try:
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            default_device = sd.default.device[0]
        except (sd.PortAudioError, TypeError, IndexError) as exc:
            QMessageBox.warning(
                self,
                "Microphones unavailable",
                f"Could not list audio input devices:\n{exc}",
            )
            return

        self.microphone_combo.clear()

        default_name = None
        if isinstance(default_device, int) and 0 <= default_device < len(devices):
            default_name = devices[default_device]["name"]
        self.microphone_combo.addItem(
            f"System default{': ' + default_name if default_name else ''}",
            None,
        )

        for index, device in enumerate(devices):
            if device["max_input_channels"] <= 0:
                continue
            self.microphone_combo.addItem(
                f"{device['name']}",
                index,
            )

        selected_index = self.microphone_combo.findText(selected_device_name)
        if selected_index >= 0:
            self.microphone_combo.setCurrentIndex(selected_index)
        elif isinstance(default_device, int):
            default_index = self.microphone_combo.findData(default_device)
            if default_index >= 0:
                self.microphone_combo.setCurrentIndex(default_index)

    def _start_listening(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        self.worker = InferenceWorker(
            self.microphone_combo.currentData(),
            self.noise_gate_slider.value(),
        )
        self.worker.prediction_ready.connect(self._show_prediction)
        self.worker.input_level_changed.connect(self.input_level_bar.setValue)
        self.worker.error_occurred.connect(self._show_worker_error)
        self.worker.finished.connect(self._worker_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.microphone_combo.setEnabled(False)
        self.refresh_devices_button.setEnabled(False)
        self.status_label.setText("Starting microphone and loading model…")
        self.worker.start()

    def _stop_listening(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        self.worker.stop()
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping…")

    def _worker_finished(self) -> None:
        self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.microphone_combo.setEnabled(True)
        self.refresh_devices_button.setEnabled(True)
        self.input_level_bar.setValue(0)
        if self.status_label.text() not in ("Microphone or inference error",):
            self.status_label.setText("Ready")
        if self._close_requested:
            self.close()

    def _noise_gate_changed(self, threshold: int) -> None:
        self.noise_gate_value.setText(f"{threshold} dBFS")
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_noise_gate_dbfs(threshold)

    def _show_worker_error(self, message: str) -> None:
        self.status_label.setText("Microphone or inference error")
        QMessageBox.critical(
            self,
            "Could not start lip sync",
            message,
        )

    def _show_prediction(
        self,
        phoneme: str,
        confidence: float,
        viseme: str | None,
    ) -> None:
        displayed_viseme = viseme or self.current_viseme or "NEUTRAL"
        self.prediction_label.setText(
            f"Phoneme: {phoneme}   Confidence: {confidence:.2f}   "
            f"Viseme: {displayed_viseme}"
        )
        self.status_label.setText("Listening")
        if viseme is not None and viseme != self.current_viseme:
            self.current_viseme = viseme
            self._refresh_viseme_grid()
            self._refresh_preview()

    def _find_viseme_image(self, viseme: str) -> Path | None:
        directories = [DEFAULT_VISEME_DIR]
        if self.custom_enabled_check.isChecked():
            directories.insert(0, CUSTOM_VISEME_DIR)

        for directory in directories:
            for extension in IMAGE_EXTENSIONS:
                image_path = directory / f"{viseme}{extension}"
                if image_path.is_file():
                    return image_path
        return None

    def _refresh_viseme_grid(self) -> None:
        for viseme, image_label in self.viseme_image_labels.items():
            image_path = self._find_viseme_image(viseme)
            if image_path is None:
                image_label.setPixmap(QPixmap())
                image_label.setText("No image")
            else:
                pixmap = QPixmap(str(image_path))
                if pixmap.isNull():
                    image_label.setPixmap(QPixmap())
                    image_label.setText("Could not load")
                else:
                    image_label.setText("")
                    image_label.setPixmap(
                        pixmap.scaled(
                            image_label.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation,
                        )
                    )

            self.viseme_edit_buttons[viseme].setEnabled(
                self.custom_enabled_check.isChecked()
            )
            self.viseme_cards[viseme].setStyleSheet(
                "QWidget#visemeCard { border: 2px solid #a0a0a0; "
                "border-radius: 8px; background-color: #303030; }"
                if viseme == self.current_viseme
                else ""
            )

    def _custom_mode_changed(self, enabled: bool) -> None:
        self._refresh_viseme_grid()
        self._refresh_preview()
        mode = "Custom viseme images enabled" if enabled else "Using default visemes"
        self.status_label.setText(mode)

    def _refresh_preview(self, _: int = 0) -> None:
        viseme = self.current_viseme or "NEUTRAL"
        image_path = self._find_viseme_image(viseme)
        if image_path is None:
            self.source_pixmap = QPixmap()
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(f"No image found for {viseme}")
            self.obs_preview.setText(f"No image found for {viseme}")
            self.obs_preview.set_source_pixmap(QPixmap())
            return

        self.source_pixmap = QPixmap(str(image_path))
        if self.source_pixmap.isNull():
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(f"Could not load {image_path.name}")
            self.obs_preview.setText(f"Could not load {image_path.name}")
            self.obs_preview.set_source_pixmap(QPixmap())
            return

        self._update_scaled_preview()
        self.obs_preview.setText("")
        self.obs_preview.set_source_pixmap(self.source_pixmap)

    def _update_scaled_preview(self) -> None:
        if self.source_pixmap.isNull():
            return
        self.preview_label.setText("")
        self.preview_label.setPixmap(
            self.source_pixmap.scaled(
                self.preview_label.contentsRect().size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_preview()

    def _import_viseme_image(self, viseme: str) -> None:
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a viseme image",
            str(PROJECT_ROOT),
            "Image files (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not image_path:
            return

        source_path = Path(image_path)
        if not QImageReader(str(source_path)).canRead():
            QMessageBox.warning(
                self,
                "Invalid image",
                "The selected file is not a readable image.",
            )
            return

        extension = source_path.suffix.lower()
        if extension not in IMAGE_EXTENSIONS:
            QMessageBox.warning(
                self,
                "Unsupported image",
                "Choose a PNG, JPG, JPEG, WEBP, or BMP image.",
            )
            return

        try:
            CUSTOM_VISEME_DIR.mkdir(parents=True, exist_ok=True)
            existing_images = [
                path
                for path in CUSTOM_VISEME_DIR.glob(f"{viseme}.*")
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            destination = CUSTOM_VISEME_DIR / f"{viseme}{extension}"

            if source_path.resolve() == destination.resolve():
                self.status_label.setText(f"Custom image selected for {viseme}")
                self._refresh_viseme_grid()
                if self.current_viseme == viseme:
                    self._refresh_preview()
                return

            if existing_images:
                answer = QMessageBox.question(
                    self,
                    "Replace custom viseme?",
                    f"A custom image already exists for {viseme}. Replace it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    return

            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=CUSTOM_VISEME_DIR,
                    prefix=f".{viseme}_",
                    suffix=extension,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                shutil.copy2(source_path, temporary_path)
                temporary_path.replace(destination)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()

            for old_image in existing_images:
                if old_image != destination and old_image.exists():
                    old_image.unlink()
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Could not add image",
                f"Could not save the custom viseme image:\n{exc}",
            )
            return

        self.status_label.setText(f"Custom image added for {viseme}")
        self._refresh_viseme_grid()
        if self.current_viseme == viseme or self.current_viseme is None:
            self._refresh_preview()

    def _toggle_obs_preview(self, visible: bool) -> None:
        if visible:
            self.obs_preview.show()
            self.obs_button.setText("Close OBS preview")
        else:
            self.obs_preview.hide()
            self.obs_button.setText("Open OBS preview")

    def _obs_preview_closed(self) -> None:
        self.obs_button.setChecked(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._close_requested = True
            self.worker.stop()
            event.ignore()
            return
        self.obs_preview.close()
        super().closeEvent(event)


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    application = QApplication(sys.argv)
    try:
        with MAPPING_PATH.open(encoding="utf-8") as mapping_file:
            viseme_mapping = json.load(mapping_file)
        if not isinstance(viseme_mapping, dict) or not all(
            isinstance(phoneme, str) and isinstance(viseme, str)
            for phoneme, viseme in viseme_mapping.items()
        ):
            raise ValueError("Viseme mapping must be a JSON object of strings.")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        QMessageBox.critical(
            None,
            "Could not load viseme mapping",
            f"Unable to read {MAPPING_PATH}:\n{exc}",
        )
        return 1

    window = MainWindow(viseme_mapping)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
