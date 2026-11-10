"""Show startup progress before importing and opening the main window."""

from pathlib import Path
import sys
import traceback

try:
    from PySide6.QtCore import QCoreApplication, QTimer, Qt
    from PySide6.QtGui import QColor, QIcon, QPalette
    from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QProgressBar, QVBoxLayout, QWidget
except ImportError as exc:
    raise SystemExit("缺少 PySide6。请在启动软件的 Python 环境执行：python -m pip install --upgrade -r requirements.txt") from exc


APP_NAME = "星地馈电链路与星座仿真验证系统"


class StartupWindow(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowType.SplashScreen)
        self.setWindowTitle(APP_NAME)
        self.setFixedSize(500, 160)
        self.main_window = None
        self._finished = False
        self._module_warnings = []
        self.setStyleSheet("""
            QWidget { background: #07131f; color: #d9ecf7; }
            QProgressBar { background: #0d2638; border: 1px solid #24516a; border-radius: 5px; }
            QProgressBar::chunk { background: #1b9eb4; border-radius: 4px; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 20, 26, 20)
        title = QLabel(APP_NAME)
        title.setStyleSheet("font-size: 21px; font-weight: 600;")
        layout.addWidget(title)
        self.status = QLabel()
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(12)
        layout.addWidget(self.progress)
        self.map_timeout = QTimer(self)
        self.map_timeout.setSingleShot(True)
        self.map_timeout.setInterval(15000)
        self.map_timeout.timeout.connect(self.finish)
        self.update_progress(0, "正在准备启动…")

    def update_progress(self, value, message):
        self.progress.setValue(value)
        self.status.setText(message)
        self.status.repaint()
        self.progress.repaint()

    def load_modules(self):
        self.update_progress(10, "正在加载仿真模块…")
        try:
            from constellation_sim.ui.main_window import MainWindow
        except (Exception, SystemExit) as exc:
            self.fail(exc)
            return
        QTimer.singleShot(0, lambda: self.build_main_window(MainWindow))

    def build_main_window(self, window_type):
        self.update_progress(40, "正在初始化主界面…")
        try:
            self.main_window = window_type(load_map=False)
            self.main_window.web.loadProgress.connect(self.map_progress)
            self.main_window.web.loadFinished.connect(self.finish)
            from link_sim.ui.workspace import LazyDialogPage
            # The waveform canvas loads on first use; preloading NodeGraphQt
            # makes Qt.py probe optional Qt DLLs during every startup.
            self._preload_pages = [page for page in self.main_window.findChildren(LazyDialogPage)
                                   if page.class_name != "WaveformWorkspace"]
            self._preload_index = 0
            QTimer.singleShot(0, self.preload_next_page)
        except Exception as exc:
            self.fail(exc)

    def preload_next_page(self):
        """Build every link-simulation page before exposing the main window."""
        if self._preload_index == len(self._preload_pages):
            self.update_progress(80, "正在加载 Cesium 地图…")
            self.map_timeout.start()
            try:
                self.main_window.load_map()
            except Exception as exc:
                self.fail(exc)
            return
        page = self._preload_pages[self._preload_index]
        labels = {
            "PAModelDialog": "功放", "FilterModelDialog": "滤波器",
            "MixerModelDialog": "混频器", "ADDAModelDialog": "ADC/DAC",
            "NoiseModelDialog": "底噪模型", "ModelLibraryDialog": "模型库",
            "WaveformWorkspace": "全链路仿真",
        }
        label = labels.get(page.class_name, page.class_name)
        total = len(self._preload_pages)
        self.update_progress(45 + 35*self._preload_index//total,
                             f"正在初始化链路仿真 · {label}（{self._preload_index+1}/{total}）…")
        if page.loaded:
            self._preload_index += 1
            QTimer.singleShot(0, self.preload_next_page)
        else:
            page.load_finished.connect(self.preload_page_finished)
            page.ensure_loaded(background=True)

    def preload_page_finished(self, ok, message):
        page = self._preload_pages[self._preload_index]
        page.load_finished.disconnect(self.preload_page_finished)
        if not ok:
            self._module_warnings.append(f"{page.class_name}: {message}")
        self._preload_index += 1
        QTimer.singleShot(0, self.preload_next_page)

    def map_progress(self, value):
        self.update_progress(80 + 19*value//100, f"正在加载 Cesium 地图… {value}%")

    def finish(self, map_ok=None):
        if self._finished:
            return
        self._finished = True
        self.map_timeout.stop()
        self.main_window.web.loadProgress.disconnect(self.map_progress)
        self.main_window.web.loadFinished.disconnect(self.finish)
        if map_ok is None:
            self.main_window.status.setText("Cesium 地图仍在加载，可先使用其他功能")
        if self._module_warnings:
            self.main_window.status.setText(f"部分链路仿真页面未加载：{self._module_warnings[0]}")
        self.update_progress(100, "正在打开主界面…")
        self.main_window.showMaximized()
        self.close()

    def fail(self, exc):
        self.map_timeout.stop()
        traceback.print_exc()
        QMessageBox.critical(self, "启动失败", str(exc))
        QApplication.instance().exit(1)


def run_app():
    # Qt WebEngine is imported after the splash is visible.
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in {
        "Window": "#07131f", "WindowText": "#d9ecf7", "Base": "#091b29",
        "AlternateBase": "#0a1b2a", "Text": "#d9ecf7", "Button": "#0d2638",
        "ButtonText": "#d9ecf7", "Highlight": "#12617b", "HighlightedText": "#ffffff",
    }.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(color))
    app.setPalette(palette)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon(str(Path(__file__).with_name("assets") / "satellite.svg")))
    splash = StartupWindow()
    splash.move(app.primaryScreen().availableGeometry().center() - splash.rect().center())
    splash.show()
    app.processEvents()  # Paint the splash before the first synchronous import.
    QTimer.singleShot(0, splash.load_modules)
    sys.exit(app.exec())
