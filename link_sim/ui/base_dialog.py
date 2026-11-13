"""模块对话框基类"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFrame, QSizePolicy, QLabel, QPushButton, QHBoxLayout,
)
from PySide6.QtCore import Qt
from link_sim.ui.styles import light_palette


class ModuleDialog(QDialog):
    MODEL_KIND   = None
    MODEL_FILE_ACTIONS = True
    TITLE        = "模块"
    SUBTITLE     = ""
    ACCENT_COLOR = "#00CFE8"
    MIN_WIDTH    = 680
    MIN_HEIGHT   = 480

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPalette(light_palette())
        self.setWindowTitle(self.TITLE)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        # 独立窗口 + 系统原生最小化/最大化/关闭按钮，非模态
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setModal(False)
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
        self._build_shell()

    def _build_shell(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        wrapper = QFrame()
        wrapper.setObjectName("moduleContent")
        wrapper.setStyleSheet("QFrame#moduleContent { background:#07131F; }")
        wlay = QVBoxLayout(wrapper)
        wlay.setContentsMargins(10, 8, 10, 10)
        wlay.setSpacing(8)

        self.content_layout = wlay
        self.build_content(wlay)
        if self.MODEL_KIND is not None:
            from link_sim.ui.model_library import ModelActions
            self.model_actions = ModelActions(self, file_actions=self.MODEL_FILE_ACTIONS)
            wlay.insertWidget(0, self.model_actions)

        if wlay.count() == 0:
            self._add_placeholder(wlay)

        root.addWidget(wrapper, stretch=1)

    def _add_placeholder(self, layout):
        ph = QFrame()
        ph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        phl = QVBoxLayout(ph)
        phl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        phl.setSpacing(10)

        icon = QLabel("⚙")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(
            f"font-size: 21pt;color:{self.ACCENT_COLOR};"
            f"background:{self._lighten(self.ACCENT_COLOR)};"
            "border-radius:14px;padding:14px;")
        icon.setFixedSize(60, 60)

        phl.addStretch()
        phl.addWidget(icon, alignment=Qt.AlignmentFlag.AlignCenter)
        phl.addSpacing(8)
        lbl = QLabel(f"{self.TITLE} — 开发中")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("font-size: 11pt;font-weight:500;color:#91AFC2;")
        phl.addWidget(lbl)
        desc = QLabel("该模块正在建设中，敬请期待。")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("font-size: 10pt;color:#648A9C;")
        phl.addWidget(desc)
        phl.addStretch()
        layout.addWidget(ph)

    def build_content(self, layout):
        pass

    def device_model_kind(self):
        return self.MODEL_KIND

    @staticmethod
    def _lighten(c):
        return {"#00CFE8": "#E6F1FB", "#1D9E75": "#E1F5EE",
                "#BA7517": "#FAEEDA", "#7F77DD": "#EEEDFE",
                "#D85A30": "#FAECE7"}.get(c, "#103C50")
