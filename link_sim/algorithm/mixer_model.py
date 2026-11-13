"""仅考虑本振相位噪声的等效复中频混频器模型。"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSizePolicy, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from link_sim.ui.base_dialog import ModuleDialog
from link_sim.ui.measured_model_page import MeasuredModelPage
from .mixer_noise_core import generate_phase_noise, phase_noise_psd as phase_noise_psd


# 默认值对应用户给出的 MATLAB 相噪模板：(频偏 Hz, L(f) dBc/Hz)
_DEFAULT_PN = ((1e3, -60.0), (1e4, -80.0), (1e5, -110.0), (1e6, -130.0))
_ACCENT = "#22C7E6"


def _setup_font():
    available = {font.name for font in fm.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"):
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return


_setup_font()


def parse_pn_anchors(text):
    """解析 ``1k:-60, 10k:-80, 100k:-110`` 格式的相噪锚点。"""
    if not text or not text.strip():
        return _DEFAULT_PN
    multiplier = {"k": 1e3, "K": 1e3, "M": 1e6, "m": 1e6, "G": 1e9, "g": 1e9}
    anchors = []
    for item in text.replace("\n", ",").split(","):
        if not item.strip() or ":" not in item:
            continue
        offset_text, level_text = (part.strip() for part in item.split(":", 1))
        try:
            suffix = offset_text[-1]
            offset = float(offset_text[:-1]) * multiplier[suffix] if suffix in multiplier else float(offset_text)
            anchors.append((offset, float(level_text)))
        except (IndexError, KeyError, ValueError):
            continue
    anchors.sort(key=lambda point: point[0])
    if len(anchors) < 2 or any(offset <= 0 for offset, _ in anchors):
        raise ValueError("相噪锚点至少需要两组有效的“频偏:电平”数据。")
    if len({offset for offset, _ in anchors}) != len(anchors):
        raise ValueError("相噪锚点的频偏不能重复。")
    return tuple(anchors)


def blackman_harris(length):
    """四项 Blackman-Harris 窗。"""
    if length < 2:
        return np.ones(length)
    n = np.arange(length)
    phase = 2 * np.pi * n / (length - 1)
    return (
        0.35875 - 0.48829 * np.cos(phase) + 0.14128 * np.cos(2 * phase)
        - 0.01168 * np.cos(3 * phase)
    )


def _integrated_phase_error(offsets, psd_single):
    """σφ = sqrt(2·∫ Sφ(f)df)，其中 Sφ 是双边 PSD 的正频率一侧。"""
    return float(np.sqrt(2.0 * np.trapezoid(psd_single, offsets)))


def _complex_spectrum(x, fs):
    n_samples = len(x)
    window = blackman_harris(n_samples)
    coherent_gain = np.mean(window)
    spectrum = np.fft.fftshift(np.fft.fft(x * window) / (n_samples * coherent_gain))
    frequency = np.fft.fftshift(np.fft.fftfreq(n_samples, 1.0 / fs))
    enbw_hz = fs * np.sum(window ** 2) / np.sum(window) ** 2
    return frequency, spectrum, enbw_hz


def run_mixer_phase_noise(fs, sim_time, f_if_in, anchors):
    """运行 MATLAB 同类的等效复中频相噪混频仿真。

    输入为复单音 ``exp(j·2π·f_IF·t)``。本振在等效复基带中为
    ``exp(-j·φ[n])``，因此输出为 ``输入 × 带相位噪声本振``。除相位噪声
    外，不包含变频损耗、热噪声、非线性、LO 泄漏或杂散。
    """
    fs = float(fs)
    sim_time = float(sim_time)
    f_if_in = float(f_if_in)
    n_samples = int(np.floor(fs * sim_time))
    if fs <= 0 or sim_time <= 0 or n_samples < 256:
        raise ValueError("采样率和仿真时长必须为正，且采样点数不得小于 256。")
    if not 0 < f_if_in < fs / 2:
        raise ValueError("等效输入中频必须位于 0 与 Nyquist 频率之间。")
    if max(offset for offset, _ in anchors) >= fs / 2:
        raise ValueError("最高相噪锚点必须低于 Nyquist 频率。")

    time = np.arange(n_samples) / fs
    input_signal = np.exp(1j * 2 * np.pi * f_if_in * time)
    phi, pn_offsets, pn_levels, pn_psd = generate_phase_noise(fs, n_samples, anchors)
    lo = np.exp(-1j * phi)
    output = input_signal * lo
    ideal_output = input_signal
    frequency, spectrum, enbw_hz = _complex_spectrum(output, fs)
    _, ideal_spectrum, _ = _complex_spectrum(ideal_output, fs)
    reference_db = 20 * np.log10(np.max(np.abs(ideal_spectrum)) + 1e-30)
    output_db = 20 * np.log10(np.abs(spectrum) + 1e-30) - reference_db
    ideal_db = 20 * np.log10(np.abs(ideal_spectrum) + 1e-30) - reference_db
    rbw_db = 10 * np.log10(enbw_hz)
    rms_phase_template = _integrated_phase_error(pn_offsets[1:], pn_psd[1:])
    rms_phase_actual = float(np.std(phi))

    return {
        "fs": fs,
        "sim_time": sim_time,
        "n_samples": n_samples,
        "t": time,
        "f_if_in": f_if_in,
        "phi": phi,
        "output": output,
        "ideal_output": ideal_output,
        "frequency": frequency,
        "offset_from_carrier": frequency - f_if_in,
        "output_dbc_hz": output_db - rbw_db,
        "ideal_dbc_hz": ideal_db - rbw_db,
        "enbw_hz": enbw_hz,
        "pn_offsets": pn_offsets,
        "pn_levels": pn_levels,
        "pn_psd": pn_psd,
        "anchors": anchors,
        "rms_phase_template_rad": rms_phase_template,
        "rms_phase_actual_rad": rms_phase_actual,
        "evm_pct": rms_phase_actual * 100.0,
    }


_EDIT_STYLE = (
    "QLineEdit{background:#091B29;border:1px solid #1C4961;border-radius:3px;"
    "padding:3px 6px;font-size:10pt;color:#D9ECF7;}"
    f"QLineEdit:focus{{border:1.5px solid {_ACCENT};}}"
)
_LABEL_STYLE = "font-size:10pt;color:#DFF8FF;"
_GROUP_STYLE = (
    "QGroupBox{background:#091B29;border:1px solid #1C4961;border-radius:6px;"
    "margin-top:8px;padding:6px 8px;}"
    "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;"
    f"color:{_ACCENT};font-size:9pt;font-weight:bold;}}"
)


def _group(title):
    group = QGroupBox(title)
    group.setStyleSheet(_GROUP_STYLE)
    content = QVBoxLayout(group)
    content.setSpacing(4)
    content.setContentsMargins(6, 4, 6, 6)
    return group


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(7, 6), dpi=96)
        self.fig.patch.set_facecolor("#07131F")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


class MixerModelDialog(ModuleDialog):
    MODEL_KIND = "mixer"
    """仅考虑相位噪声的混频器模型。"""

    TITLE = "混频器模型（相位噪声）"
    ACCENT_COLOR = _ACCENT
    MIN_WIDTH = 1000
    MIN_HEIGHT = 650

    def __init__(self, *args, **kwargs):
        self._last_result = None
        super().__init__(*args, **kwargs)
        self.content_layout.removeWidget(self.model_actions)
        self.parameter_page.layout().insertWidget(0, self.model_actions)

    def build_content(self, layout: QVBoxLayout):
        self.tabs = QTabWidget()
        self.parameter_page = QWidget()
        parameter_layout = QVBoxLayout(self.parameter_page)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        self._build_parameter_model(parameter_layout)
        self.measured_page = MeasuredModelPage("mixer")
        self.tabs.addTab(self.parameter_page, "参数建模")
        self.tabs.addTab(self.measured_page, "实测数据建模与导出")
        layout.addWidget(self.tabs)

    def _build_parameter_model(self, layout: QVBoxLayout):
        layout.setContentsMargins(10, 8, 10, 10)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle{background:#17384A;}")

        left = QWidget()
        left.setMinimumWidth(305)
        left.setMaximumWidth(385)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(7)

        sim_group = _group("① 等效复中频仿真")
        sim_form = QFormLayout()
        sim_form.setSpacing(6)
        self.e_fs = self._edit("10.0")
        self.e_tsim = self._edit("1.0")
        self.e_f_if = self._edit("1.0")
        sim_form.addRow(self._label("采样率 (MHz):"), self.e_fs)
        sim_form.addRow(self._label("仿真时长 (ms):"), self.e_tsim)
        sim_form.addRow(self._label("输入中频 (MHz):"), self.e_f_if)
        sim_group.layout().addLayout(sim_form)
        sim_hint = QLabel("输入为复单音；本振只包含零均值相位噪声。\n输出 = 输入 × exp(-j·φ[n])。")
        sim_hint.setWordWrap(True)
        sim_hint.setStyleSheet("font-size:8.5pt;color:#648A9C;")
        sim_group.layout().addWidget(sim_hint)
        left_layout.addWidget(sim_group)

        pn_group = _group("② 本振相位噪声")
        pn_form = QFormLayout()
        pn_form.setSpacing(6)
        self.e_anchors = self._edit("1k:-60, 10k:-80, 100k:-110, 1M:-130", width=245)
        pn_form.addRow(self._label("L(f) 锚点:"), self.e_anchors)
        pn_group.layout().addLayout(pn_form)
        pn_hint = QLabel(
            "格式：频偏:电平，单位为 dBc/Hz。\n"
            "例：1k:-60, 10k:-80, 100k:-110, 1M:-130\n"
            "所有相噪由固定随机种子生成，重复运行结果一致。"
        )
        pn_hint.setWordWrap(True)
        pn_hint.setStyleSheet("font-size:8pt;color:#648A9C;")
        pn_group.layout().addWidget(pn_hint)
        left_layout.addWidget(pn_group)

        self.btn_run = QPushButton("计算 / 绘图")
        self.btn_run.setFixedHeight(34)
        self.btn_run.setStyleSheet(
            f"QPushButton{{background:{_ACCENT};color:#FFF;border:none;border-radius:5px;"
            "font-size:10pt;font-weight:bold;}QPushButton:hover{background:#168BA5;}"
        )
        self.btn_run.clicked.connect(self._run)
        left_layout.addWidget(self.btn_run)
        self.btn_save = QPushButton("保存图像")
        self.btn_save.setFixedHeight(30)
        self.btn_save.setStyleSheet(
            "QPushButton{background:#091B29;color:#C9E6F2;border:1px solid #CCC;border-radius:5px;"
            "font-size:9.5pt;}QPushButton:hover{background:#0D3042;}"
        )
        self.btn_save.clicked.connect(self._save)
        left_layout.addWidget(self.btn_save)
        left_layout.addStretch()
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(4)
        self.canvas = PlotCanvas()
        right_layout.addWidget(self.canvas, stretch=1)

        self.status = QLabel("就绪 — 设置仿真与相噪参数后点击「计算 / 绘图」")
        self.status.setStyleSheet("font-size:9pt;color:#648A9C;")
        right_layout.addWidget(self.status)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)
        self._run()

    @staticmethod
    def _value(text, default):
        try:
            return float(str(text).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _edit(default, width=110):
        edit = QLineEdit(default)
        edit.setFixedWidth(width)
        edit.setStyleSheet(_EDIT_STYLE)
        return edit

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setStyleSheet(_LABEL_STYLE)
        return label

    def _collect_cfg(self):
        return {
            "fs": self._value(self.e_fs.text(), 10.0) * 1e6,
            "sim_time": self._value(self.e_tsim.text(), 1.0) * 1e-3,
            "f_if_in": self._value(self.e_f_if.text(), 1.0) * 1e6,
            "anchors": parse_pn_anchors(self.e_anchors.text()),
        }

    def _run(self):
        try:
            result = run_mixer_phase_noise(**self._collect_cfg())
            self._last_result = result
            self._plot_spectrum(result)
        except Exception as error:
            QMessageBox.critical(self, "计算错误", str(error))

    def _plot_spectrum(self, result):
        fig = self.canvas.fig
        fig.clf()
        fig.set_constrained_layout(True)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#091B29")
        offset_khz = result["offset_from_carrier"] / 1e3
        ax.plot(offset_khz, result["output_dbc_hz"], color=_ACCENT, lw=0.85,
                label="含相位噪声")
        ax.plot(offset_khz, result["ideal_dbc_hz"], color="#3DDC97", lw=1.0, ls="--",
                label="理想载波谱（参考）")
        span_hz = min(result["fs"] / 4.0, 2.0 * result["f_if_in"])
        ax.set_xlim(-span_hz / 1e3, span_hz / 1e3)
        ax.set_ylim(-300, 5)
        ax.set_xlabel("相对载波频偏 (kHz)", fontsize=10)
        ax.set_ylabel("单边带相位噪声密度 (dBc/Hz)", fontsize=10)
        ax.set_title("混频输出频谱：本振相位噪声导致的载波裙边", fontsize=10)
        ax.grid(True, color="#17384A", lw=0.6)
        for spine in ax.spines.values():
            spine.set_color("#1C4961")
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95, edgecolor="#39728A")
        self.canvas.draw()
        self.status.setText(
            f"等效复中频 = {result['f_if_in'] / 1e6:.3f} MHz  |  "
            f"RBW = {result['enbw_hz']:.1f} Hz  |  "
            f"L(f) → 随机相位噪声 φ[n] → 输出频谱裙边  |  "
            f"不含热噪声、非线性、变频损耗、LO 泄漏和杂散"
        )

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "mixer_phase_noise.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self.canvas.save(path)
            QMessageBox.information(self, "保存成功", f"已保存：\n{path}")
