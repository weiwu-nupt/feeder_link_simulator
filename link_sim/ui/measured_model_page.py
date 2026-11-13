"""Separate measured-data workflows for the filter and mixer editors."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget,
)
from scipy import signal

from link_sim.algorithm.measured_data import load_measured_filter, load_measured_mixer
from link_sim.ui.model_library import choose_export, save_named


class MeasuredModelPage(QWidget):
    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.model = None
        is_filter = kind == "filter"
        self.file_filter = "实测数据 (*.csv *.txt *.mat *.npz)" if is_filter else "实测数据 (*.csv *.txt *.mat)"

        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        controls = QWidget()
        controls.setMinimumWidth(285)
        left = QVBoxLayout(controls)
        left.setContentsMargins(0, 0, 8, 0)

        data_group = QGroupBox("实测数据")
        data_layout = QVBoxLayout(data_group)
        file_row = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setReadOnly(True)
        self.path.setPlaceholderText("请选择实测数据文件")
        self.path.textChanged.connect(self._invalidate_model)
        file_row.addWidget(self.path, 1)
        self.browse_button = QPushButton("浏览…")
        self.browse_button.clicked.connect(self._browse)
        file_row.addWidget(self.browse_button)
        data_layout.addLayout(file_row)
        formats = QLabel(
            "CSV/TXT：单列冲激响应 h，或三列频率 (Hz)、增益 (dB)、相位 (°)。"
            "\nMAT/NPZ：h、b/a 或 sos；若含 fs (Hz)，优先使用文件采样率。"
            if is_filter else
            "CSV/TXT：两列频偏 (Hz)、相噪 (dBc/Hz)。\nMAT：offsets 和 levels 数组。"
        )
        formats.setWordWrap(True)
        data_layout.addWidget(formats)
        left.addWidget(data_group)

        model_group = QGroupBox("建模参数")
        form = QFormLayout(model_group)
        self.fs = self._field(form, "采样率 (MHz)", "100" if is_filter else "10")
        if not is_filter:
            self.duration = self._field(form, "仿真时长 (ms)", "1")
            self.if_frequency = self._field(form, "输入中频 (MHz)", "1")
        left.addWidget(model_group)

        self.build_button = QPushButton("建立模型并预览")
        self.build_button.setProperty("class", "primary")
        self.build_button.setEnabled(False)
        self.build_button.clicked.connect(self._build_model)
        left.addWidget(self.build_button)
        self.save_button = QPushButton("保存到模型库")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_model)
        left.addWidget(self.save_button)
        self.export_button = QPushButton("导出模型文件…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_model)
        left.addWidget(self.export_button)
        left.addStretch()
        splitter.addWidget(controls)

        preview = QWidget()
        right = QVBoxLayout(preview)
        right.setContentsMargins(6, 0, 0, 0)
        self.canvas = FigureCanvasQTAgg(Figure(facecolor="#07131f", layout="constrained"))
        right.addWidget(self.canvas, 1)
        self.status = QLabel("选择实测数据文件，设置参数后建立模型。")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        splitter.addWidget(preview)
        splitter.setSizes([320, 660])
        splitter.setStretchFactor(1, 1)

    def _field(self, form, label, value):
        field = QLineEdit(value)
        field.textChanged.connect(self._invalidate_model)
        form.addRow(label, field)
        return field

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择实测数据", self.path.text(), self.file_filter)
        if path:
            self.path.setText(path)
            self.path.setToolTip(path)
            self._invalidate_model()

    def _invalidate_model(self):
        if self.model is not None:
            self.canvas.figure.clear()
            self.canvas.draw_idle()
        self.model = None
        self.save_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.build_button.setEnabled(bool(self.path.text()))
        self.status.setText("点击「建立模型并预览」生成可保存的模型。")

    def _build_model(self):
        self._invalidate_model()
        try:
            fs = float(self.fs.text()) * 1e6
            path = self.path.text()
            if self.kind == "filter":
                model = load_measured_filter(path, fs)
            else:
                settings = dict(fs=fs, sim_time=float(self.duration.text()) * 1e-3,
                                f_if_in=float(self.if_frequency.text()) * 1e6)
                model = load_measured_mixer(path, settings)
            self._plot_model(model)
        except Exception as exc:
            QMessageBox.warning(self, "实测数据建模失败", str(exc))
            return
        self.model = model
        self.save_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.status.setText(f"已建立模型 · {model.name} · 采样率 {model.parameters['fs']/1e6:g} MHz")

    def _plot_model(self, model):
        figure = self.canvas.figure
        figure.clear()
        if model.kind == "filter":
            arrays = model.arrays
            fs = model.parameters["fs"]
            if "sos" in arrays:
                sos = arrays["sos"]
                frequency, response = signal.freqz_sos(sos / sos[:, 3:4], worN=4096, fs=fs)
            else:
                b = arrays["b"] if "b" in arrays else arrays["h"]
                frequency, response = signal.freqz(b, arrays.get("a", 1), worN=4096, fs=fs)
            gain, phase = figure.subplots(2, 1, sharex=True)
            gain.plot(frequency/1e6, 20*np.log10(np.maximum(abs(response), 1e-15)), color="#22c7e6")
            gain.set_ylabel("增益 (dB)")
            phase.plot(frequency/1e6, np.rad2deg(np.unwrap(np.angle(response))), color="#22c7e6")
            phase.set(xlabel="频率 (MHz)", ylabel="相位 (°)")
        else:
            offsets, levels = np.asarray(model.parameters["anchors"]).T
            grid = np.geomspace(offsets[0], offsets[-1], max(512, len(offsets)))
            ax = figure.add_subplot(111)
            ax.semilogx(grid, np.interp(np.log10(grid), np.log10(offsets), levels),
                        color="#22c7e6", label="相噪模型")
            ax.scatter(offsets, levels, color="#f2c14e", s=16, label="实测锚点", zorder=3)
            ax.set(xlabel="频偏 (Hz)", ylabel="单边带相噪 (dBc/Hz)")
            ax.legend(facecolor="#091b29", edgecolor="#39728a", labelcolor="#d9ecf7")
        for ax in figure.axes:
            ax.set_facecolor("#091b29")
            ax.tick_params(colors="#91afc2")
            ax.xaxis.label.set_color("#d9ecf7")
            ax.yaxis.label.set_color("#d9ecf7")
            for spine in ax.spines.values():
                spine.set_color("#39728a")
            ax.grid(True, color="#17384a", alpha=.7)
        self.canvas.draw_idle()

    def _save_model(self):
        if self.model is None:
            return
        try:
            model = self.model.validated()
            key = save_named(model, self)
            if key is not None:
                self.status.setText(f"已入库：#{key} {model.name}")
        except Exception as exc:
            QMessageBox.warning(self, "保存模型失败", str(exc))

    def _export_model(self):
        if self.model is None:
            return
        try:
            path = choose_export(self, self.model)
            if path:
                self.status.setText(f"已导出：{Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self, "导出模型失败", str(exc))
