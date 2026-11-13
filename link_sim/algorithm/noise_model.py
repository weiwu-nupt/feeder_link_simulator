"""Independent complex white noise floor, referenced to unit full-scale power."""

import numpy as np
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox, QFormLayout, QLabel, QPushButton
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from link_sim.ui.base_dialog import ModuleDialog


from .noise_core import generate_noise


class NoiseModelDialog(ModuleDialog):
    MODEL_KIND = "noise"
    MODEL_FILE_ACTIONS = False
    TITLE = "底噪模型"

    def build_content(self, layout):
        form = QFormLayout()
        self.fs = QDoubleSpinBox()
        self.fs.setDecimals(6)
        self.fs.setRange(.000001, 1e6)
        self.fs.setValue(100)
        self.density = QDoubleSpinBox()
        self.density.setRange(-300, 0)
        self.density.setValue(-150)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(2025)
        form.addRow("预览采样率 (MHz)", self.fs)
        form.addRow("底噪密度 (dBFS/Hz)", self.density)
        form.addRow("随机种子", self.seed)
        layout.addLayout(form)
        hint = QLabel("复高斯白噪声；满量程功率为 1。总噪声功率 = 10^(密度/10) × 采样率。\n加载到全链路时按输入采样率生成底噪，输出 = 输入 + 底噪。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        button = QPushButton("计算 / 绘图")
        button.clicked.connect(self._run)
        layout.addWidget(button)
        self.canvas = FigureCanvasQTAgg(Figure(facecolor="#07131f"))
        layout.addWidget(self.canvas, 1)
        self.status = QLabel("设置底噪参数后点击计算，可保存到模型库供全链路使用。")
        layout.addWidget(self.status)

    def _collect_cfg(self):
        return dict(fs=self.fs.value()*1e6, density_dbfs_hz=self.density.value(), seed=self.seed.value())

    def _run(self):
        from scipy.signal import welch
        cfg = self._collect_cfg()
        x = generate_noise(65536, **cfg)
        f, psd = welch(x, cfg["fs"], nperseg=2048, return_onesided=False)
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.plot(np.fft.fftshift(f)/1e6, 10*np.log10(np.fft.fftshift(psd)+1e-30))
        ax.axhline(cfg["density_dbfs_hz"], color="orange", linestyle="--")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("PSD (dBFS/Hz)")
        self.canvas.draw_idle()
        self.status.setText(f"总噪声功率：{10*np.log10(np.mean(abs(x)**2)):.2f} dBFS")
