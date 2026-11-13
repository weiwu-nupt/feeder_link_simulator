"""
滤波器建模模块 — Filter Design & Analysis
==========================================
对馈电链路收发链路中的滤波环节做设计与频响分析：

  抗混叠 / 重建滤波   ADC 前 / DAC 后的低通，抑制混叠与镜像
  脉冲成形滤波        根升余弦(RRC)，控制带宽与码间串扰(ISI)
  信道选择 / 带通      中频带通，选出目标信道
  匹配滤波            收端 RRC，与发端级联成升余弦(RC)零 ISI

支持两类实现
  IIR   Butterworth / Chebyshev I / Chebyshev II / Elliptic / Bessel
        低阶高效，相位非线性（Bessel 群时延最平坦）
  FIR   窗函数法(firwin) / 等波纹(remez)
        线性相位、群时延恒定，阶数较高

核心指标
  通带波纹 / 阻带衰减 / 过渡带宽 / 群时延平坦度 / 3 dB 截止

参考
  - ITU-R 频谱模板与邻道泄漏要求
  - DVB-S2X RRC 滚降系数 α = 0.05 / 0.10 / 0.15 / 0.20 / 0.25 / 0.35
  - scipy.signal 滤波器设计例程
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import scipy.signal as signal

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QSizePolicy,
    QGridLayout, QScrollArea, QTabWidget,
)
from PySide6.QtCore import Qt

from link_sim.ui.base_dialog import ModuleDialog
from link_sim.ui.measured_model_page import MeasuredModelPage


# ── 字体 ──────────────────────────────────────────────────
def _setup_font():
    for n in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"]:
        if n in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = n
            plt.rcParams["axes.unicode_minus"] = False
            return
_setup_font()


# ══════════════════════════════════════════════════════════
#  核心算法 — 滤波器设计
# ══════════════════════════════════════════════════════════

_BAND_MAP = {
    "低通": "lowpass",
    "高通": "highpass",
    "带通": "bandpass",
    "带阻": "bandstop",
}

_IIR_TYPES = ["Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic", "Bessel"]


def design_iir(cfg):
    """
    设计 IIR 滤波器，返回 (b, a, sos)。

    cfg 关键字段
    ----
    iir_type : 上述 _IIR_TYPES 之一
    band     : '低通'/'高通'/'带通'/'带阻'
    order    : 阶数 N
    fs       : 采样率 (Hz)
    fc1, fc2 : 截止频率 (Hz)；带通/带阻用两个，其余只用 fc1
    rp       : 通带波纹 (dB)，Cheby I / Elliptic 用
    rs       : 阻带衰减 (dB)，Cheby II / Elliptic 用
    """
    band = _BAND_MAP[cfg["band"]]
    fs = cfg["fs"]
    N = cfg["order"]
    rp = cfg["rp"]
    rs = cfg["rs"]

    if band in ("bandpass", "bandstop"):
        wn = [cfg["fc1"], cfg["fc2"]]
    else:
        wn = cfg["fc1"]

    t = cfg["iir_type"]
    common = dict(btype=band, fs=fs, output="sos")
    if t == "Butterworth":
        sos = signal.butter(N, wn, **common)
    elif t == "Chebyshev I":
        sos = signal.cheby1(N, rp, wn, **common)
    elif t == "Chebyshev II":
        sos = signal.cheby2(N, rs, wn, **common)
    elif t == "Elliptic":
        sos = signal.ellip(N, rp, rs, wn, **common)
    elif t == "Bessel":
        sos = signal.bessel(N, wn, btype=band, fs=fs, output="sos", norm="mag")
    else:
        raise ValueError(f"未知 IIR 类型: {t}")

    b, a = signal.sos2tf(sos)
    return b, a, sos


def design_fir(cfg):
    """
    设计 FIR 滤波器，返回 (taps, taps_a=1)。

    cfg 关键字段
    ----
    fir_method : '窗函数法' / '等波纹(remez)'
    band       : '低通'/'高通'/'带通'/'带阻'
    numtaps    : 抽头数（线性相位，奇数对高通/带阻更稳）
    fs, fc1, fc2
    window     : 窗函数名（窗函数法用）
    trans_bw   : 过渡带宽 (Hz)，remez 用
    """
    band = _BAND_MAP[cfg["band"]]
    fs = cfg["fs"]
    nyq = fs / 2.0
    numtaps = cfg["numtaps"]
    # 高通 / 带阻需要奇数抽头（type I），保证 nyquist 处增益非零
    if band in ("highpass", "bandstop") and numtaps % 2 == 0:
        numtaps += 1

    if cfg["fir_method"].startswith("窗"):
        if band == "lowpass":
            taps = signal.firwin(numtaps, cfg["fc1"], window=cfg["window"], fs=fs)
        elif band == "highpass":
            taps = signal.firwin(numtaps, cfg["fc1"], window=cfg["window"],
                                 pass_zero=False, fs=fs)
        elif band == "bandpass":
            taps = signal.firwin(numtaps, [cfg["fc1"], cfg["fc2"]],
                                 window=cfg["window"], pass_zero=False, fs=fs)
        else:  # bandstop
            taps = signal.firwin(numtaps, [cfg["fc1"], cfg["fc2"]],
                                 window=cfg["window"], pass_zero=True, fs=fs)
    else:
        # 等波纹 remez
        tb = cfg["trans_bw"]
        if band == "lowpass":
            f = cfg["fc1"]
            bands = [0, f - tb / 2, f + tb / 2, nyq]
            desired = [1, 0]
        elif band == "highpass":
            f = cfg["fc1"]
            bands = [0, f - tb / 2, f + tb / 2, nyq]
            desired = [0, 1]
        elif band == "bandpass":
            bands = [0, cfg["fc1"] - tb / 2, cfg["fc1"] + tb / 2,
                     cfg["fc2"] - tb / 2, cfg["fc2"] + tb / 2, nyq]
            desired = [0, 1, 0]
        else:  # bandstop
            bands = [0, cfg["fc1"] - tb / 2, cfg["fc1"] + tb / 2,
                     cfg["fc2"] - tb / 2, cfg["fc2"] + tb / 2, nyq]
            desired = [1, 0, 1]
        bands = [max(0.0, min(b, nyq)) for b in bands]
        taps = signal.remez(numtaps, bands, desired, fs=fs)

    return taps, np.array([1.0])


def design_rrc(beta, sps, span_syms):
    """
    根升余弦(RRC)脉冲成形滤波器抽头。

    参数
    ----
    beta      : 滚降系数 α (0 < α ≤ 1)
    sps       : 每符号采样数 (samples per symbol)
    span_syms : 滤波器跨度（符号数），总抽头 = span*sps + 1

    返回归一化（能量为 1）的抽头数组。RRC 与自身级联得到升余弦(RC)，
    满足 Nyquist 第一准则（采样点零 ISI）。
    """
    N = span_syms * sps
    t = (np.arange(N + 1) - N / 2.0) / sps   # 以符号周期 T 为单位
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * ti * (1 - beta)) +
                   4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


# ══════════════════════════════════════════════════════════
#  频响 / 指标提取
# ══════════════════════════════════════════════════════════

def freq_response(b, a, fs, n=4096):
    """返回 (freq_Hz, H_complex, mag_dB, phase_deg)。"""
    w, H = signal.freqz(b, a, worN=n, fs=fs)
    mag = 20 * np.log10(np.abs(H) + 1e-12)
    phase = np.unwrap(np.angle(H))
    return w, H, mag, np.degrees(phase)


def group_delay(b, a, fs, n=4096):
    """返回 (freq_Hz, gd_seconds)。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, gd = signal.group_delay((b, a), w=n, fs=fs)
    return w, gd / fs   # gd 以采样点计 → 转换为秒


def filter_metrics(b, a, fs, cfg):
    """
    提取关键指标 dict：
      f3dB    : -3 dB 截止频率 (Hz)
      ripple  : 通带波纹峰峰 (dB)
      atten   : 阻带最小衰减 (dB)
      gd_var  : 通带群时延起伏 (ns)
    """
    w, H, mag, _ = freq_response(b, a, fs, n=8192)
    band = _BAND_MAP[cfg["band"]]
    nyq = fs / 2.0
    fc1 = cfg.get("fc1", nyq / 2)
    fc2 = cfg.get("fc2", nyq / 2)

    # 通带 / 阻带掩膜
    if band == "lowpass":
        pb = w <= fc1
        sb = w >= min(fc1 * 1.5, nyq)
    elif band == "highpass":
        pb = w >= fc1
        sb = w <= fc1 / 1.5
    elif band == "bandpass":
        pb = (w >= fc1) & (w <= fc2)
        sb = (w <= fc1 / 1.5) | (w >= min(fc2 * 1.5, nyq))
    else:  # bandstop
        pb = (w <= fc1) | (w >= fc2)
        # 阻带取陷波内区（避开过渡带边缘），以反映真实陷波深度
        margin = 0.15 * (fc2 - fc1)
        sb = (w >= fc1 + margin) & (w <= fc2 - margin)

    pb_mag = mag[pb]
    sb_mag = mag[sb]
    ripple = (np.max(pb_mag) - np.min(pb_mag)) if pb_mag.size else float("nan")
    atten = (-np.max(sb_mag)) if sb_mag.size else float("nan")

    # -3 dB 点（相对通带峰值）
    ref = np.max(pb_mag) if pb_mag.size else 0.0
    cross = np.where(np.diff(np.sign(mag - (ref - 3.0))))[0]
    f3 = float(w[cross[0]]) if cross.size else float("nan")

    # 通带群时延起伏
    wg, gd = group_delay(b, a, fs, n=8192)
    if band == "lowpass":
        m = wg <= fc1
    elif band == "highpass":
        m = wg >= fc1
    elif band == "bandpass":
        m = (wg >= fc1) & (wg <= fc2)
    else:
        m = (wg <= fc1) | (wg >= fc2)
    gd_band = gd[m]
    gd_var = (np.max(gd_band) - np.min(gd_band)) * 1e9 if gd_band.size else float("nan")

    return dict(f3dB=f3, ripple=ripple, atten=atten, gd_var=gd_var)


# ══════════════════════════════════════════════════════════
#  UI 样式（与 ad_model / ber_analysis 对齐）
# ══════════════════════════════════════════════════════════

_ACCENT = "#22C7E6"
_ES = ("QLineEdit{background:#091B29;border:1px solid #1C4961;"
       "border-radius:3px;padding:3px 6px;font-size:10pt;color:#D9ECF7;}"
       f"QLineEdit:focus{{border:1.5px solid {_ACCENT};}}")
_LS = "font-size:10pt;color:#DFF8FF;"
_GB = ("QGroupBox{background:#091B29;border:1px solid #1C4961;"
       "border-radius:6px;margin-top:8px;padding:6px 8px;}"
       "QGroupBox::title{subcontrol-origin:margin;left:10px;"
       f"padding:0 4px;color:{_ACCENT};font-size:9pt;font-weight:bold;}}")


def _group(title):
    gb = QGroupBox(title)
    gb.setStyleSheet(_GB)
    vl = QVBoxLayout(gb)
    vl.setSpacing(4)
    vl.setContentsMargins(6, 4, 6, 6)
    return gb


def _cb_style():
    return ("QComboBox{background:#091B29;border:1px solid #1C4961;"
            "border-radius:3px;padding:3px 8px;font-size:10pt;color:#D9ECF7;}"
            "QComboBox QAbstractItemView{background:#091B29;color:#D9ECF7;"
            "selection-background-color:#E3EDF6;font-size:10pt;}")


# ══════════════════════════════════════════════════════════
#  画布
# ══════════════════════════════════════════════════════════

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(7, 6.2), dpi=96)
        self.fig.patch.set_facecolor("#07131F")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


# 统一坐标轴美化
def _style_ax(ax):
    ax.set_facecolor("#091B29")
    ax.grid(True, which="major", color="#17384A", lw=0.6)
    ax.grid(True, which="minor", color="#102C3B", lw=0.4)
    for s in ax.spines.values():
        s.set_color("#1C4961")
    ax.tick_params(labelsize=9)


# ══════════════════════════════════════════════════════════
#  对话框
# ══════════════════════════════════════════════════════════

class FilterModelDialog(ModuleDialog):
    MODEL_KIND = "filter"
    TITLE        = "滤波器建模"
    ACCENT_COLOR = "#22C7E6"
    MIN_WIDTH    = 1060
    MIN_HEIGHT   = 720

    def __init__(self, *a, **kw):
        self._last = None
        self._library_filter = None
        super().__init__(*a, **kw)
        self.content_layout.removeWidget(self.model_actions)
        self.parameter_page.layout().insertWidget(0, self.model_actions)

    # ──────────────────────────────────────────────────────
    def build_content(self, layout: QVBoxLayout):
        self.tabs = QTabWidget()
        self.parameter_page = QWidget()
        parameter_layout = QVBoxLayout(self.parameter_page)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        self._build_parameter_model(parameter_layout)
        self.measured_page = MeasuredModelPage("filter")
        self.tabs.addTab(self.parameter_page, "参数建模")
        self.tabs.addTab(self.measured_page, "实测数据建模与导出")
        layout.addWidget(self.tabs)

    def _build_parameter_model(self, layout: QVBoxLayout):
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(0)

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setHandleWidth(1)
        sp.setStyleSheet("QSplitter::handle{background:#17384A;}")

        # ══ 左侧配置（包在滚动区内，适应不同显示设备）══════
        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(360)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)
        lv.setSpacing(7)

        # ── ① 实现方式 ────────────────────────────────────
        ig = _group("① 实现方式")
        igf = QFormLayout(); igf.setSpacing(5); igf.setContentsMargins(0, 0, 0, 0)
        self._igf = igf
        self.cb_impl = QComboBox(); self.cb_impl.setStyleSheet(_cb_style())
        self.cb_impl.addItems(["IIR", "FIR", "RRC 成形"])
        self.cb_impl.currentIndexChanged.connect(self._on_impl_changed)
        igf.addRow(self._lbl("类别:"), self.cb_impl)

        self.cb_iir = QComboBox(); self.cb_iir.setStyleSheet(_cb_style())
        self.cb_iir.addItems(_IIR_TYPES)
        self._row_iir = self._add_row(igf, "IIR 原型:", self.cb_iir)

        self.cb_firm = QComboBox(); self.cb_firm.setStyleSheet(_cb_style())
        self.cb_firm.addItems(["窗函数法", "等波纹(remez)"])
        self.cb_firm.currentIndexChanged.connect(self._on_firm_changed)
        self._row_firm = self._add_row(igf, "FIR 方法:", self.cb_firm)

        self.cb_win = QComboBox(); self.cb_win.setStyleSheet(_cb_style())
        self.cb_win.addItems(["hamming", "hann", "blackman", "kaiser", "boxcar"])
        self._row_win = self._add_row(igf, "FIR 窗:", self.cb_win)
        ig.layout().addLayout(igf)
        lv.addWidget(ig)

        # ── ② 频率/带型 ───────────────────────────────────
        fg = _group("② 频率与带型")
        ff = QFormLayout(); ff.setSpacing(5); ff.setContentsMargins(0, 0, 0, 0)
        self.cb_band = QComboBox(); self.cb_band.setStyleSheet(_cb_style())
        self.cb_band.addItems(["低通", "高通", "带通", "带阻"])
        self.cb_band.currentIndexChanged.connect(self._on_band_changed)
        self.e_fs = self._edit("100.0")
        self.e_fc1 = self._edit("20.0")
        self.e_fc2 = self._edit("40.0")
        self._row_band = self._add_row(ff, "带型:", self.cb_band)
        ff.addRow(self._lbl("采样率 fs (MHz):"), self.e_fs)
        self.lbl_fc1 = self._lbl("截止 fc (MHz):")
        self.lbl_fc2 = self._lbl("上截止 fc2 (MHz):")
        ff.addRow(self.lbl_fc1, self.e_fc1)
        ff.addRow(self.lbl_fc2, self.e_fc2)
        fg.layout().addLayout(ff)
        self._freq_group = fg
        lv.addWidget(fg)

        # ── ③ 阶数/容差 ───────────────────────────────────
        og = _group("③ 阶数与容差")
        of = QFormLayout(); of.setSpacing(5); of.setContentsMargins(0, 0, 0, 0)
        self.e_order = self._edit("6")
        self.e_taps = self._edit("65")
        self.e_rp = self._edit("0.5")
        self.e_rs = self._edit("60")
        self.e_tb = self._edit("4.0")
        self._row_order = self._add_row(of, "IIR 阶数 N:", self.e_order)
        self._row_taps = self._add_row(of, "FIR 抽头数:", self.e_taps)
        self._row_rp = self._add_row(of, "通带波纹 (dB):", self.e_rp)
        self._row_rs = self._add_row(of, "阻带衰减 (dB):", self.e_rs)
        self._row_tb = self._add_row(of, "过渡带 (MHz):", self.e_tb)
        og.layout().addLayout(of)
        self._order_group = og
        lv.addWidget(og)

        # ── ④ RRC 成形 ────────────────────────────────────
        rg = _group("④ RRC 脉冲成形")
        rf = QFormLayout(); rf.setSpacing(5); rf.setContentsMargins(0, 0, 0, 0)
        self.e_beta = self._edit("0.25")
        self.e_sps = self._edit("8")
        self.e_span = self._edit("10")
        rf.addRow(self._lbl("滚降系数 α:"), self.e_beta)
        rf.addRow(self._lbl("每符号采样 sps:"), self.e_sps)
        rf.addRow(self._lbl("跨度 (符号):"), self.e_span)
        rg.layout().addLayout(rf)
        hint = QLabel("DVB-S2X 常用 α = 0.05~0.35；\n"
                      "收发各一级 RRC 级联得到零 ISI 升余弦")
        hint.setStyleSheet("font-size:8pt;color:#648A9C;")
        hint.setWordWrap(True)
        rg.layout().addWidget(hint)
        lv.addWidget(rg)
        self._rrc_group = rg

        # ── ⑤ 视图 ────────────────────────────────────────
        vg = _group("⑤ 视图")
        vf = QFormLayout(); vf.setSpacing(5); vf.setContentsMargins(0, 0, 0, 0)
        self.cb_view = QComboBox(); self.cb_view.setStyleSheet(_cb_style())
        vf.addRow(self._lbl("显示:"), self.cb_view)
        vg.layout().addLayout(vf)
        lv.addWidget(vg)

        self.btn_run = QPushButton("设计并绘制")
        self.btn_run.setFixedHeight(34)
        self.btn_run.setStyleSheet(
            f"QPushButton{{background:{_ACCENT};color:#FFF;border:none;"
            "border-radius:5px;font-size:10pt;font-weight:bold;}"
            "QPushButton:hover{background:#168BA5;}")
        self.btn_run.clicked.connect(self._run)
        lv.addWidget(self.btn_run)

        self.btn_save = QPushButton("保存图像")
        self.btn_save.setFixedHeight(30)
        self.btn_save.setStyleSheet(
            "QPushButton{background:#091B29;color:#C9E6F2;border:1px solid #CCC;"
            "border-radius:5px;font-size:9.5pt;}"
            "QPushButton:hover{background:#0D3042;}")
        self.btn_save.clicked.connect(self._save)
        lv.addWidget(self.btn_save)

        lv.addStretch()

        # 滚动区：内容超出可视高度时显示滚轮，适应小屏 / 高 DPI 设备
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{background:transparent;}")
        scroll.setWidget(left)
        scroll.setMinimumWidth(318)
        scroll.setMaximumWidth(380)
        sp.addWidget(scroll)

        # ══ 右侧：画布 + 指标栏 ═══════════════════════════
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)
        rv.setSpacing(4)

        self.canvas = PlotCanvas()
        rv.addWidget(self.canvas, stretch=1)

        self.metric_box = QWidget()
        self.metric_box.setStyleSheet(
            "background:#091B29;border:1px solid #1C4961;border-radius:6px;")
        mg = QGridLayout(self.metric_box)
        mg.setContentsMargins(10, 6, 10, 6)
        mg.setSpacing(4)
        self._metric_labels = {}
        specs = [("f3dB", "MHz"), ("通带波纹", "dB"),
                 ("阻带衰减", "dB"), ("群时延起伏", "ns")]
        for i, (name, unit) in enumerate(specs):
            cap = QLabel(name)
            cap.setStyleSheet("font-size:8.5pt;color:#648A9C;")
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = QLabel("—")
            val.setStyleSheet(f"font-size:13pt;font-weight:bold;color:{_ACCENT};")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            uni = QLabel(unit)
            uni.setStyleSheet("font-size:8pt;color:#91AFC2;")
            uni.setAlignment(Qt.AlignmentFlag.AlignCenter)
            mg.addWidget(cap, 0, i)
            mg.addWidget(val, 1, i)
            mg.addWidget(uni, 2, i)
            self._metric_labels[name] = val
        rv.addWidget(self.metric_box)

        self.status = QLabel("就绪 — 设置参数后点击「设计并绘制」")
        self.status.setStyleSheet("font-size:9pt;color:#648A9C;")
        rv.addWidget(self.status)

        sp.addWidget(right)
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        layout.addWidget(sp, stretch=1)

        self._on_impl_changed(0)
        self._on_band_changed(0)
        self._run()

    # ── 辅助 ──────────────────────────────────────────────
    def _lbl(self, t):
        l = QLabel(t); l.setStyleSheet(_LS); return l

    def _edit(self, default, w=110):
        e = QLineEdit(default); e.setFixedWidth(w); e.setStyleSheet(_ES)
        return e

    def _add_row(self, form: QFormLayout, label_text, field):
        """添加一行并返回 (label, field)，便于整行显示/隐藏。"""
        lbl = self._lbl(label_text)
        form.addRow(lbl, field)
        return (lbl, field)

    @staticmethod
    def _set_row_visible(row, visible):
        """整行（标签 + 控件）一并显示或隐藏。"""
        lbl, field = row
        lbl.setVisible(visible)
        field.setVisible(visible)

    @staticmethod
    def _fv(text, default=0.0):
        try:
            return float(str(text).strip())
        except (ValueError, AttributeError):
            return default

    def _set_views(self, items):
        cur = self.cb_view.currentText()
        self.cb_view.blockSignals(True)
        self.cb_view.clear()
        self.cb_view.addItems(items)
        if cur in items:
            self.cb_view.setCurrentText(cur)
        self.cb_view.blockSignals(False)

    # ── 联动 ──────────────────────────────────────────────
    def _on_impl_changed(self, idx):
        impl = self.cb_impl.currentText()
        is_iir = impl == "IIR"
        is_fir = impl == "FIR"
        is_rrc = impl.startswith("RRC")

        # ① 实现方式：仅显示当前类别相关的行
        self._set_row_visible(self._row_iir, is_iir)
        self._set_row_visible(self._row_firm, is_fir)
        # FIR 窗只在「窗函数法」时出现
        win_visible = is_fir and self.cb_firm.currentText().startswith("窗")
        self._set_row_visible(self._row_win, win_visible)

        # ②③ 频率与带型、阶数容差：IIR/FIR 显示，RRC 隐藏整组
        self._freq_group.setVisible(not is_rrc)
        self._order_group.setVisible(not is_rrc)
        # 阶数容差里再按 IIR/FIR 精细显示
        self._set_row_visible(self._row_order, is_iir)
        self._set_row_visible(self._row_taps, is_fir)
        # 波纹仅 IIR 用；过渡带仅 FIR 等波纹用
        self._set_row_visible(self._row_rp, is_iir)
        self._set_row_visible(self._row_rs, is_iir)
        self._set_row_visible(
            self._row_tb, is_fir and self.cb_firm.currentText().startswith("等波纹"))

        # ④ RRC 组：仅 RRC 显示
        self._rrc_group.setVisible(is_rrc)

        if is_rrc:
            self._set_views(["脉冲响应 (RRC vs RC)", "幅频响应", "眼图 (RC 级联)"])
            self.metric_box.setVisible(False)
        else:
            self._set_views(["幅频响应", "相位响应", "群时延",
                             "冲激/阶跃响应", "零极点图"])
            self.metric_box.setVisible(True)

    def _on_firm_changed(self, idx):
        # 切换 FIR 方法时，重新评估窗 / 过渡带行的可见性
        if self.cb_impl.currentText() == "FIR":
            is_win = self.cb_firm.currentText().startswith("窗")
            self._set_row_visible(self._row_win, is_win)
            self._set_row_visible(self._row_tb, not is_win)

    def _on_band_changed(self, idx):
        two = self.cb_band.currentText() in ("带通", "带阻")
        self.lbl_fc2.setVisible(two)
        self.e_fc2.setVisible(two)
        if self.cb_band.currentText() in ("带通", "带阻"):
            self.lbl_fc1.setText("下截止 fc1 (MHz):")
        else:
            self.lbl_fc1.setText("截止 fc (MHz):")

    # ── 收集参数 ──────────────────────────────────────────
    def _collect_cfg(self):
        return dict(
            impl=self.cb_impl.currentText(),
            iir_type=self.cb_iir.currentText(),
            fir_method=self.cb_firm.currentText(),
            window=self.cb_win.currentText(),
            band=self.cb_band.currentText(),
            fs=self._fv(self.e_fs.text(), 100.0) * 1e6,
            fc1=self._fv(self.e_fc1.text(), 20.0) * 1e6,
            fc2=self._fv(self.e_fc2.text(), 40.0) * 1e6,
            order=max(1, int(self._fv(self.e_order.text(), 6))),
            numtaps=max(3, int(self._fv(self.e_taps.text(), 65))),
            rp=self._fv(self.e_rp.text(), 0.5),
            rs=self._fv(self.e_rs.text(), 60.0),
            trans_bw=self._fv(self.e_tb.text(), 4.0) * 1e6,
            beta=min(1.0, max(0.0, self._fv(self.e_beta.text(), 0.25))),
            sps=max(2, int(self._fv(self.e_sps.text(), 8))),
            span=max(2, int(self._fv(self.e_span.text(), 10))),
        )

    # ── 运行 ──────────────────────────────────────────────
    def _run(self):
        try:
            self._do_run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "设计错误", str(e))

    def _do_run(self):
        cfg = self._collect_cfg()
        impl = cfg["impl"]
        view = self.cb_view.currentText()
        frozen = self._library_filter
        arrays = frozen[1] if frozen is not None and cfg == frozen[0] else None

        if impl.startswith("RRC"):
            h = arrays["h"].copy() if arrays is not None else design_rrc(cfg["beta"], cfg["sps"], cfg["span"])
            self._last = ("rrc", cfg, h)
            if view.startswith("脉冲"):
                self._plot_rrc_impulse(cfg, h)
            elif view.startswith("眼图"):
                self._plot_rrc_eye(cfg, h)
            else:
                self._plot_rrc_freq(cfg, h)
            return

        if arrays is not None:
            if "b" in arrays:
                b, a, sos = arrays["b"].copy(), arrays["a"].copy(), arrays.get("sos")
            else:
                b, a, sos = arrays["h"].copy(), np.array([1.]), None
        elif impl == "外部系数":
            raise ValueError("外部系数模型的参数已修改，请切换 IIR/FIR 重新设计，或重新导入系数")
        elif impl == "IIR":
            b, a, sos = design_iir(cfg)
        else:
            b, a = design_fir(cfg)
            sos = None
        self._last = ("filt", cfg, (b, a, sos))

        m = filter_metrics(b, a, cfg["fs"], cfg)
        self._update_metrics(m, cfg)

        if view.startswith("幅频"):
            self._plot_mag(cfg, b, a, m)
        elif view.startswith("相位"):
            self._plot_phase(cfg, b, a)
        elif view.startswith("群时延"):
            self._plot_gd(cfg, b, a)
        elif view.startswith("冲激"):
            self._plot_impulse(cfg, b, a)
        else:
            self._plot_pz(cfg, b, a)

    # ── 指标栏 ────────────────────────────────────────────
    def _update_metrics(self, m, cfg):
        self._metric_labels["f3dB"].setText(
            "—" if not np.isfinite(m["f3dB"]) else f"{m['f3dB']/1e6:.2f}")
        self._metric_labels["通带波纹"].setText(
            "—" if not np.isfinite(m["ripple"]) else f"{m['ripple']:.3f}")
        self._metric_labels["阻带衰减"].setText(
            "—" if not np.isfinite(m["atten"]) else f"{m['atten']:.1f}")
        self._metric_labels["群时延起伏"].setText(
            "—" if not np.isfinite(m["gd_var"]) else f"{m['gd_var']:.2f}")

    # ════════ 绘图：幅频响应 ════════
    def _plot_mag(self, cfg, b, a, m):
        fig = self.canvas.fig; fig.clf()
        fig.set_constrained_layout(True); fig.patch.set_facecolor("#07131F")
        ax = fig.add_subplot(111); _style_ax(ax)

        w, H, mag, _ = freq_response(b, a, cfg["fs"], n=4096)
        ax.plot(w / 1e6, mag, color=_ACCENT, lw=1.6, zorder=3)

        # 标注截止频率与 -3 dB 线
        ax.axhline(-3, color="#999", lw=0.9, ls="--", zorder=2)
        for fc in self._cutoffs(cfg):
            ax.axvline(fc / 1e6, color="#F2C14E", lw=0.9, ls=":", zorder=2)
        # 阻带衰减参考
        if np.isfinite(m["atten"]):
            ax.axhline(-m["atten"], color="#5BA85B", lw=0.8, ls="-.",
                       alpha=0.7, zorder=2,
                       label=f"阻带 ≈ -{m['atten']:.0f} dB")

        ax.set_xlim(0, cfg["fs"] / 2e6)
        ax.set_ylim(max(-120, -m["atten"] * 1.4 if np.isfinite(m["atten"]) else -100), 5)
        ax.set_xlabel("频率 (MHz)", fontsize=10)
        ax.set_ylabel("幅度 (dB)", fontsize=10)
        ax.set_title(self._title(cfg) + "  幅频响应", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD", loc="upper right")
        self.canvas.draw()
        self.status.setText(
            f"幅频响应  |  f-3dB={m['f3dB']/1e6:.2f}MHz  "
            f"波纹={m['ripple']:.3f}dB  阻带={m['atten']:.1f}dB")

    # ════════ 绘图：相位响应 ════════
    def _plot_phase(self, cfg, b, a):
        fig = self.canvas.fig; fig.clf()
        fig.set_constrained_layout(True); fig.patch.set_facecolor("#07131F")
        ax = fig.add_subplot(111); _style_ax(ax)
        w, H, _, phase = freq_response(b, a, cfg["fs"], n=4096)
        ax.plot(w / 1e6, phase, color=_ACCENT, lw=1.6)
        for fc in self._cutoffs(cfg):
            ax.axvline(fc / 1e6, color="#F2C14E", lw=0.9, ls=":")
        ax.set_xlim(0, cfg["fs"] / 2e6)
        ax.set_xlabel("频率 (MHz)", fontsize=10)
        ax.set_ylabel("相位 (°)", fontsize=10)
        lin = "线性相位" if cfg["impl"] == "FIR" else "非线性相位"
        ax.set_title(self._title(cfg) + f"  相位响应（{lin}）", fontsize=10)
        self.canvas.draw()
        self.status.setText(f"相位响应  |  {lin}（FIR 为严格线性相位）")

    # ════════ 绘图：群时延 ════════
    def _plot_gd(self, cfg, b, a):
        fig = self.canvas.fig; fig.clf()
        fig.set_constrained_layout(True); fig.patch.set_facecolor("#07131F")
        ax = fig.add_subplot(111); _style_ax(ax)
        w, gd = group_delay(b, a, cfg["fs"], n=4096)
        ax.plot(w / 1e6, gd * 1e9, color=_ACCENT, lw=1.6)
        for fc in self._cutoffs(cfg):
            ax.axvline(fc / 1e6, color="#F2C14E", lw=0.9, ls=":")
        ax.set_xlim(0, cfg["fs"] / 2e6)
        ax.set_xlabel("频率 (MHz)", fontsize=10)
        ax.set_ylabel("群时延 (ns)", fontsize=10)
        ax.set_title(self._title(cfg) + "  群时延", fontsize=10)
        self.canvas.draw()
        self.status.setText(
            "群时延  |  通带内起伏越小，对宽带信号的色散失真越小")

    # ════════ 绘图：冲激/阶跃响应 ════════
    def _plot_impulse(self, cfg, b, a):
        fig = self.canvas.fig; fig.clf()
        fig.set_constrained_layout(True); fig.patch.set_facecolor("#07131F")
        ax1 = fig.add_subplot(211); _style_ax(ax1)
        ax2 = fig.add_subplot(212); _style_ax(ax2)

        N = 80 if cfg["impl"] == "IIR" else cfg["numtaps"] + 10
        imp = np.zeros(N); imp[0] = 1.0
        h = signal.lfilter(b, a, imp)
        step = signal.lfilter(b, a, np.ones(N))
        n = np.arange(N) / cfg["fs"] * 1e6

        ax1.stem(n, h, linefmt=_ACCENT, markerfmt="o", basefmt="#CCC")
        ax1.set_ylabel("冲激响应 h[n]", fontsize=10)
        ax1.set_title(self._title(cfg) + "  时域响应", fontsize=10)
        for ln in ax1.get_lines():
            ln.set_markersize(3)

        ax2.plot(n, step, color="#3DDC97", lw=1.5)
        ax2.axhline(1.0, color="#999", lw=0.8, ls="--")
        ax2.set_xlabel("时间 (μs)", fontsize=10)
        ax2.set_ylabel("阶跃响应", fontsize=10)
        self.canvas.draw()
        self.status.setText("冲激/阶跃响应  |  上：h[n]  下：阶跃（看过冲与稳定时间）")

    # ════════ 绘图：零极点图 ════════
    def _plot_pz(self, cfg, b, a):
        fig = self.canvas.fig; fig.clf()
        fig.set_constrained_layout(True); fig.patch.set_facecolor("#07131F")
        ax = fig.add_subplot(111); _style_ax(ax)
        z, p, k = signal.tf2zpk(b, a)
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), color="#AAB4C0", lw=1.0)
        ax.scatter(z.real, z.imag, s=55, facecolors="none",
                   edgecolors=_ACCENT, lw=1.6, label="零点", zorder=3)
        ax.scatter(p.real, p.imag, s=55, marker="x",
                   color="#F2C14E", lw=1.6, label="极点", zorder=3)
        ax.axhline(0, color="#DDD", lw=0.7); ax.axvline(0, color="#DDD", lw=0.7)
        ax.set_aspect("equal")
        lim = max(1.2, np.max(np.abs(np.r_[z, p, 1.0])) * 1.15)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Re(z)", fontsize=10); ax.set_ylabel("Im(z)", fontsize=10)
        stable = np.all(np.abs(p) < 1.0)
        ax.set_title(self._title(cfg) +
                     f"  零极点图（{'稳定' if stable else '不稳定'}）", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD", loc="upper right")
        self.canvas.draw()
        self.status.setText(
            f"零极点图  |  极点全在单位圆内 → 稳定；当前：{'稳定' if stable else '不稳定'}")

    # ════════ 绘图：RRC 脉冲响应 ════════
    def _plot_rrc_impulse(self, cfg, h):
        fig = self.canvas.fig; fig.clf()
        fig.set_constrained_layout(True); fig.patch.set_facecolor("#07131F")
        ax = fig.add_subplot(111); _style_ax(ax)
        sps = cfg["sps"]
        n = (np.arange(len(h)) - len(h) // 2) / sps
        # RC = RRC ⊛ RRC （理论零 ISI 脉冲）
        rc = np.convolve(h, h)
        rc = rc / np.max(rc)
        n_rc = (np.arange(len(rc)) - len(rc) // 2) / sps

        ax.plot(n, h / np.max(h), color=_ACCENT, lw=1.5, label="RRC（单级）")
        ax.plot(n_rc, rc, color="#3DDC97", lw=1.5, ls="--",
                label="RC = RRC⊛RRC")
        # 符号采样点（RC 在整数符号处过零，t=0 处为 1）
        sym = np.arange(-(cfg["span"]), cfg["span"] + 1)
        ax.scatter(sym, np.sinc(sym) * (np.abs(sym) < 0.5),
                   s=30, color="#444", zorder=4, label="符号采样点(零 ISI)")
        ax.axhline(0, color="#CCC", lw=0.7)
        ax.set_xlabel("时间 (符号周期 T)", fontsize=10)
        ax.set_ylabel("归一化幅度", fontsize=10)
        ax.set_title(f"RRC 脉冲成形  α={cfg['beta']:.2f}  sps={sps}  "
                     f"跨度={cfg['span']} 符号", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD", loc="upper right")
        self.canvas.draw()
        self.status.setText(
            f"RRC 脉冲  |  α={cfg['beta']:.2f}  抽头数={len(h)}  "
            f"RC 在整数符号点过零 → 零码间串扰")

    # ════════ 绘图：RRC 频响 ════════
    def _plot_rrc_freq(self, cfg, h):
        fig = self.canvas.fig; fig.clf()
        fig.set_constrained_layout(True); fig.patch.set_facecolor("#07131F")
        ax = fig.add_subplot(111); _style_ax(ax)
        sps = cfg["sps"]
        # 以符号率为基准，fs = sps（归一化），奈奎斯特带宽 = 0.5/T
        w, H = signal.freqz(h, worN=4096, fs=sps)
        mag = 20 * np.log10(np.abs(H) + 1e-12)
        ax.plot(w, mag, color=_ACCENT, lw=1.6, label=f"RRC α={cfg['beta']:.2f}")
        # 标注奈奎斯特带宽边界 0.5 与滚降边界 (1±α)/2
        ax.axvline(0.5, color="#999", lw=0.9, ls="--", label="奈奎斯特 0.5/T")
        ax.axvline((1 + cfg["beta"]) / 2, color="#F2C14E", lw=0.9, ls=":",
                   label=f"占用带宽 (1+α)/2={ (1+cfg['beta'])/2:.3f}/T")
        ax.set_xlim(0, sps / 2)
        ax.set_ylim(-80, 5)
        ax.set_xlabel("归一化频率 (× 符号率)", fontsize=10)
        ax.set_ylabel("幅度 (dB)", fontsize=10)
        ax.set_title(f"RRC 幅频响应  占用带宽 = (1+α)·Rs = "
                     f"{1+cfg['beta']:.2f}·Rs", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD", loc="upper right")
        self.canvas.draw()
        self.status.setText(
            f"RRC 频响  |  α={cfg['beta']:.2f} → 占用带宽 {1+cfg['beta']:.2f}×符号率")

    # ════════ 绘图：眼图 ════════
    def _plot_rrc_eye(self, cfg, h):
        fig = self.canvas.fig; fig.clf()
        fig.set_constrained_layout(True); fig.patch.set_facecolor("#07131F")
        ax = fig.add_subplot(111); _style_ax(ax)
        sps = cfg["sps"]
        rng = np.random.RandomState(2024)
        n_sym = 400
        syms = rng.choice([-1.0, 1.0], n_sym)            # BPSK 符号
        up = np.zeros(n_sym * sps); up[::sps] = syms
        # 收发各一级 RRC → 等效 RC，零 ISI
        tx = np.convolve(up, h, mode="same")
        rx = np.convolve(tx, h, mode="same")
        rx /= np.max(np.abs(rx))

        span = 2 * sps
        n_trace = (len(rx) - span) // sps
        t = np.linspace(-1, 1, span)
        for i in range(20, n_trace - 1):
            seg = rx[i * sps: i * sps + span]
            if len(seg) == span:
                ax.plot(t, seg, color=_ACCENT, lw=0.5, alpha=0.35)
        ax.axvline(0, color="#F2C14E", lw=1.0, ls="--", label="判决时刻")
        ax.set_xlabel("时间 (符号周期 T)", fontsize=10)
        ax.set_ylabel("归一化幅度", fontsize=10)
        ax.set_title(f"接收眼图  RRC⊛RRC=RC  α={cfg['beta']:.2f}  "
                     f"（判决点张开越大越好）", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.95, edgecolor="#DDD", loc="upper right")
        self.canvas.draw()
        self.status.setText(
            f"眼图  |  收发级联 RRC=升余弦，判决时刻 t=0 处眼睛张开 → 无 ISI")

    # ── 通用小工具 ────────────────────────────────────────
    def _cutoffs(self, cfg):
        if cfg["band"] in ("带通", "带阻"):
            return [cfg["fc1"], cfg["fc2"]]
        return [cfg["fc1"]]

    def _title(self, cfg):
        if cfg["impl"] == "外部系数":
            return "外部导入滤波器"
        if cfg["impl"] == "IIR":
            tag = f"{cfg['iir_type']} {cfg['band']} N={cfg['order']}"
        else:
            tag = f"FIR {cfg['fir_method']} {cfg['band']} {cfg['numtaps']}抽头"
        return tag

    # ── 保存 ──────────────────────────────────────────────
    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "filter_model.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.canvas.save(path)
            QMessageBox.information(self, "保存成功", f"已保存：\n{path}")
