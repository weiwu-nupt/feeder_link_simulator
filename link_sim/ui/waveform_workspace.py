"""NodeGraphQt transceiver canvas with three payload observation points."""

import copy
import os
import threading

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import rcParams, font_manager

# Qt.py must select the same binding as the rest of this desktop application.
os.environ["QT_PREFERRED_BINDING"] = "PySide6"
try:
    from NodeGraphQt import BaseNode, NodeGraph
except ImportError as exc:
    raise ImportError("缺少 NodeGraphQt 画布依赖。请在启动本软件的 Python 环境执行："
                      "python -m pip install --upgrade -r requirements.txt") from exc

from link_sim.algorithm.waveform_chain import (
    Stage, Wave, KINDS, DEVICE_KINDS, DUAL_KINDS, AWGN_CHANNEL_KINDS, SOURCE_KINDS,
    OBSERVATION_KINDS, MAX_SAMPLES, default_link,
    default_settings, edge_parts, export_wave, output_key, upstream_graph,
    import_source, load_chain, model_from_dict, model_to_dict, order_graph,
    run_graph, save_chain, source_wave,
)
from link_sim.algorithm.device_models import json_text
from link_sim.algorithm.waveform_spectrum import amplitude_spectrum, signal_statistics
from link_sim.ui.model_library import ModelLibraryDialog
from link_sim.ui.numeric_inputs import hide_step_buttons
from link_sim.ui.waveform_graphics import WaveformNodeItem, CanvasPanFilter


class SignalNode(BaseNode):
    __identifier__ = "feeder.waveform"
    NODE_NAME = "模块"
    KIND = "gain"

    def __init__(self):
        super().__init__(qgraphics_item=WaveformNodeItem)
        self.create_property("settings", default_settings(self.KIND))
        self.create_property("device_model", None)
        if self.KIND not in SOURCE_KINDS:
            for label in (("A 输入", "B 输入") if self.KIND in DUAL_KINDS else ("I/Q 输入",)):
                self.add_input(label, color=(83, 191, 221), display_name=False)
        for label in (("A 输出", "B 输出") if self.KIND in DUAL_KINDS else ("I/Q 输出",)):
            self.add_output(label, multi_output=True, color=(83, 222, 177), display_name=False)
        self.set_color(*(dict(source=(28, 112, 116), dac=(47, 91, 146),
                             filter=(52, 92, 142), mixer=(83, 74, 141), pa=(131, 75, 64))
                         .get(self.KIND, (55, 101, 115))))


class SourceNode(SignalNode):
    NODE_NAME, KIND = "基带信号源", "source"


class DPDNode(SignalNode):
    NODE_NAME, KIND = "数字预失真（DPD）", "dpd"


class DACNode(SignalNode):
    NODE_NAME, KIND = "AD/DA", "dac"


class FilterNode(SignalNode):
    NODE_NAME, KIND = "滤波器", "filter"


class PrecompensationNode(SignalNode):
    NODE_NAME, KIND = "预补偿", "precomp"


class LowpassNode(SignalNode):
    NODE_NAME, KIND = "低通滤波", "lowpass"


class MixerNode(SignalNode):
    NODE_NAME, KIND = "上变频器", "mixer"


class PANode(SignalNode):
    NODE_NAME, KIND = "功率放大器", "pa"


class GainNode(SignalNode):
    NODE_NAME, KIND = "增益", "gain"


class FrequencyNode(SignalNode):
    NODE_NAME, KIND = "频率偏移", "frequency"


class NoiseNode(SignalNode):
    NODE_NAME, KIND = "加性白噪声", "awgn"


class NoiseFloorNode(SignalNode):
    NODE_NAME, KIND = "底噪模型", "noise"


class FramedSourceNode(SignalNode):
    NODE_NAME, KIND = "按帧发射信号", "frame_source"


class TransmitRRCNode(SignalNode):
    NODE_NAME, KIND = "发射 RRC", "rrc_tx"


class ChannelNode(SignalNode):
    NODE_NAME, KIND = "无线信道", "channel"


class DualChannelNode(SignalNode):
    NODE_NAME, KIND = "无线信道（双路）", "dual_channel"


class XPICNode(SignalNode):
    NODE_NAME, KIND = "交叉极化信道（XPIC）", "xpic"


class FastICANode(SignalNode):
    NODE_NAME, KIND = "FastICA 双路分离", "fastica"


class DownmixerNode(SignalNode):
    NODE_NAME, KIND = "下变频器", "downmixer"


class ReceiveADCNode(SignalNode):
    NODE_NAME, KIND = "接收 ADC", "rx_adc"


class ReceiveRRCNode(SignalNode):
    NODE_NAME, KIND = "接收 RRC", "rrc_rx"


class SynchronizationNode(SignalNode):
    NODE_NAME, KIND = "同步模块", "sync"


class ReceiverNode(SignalNode):
    NODE_NAME, KIND = "接收信号", "sink"


NODE_CLASSES = {cls.KIND: cls for cls in (
    SourceNode, DPDNode, DACNode, FilterNode, PrecompensationNode, LowpassNode, MixerNode, PANode, GainNode, FrequencyNode, NoiseNode, NoiseFloorNode,
    FramedSourceNode, TransmitRRCNode, ChannelNode, DualChannelNode, XPICNode, FastICANode, DownmixerNode,
    ReceiveADCNode, ReceiveRRCNode, SynchronizationNode, ReceiverNode)}


# key, label, divisor for display units, min, max, decimals
FIELDS = {
    "frame_source": (("symbol_rate", "码元速率 (Mbaud)", 1e6, .000001, 1e6, 6),
                     ("segments", "每突发帧数", 1, 1, 32, 0),
                     ("payload_symbols", "业务数据符号数（每帧）", 1, 1, MAX_SAMPLES, 0),
                     ("seed", "业务随机种子（0 = 随机）", 1, 0, 2**32-1, 0)),
    "rrc_tx": (),
    "rrc_rx": (), "sink": (),
    "sync": (("threshold", "同步相关门限", 1, .01, 1, 3),),
    "channel": (("snr_db", "信噪比 SNR (dB)", 1, -40, 200, 2),),
    "xpic": (("xpd_a", "发射 A 极化 XPD (dB)", 1, 0, 100, 2),
             ("xpd_b", "发射 B 极化 XPD (dB)", 1, 0, 100, 2),
             ("delay_a", "A 相对时延（采样）", 1, 0, 4096, 3),
             ("delay_b", "B 相对时延（采样）", 1, 0, 4096, 3),
             ("snr_db", "附加噪声 SNR (dB)", 1, -40, 200, 2)),
    "fastica": (("window_offset", "同步后训练窗口偏移（采样）", 1, 0, 200000, 0),
                ("window_length", "训练窗口长度（采样）", 1, 128, 200000, 0)),
    "source": (("fs", "采样率 (MHz)", 1e6, .000001, 1e6, 6),
               ("count", "采样点数", 1, 256, MAX_SAMPLES, 0),
               ("sps", "每符号采样点", 1, 2, 256, 0),
               ("amplitude", "标称 RMS 幅度 (归一化)", 1, .000001, 1, 6),
               ("seed", "随机种子", 1, 0, 2**32-1, 0)),
    "gain": (("gain_db", "增益 (dB)", 1, -100, 100, 3),),
    "frequency": (("offset_hz", "频偏 (MHz)", 1e6, -1e6, 1e6, 6),),
    "awgn": (("snr_db", "输入信噪比 (dB)", 1, -40, 200, 3),
             ("seed", "随机种子", 1, 0, 2**32-1, 0)),
}
FIELDS["dual_channel"] = FIELDS["channel"]


class StageDialog(QDialog):
    def __init__(self, stage, parent=None):
        super().__init__(parent)
        self.stage = copy.deepcopy(stage)
        self.setWindowTitle("配置模块 · "+stage.name)
        self.resize(620, 590)
        root = QVBoxLayout(self)
        content = root
        self.content_scroll = None
        self._initial_fit_done = False
        if stage.kind == "frame_source":
            body = QWidget()
            content = QVBoxLayout(body)
            content.setContentsMargins(0, 0, 0, 0)
            self.content_scroll = QScrollArea()
            self.content_scroll.setWidgetResizable(True)
            self.content_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.content_scroll.setWidget(body)
            root.addWidget(self.content_scroll, 1)
        self.enabled = QCheckBox("启用信号源（关闭时输出零 I/Q）" if stage.kind in SOURCE_KINDS else
                                "启用模块（关闭时 Bypass）")
        self.enabled.setChecked(stage.enabled)
        content.addWidget(self.enabled)
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        content.addWidget(self.model_label)
        buttons = QHBoxLayout()
        if stage.kind in DEVICE_KINDS:
            for label, action in (("从模型库加载…", self.choose_model),):
                button = QPushButton(label)
                button.clicked.connect(action)
                buttons.addWidget(button)
        elif stage.kind == "source":
            for label, action in (("导入已调制 I/Q…", self.import_iq), ("恢复测试信号", self.clear_iq)):
                button = QPushButton(label)
                button.clicked.connect(action)
                buttons.addWidget(button)
        content.addLayout(buttons)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.fields = {}
        self.modulation = None
        self.frame_editor = None
        if stage.kind == "source":
            self.modulation = QComboBox()
            self.modulation.addItems(["BPSK", "QPSK", "16QAM"])
            self.modulation.setCurrentText(stage.settings.get("modulation", "QPSK"))
            form.addRow("未编码测试调制", self.modulation)
        settings = dict(default_settings(stage.kind), **stage.settings)
        if stage.kind == "frame_source":
            from link_sim.algorithm.frame_config import symbol_rate, editor_settings
            settings = editor_settings(settings)
            settings["symbol_rate"] = symbol_rate(stage.settings)
        for key, label, scale, low, high, decimals in FIELDS.get(stage.kind, ()):
            edit = QDoubleSpinBox()
            hide_step_buttons(edit)
            edit.setDecimals(decimals)
            edit.setRange(low, high)
            edit.setValue(settings[key]/scale)
            edit.setGroupSeparatorShown(True)
            self.fields[key] = (edit, scale, decimals)
            form.addRow(label, edit)
        if stage.kind == "frame_source":
            from .frame_source_editor import FrameSourceEditor
            self.resize(1180, 720)
            content.addLayout(form)
            self.frame_editor = FrameSourceEditor(settings, self.fields, self)
            content.addWidget(self.frame_editor, 2)
        else:
            content.addLayout(form)
        self.details = None
        if stage.kind != "frame_source":
            self.details = QPlainTextEdit()
            self.details.setReadOnly(True)
            content.addWidget(self.details, 1)
        if stage.kind in DEVICE_KINDS:
            hint_kind = {"rx_adc": "dac", "downmixer": "mixer", "rrc_tx": "filter", "rrc_rx": "filter", "lowpass": "filter"}.get(stage.kind, stage.kind)
            hint = QLabel({
                "dpd": "在时域处理复数 I/Q。请在下游功放节点加载与训练时配对的功放模型。",
                "dac": "等效噪声使用原模型的估计量化噪声基底，自动匹配链路采样率；不叠加杂散。",
                "filter": "在时域处理复数 I/Q。FIR/RRC 使用 h[n] 卷积，IIR 使用原系数递推。",
                "precomp": "选择对应低通模型，自动设计通带逆响应；与配对低通节点同步使用同一模型。阻带限制逆增益。",
                "mixer": "读取模型的相噪锚点，按输入 I/Q 的采样率和点数生成本振相噪，再逐点旋转。",
                "noise": "按模型底噪密度和链路采样率生成复高斯白噪声，逐点叠加到输入。",
                "pa": "按保存的 MP/GMP 系数计算非线性和记忆效应，保留信号幅度基准。",
            }[hint_kind])
            hint.setWordWrap(True)
            content.addWidget(hint)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.button(QDialogButtonBox.StandardButton.Ok).setText("应用")
        box.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        root.addWidget(box)
        self.refresh_model()

    def available_area(self):
        parent = self.parentWidget()
        screen = parent.window().screen() if parent is not None else self.screen()
        return screen.availableGeometry()

    def fit_to_screen(self):
        # Qt screen geometry already uses logical pixels, including display scaling.
        area = self.available_area().adjusted(12, 12, -12, -12)
        frame = self.frameGeometry()
        border_width = max(0, frame.width()-self.width())
        border_height = max(0, frame.height()-self.height())
        self.resize(min(1180, max(1, area.width()-border_width)),
                    min(720, max(1, area.height()-border_height)))
        self.layout().activate()
        frame = self.frameGeometry()
        parent = self.parentWidget()
        frame.moveCenter(parent.window().frameGeometry().center() if parent is not None else area.center())
        frame.moveLeft(max(area.left(), min(frame.left(), area.right()-frame.width()+1)))
        frame.moveTop(max(area.top(), min(frame.top(), area.bottom()-frame.height()+1)))
        self.move(frame.topLeft())

    def showEvent(self, event):
        super().showEvent(event)
        if self.content_scroll is not None and not self._initial_fit_done:
            self._initial_fit_done = True
            self.fit_to_screen()
            # Native title-bar dimensions can settle after the first show event.
            QTimer.singleShot(0, self.fit_to_screen)

    def refresh_model(self):
        stage = self.stage
        if stage.kind == "frame_source":
            self.model_label.setText("同步头保持固定序列；每帧使用相同的调制编码配置，业务数据独立生成")
            return
        model = stage.model
        imported = stage.kind == "source" and "iq" in stage.settings
        if model:
            self.model_label.setText(f"已加载：{model.name or '未命名模型'}")
            arrays = "\n".join(f"{k}：{a.shape} {a.dtype}" for k, a in model.arrays.items())
            self.details.setPlainText(json_text(model.parameters)+"\n"+arrays)
            if stage.kind in ("pa", "dpd") and model.parameters.get("sample_rate"):
                rate = model.parameters["sample_rate"]
                self.details.setPlainText(
                    f"模型记录采样率：{rate/1e6:g} MHz（仅作记录）\n"
                    "按保存的离散时域系数直接计算，延迟 1 点即当前信号的 1 个采样点。\n"
                    "不检查功放/DPD 与输入采样率是否一致，不重采样或换算记忆延迟。\n\n"
                    +self.details.toPlainText())
            if stage.kind in ("rrc_tx", "rrc_rx"):
                p = model.parameters
                self.details.setPlainText(
                    f"每码元采样点：{p['sps']}\n滚降系数：{p['beta']:g}\n跨度：{p['span']} 个码元\n"
                    f"抽头数：{len(model.arrays.get('h', model.arrays.get('b', [])))}\n\n"
                    "参数和抽头由加载的模型提供。\n实际采样率 = 码元速率 × 每码元采样点。")
            if stage.kind == "precomp":
                self.details.appendPlainText("\n预补偿 = 对应低通 H(f) 的通带逆响应，时域卷积实现；最大补偿 20 dB，深阻带不反演。")
        elif imported:
            self.model_label.setText(f"外部波形：{stage.settings.get('filename', '')} · {len(stage.settings['iq']):,} 点")
            self.details.setPlainText("保留文件 I/Q 幅度；文件中的 sampleRate 优先作为初始采样率。\n"
                                      "编码和调制已由外部信号源完成。本页不重新编码或归一化。")
        else:
            if stage.kind == "rrc_tx":
                self.model_label.setText("未加载 RRC 模型：请从模型库加载")
            elif stage.kind == "rrc_rx":
                self.model_label.setText("接收匹配 RRC：默认使用发射端实际抽头；也可加载相同 RRC 模型")
            elif stage.kind in DEVICE_KINDS:
                self.model_label.setText("未加载器件模型（理想直通）")
            elif stage.kind in SOURCE_KINDS:
                self.model_label.setText("当前：内置测试信号")
            descriptions = {
                "rrc_tx": "加载滤波器库中的 RRC 成形模型，使用模型中的抽头和采样倍数完成上采样及完整时域卷积。",
                "rrc_rx": "使用帧约定的 RRC 抽头完成接收匹配滤波。",
                "sync": "根据接收波形与已知同步头进行相关检测及小数定时插值。",
                "sink": "帧头辅助恢复相位与幅度，软解调并译码，输出接收比特、BER 和 EVM。",
                "channel": "AWGN 信道按输入波形的平均功率和设定信噪比叠加复高斯白噪声。",
                "dual_channel": "A/B 两路共用信噪比，各自按输入平均功率叠加独立的复高斯白噪声。",
                "xpic": "双路共享时间轴；2×2 交叉极化耦合。XPD 按发射 A/B 极化分别配置。",
                "fastica": "两路联合估计复数瞬时分离矩阵；以已知同步头定位训练窗口。\n"
                           "适用于近似独立、非高斯源及瞬时满秩混合；退化或未收敛时报告失败。",
                "dpd": "加载数字预失真模型后，使用保存的多项式系数生成预失真 I/Q。",
                "source": "未编码随机符号，矩形脉冲。实际编码调制信号请通过 I/Q 文件导入。",
                "dac": "请从原 ADC/DAC 模型库加载器件模型。未加载时为理想直通。\n"
                       "读取模型的估计量化噪声基底，换算噪声功率；链路输出 = 链路输入 + 等效噪声。\n"
                       "自动匹配链路采样率，忽略杂散。原模型参数不在此修改。",
                "filter": "加载滤波器模型后，使用其冲激响应 h[n] 或递推系数计算输出。",
                "lowpass": "选择低通模型；与配对的预补偿节点同步使用同一个模型。",
                "precomp": "选择对应低通模型，自动计算 H(f) 的通带逆响应。与配对低通节点共用模型，可各自 Bypass。",
                "mixer": "加载混频器模型后，应用已保存的相噪锚点。",
                "pa": "加载功放 MP/GMP 模型后，使用保存的辨识系数计算输出。",
                "noise": "加载底噪模型后，以绝对噪声密度生成复高斯噪声并叠加到输入。",
                "gain": "按设置的增益在时域逐点缩放复数 I/Q。",
                "frequency": "对应信号源×复乘法的频移；请为频偏预留采样带宽。",
                "awgn": "按输入有效区间平均功率生成复高斯噪声；种子固定时可复现。",
            }
            self.details.setPlainText(descriptions[{"rx_adc": "dac", "downmixer": "mixer"}.get(stage.kind, stage.kind)])
        for key, (edit, _, _) in self.fields.items():
            edit.setEnabled(not imported or key == "fs")
        if stage.kind in ("dac", "rx_adc") and model:
            p = model.parameters
            from link_sim.algorithm.adc_noise_core import calibrate_model_noise
            estimate = calibrate_model_noise(model)
            self.details.setPlainText(f"原器件模型：{p['device_profile']}\n"
                                      f"量化位数：{p['n_bits']} bit；模型采样率：{p['fs']/1e6:g} MHz\n"
                                      f"校准基波：{p['f_sig']/1e6:g} MHz；FFT 点数：{p['nfft']}\n"
                                      f"模型 NSR：{p['noise_density_dbfs_hz']:g} dBFS/Hz\n\n"
                                      f"估计量化噪声基底：{estimate.floor_dbfs:.2f} dBFS/频点\n"
                                      f"FFT 频点宽度：{estimate.bin_width_hz:g} Hz\n"
                                      f"每 Hz 噪声功率密度 D_n：{estimate.density:.6e} 归一化功率/Hz\n"
                                      f"对应噪声密度：{estimate.density_dbfs_hz:.4f} dBFS/Hz\n\n"
                                      "使用模型噪声基底换算 Pn，生成链路 I/Q 等效噪声；不叠加杂散。")
        if self.modulation:
            self.modulation.setEnabled(not imported)

    def choose_model(self):
        try:
            dialog = ModelLibraryDialog(self, DEVICE_KINDS[self.stage.kind])
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.apply_loaded_model(dialog.selected_model)
        except Exception as exc:
            QMessageBox.warning(self, "加载模型失败", str(exc))

    def apply_loaded_model(self, model):
        model = model.validated()
        if self.stage.kind in ("rrc_tx", "rrc_rx"):
            from link_sim.algorithm.rrc_stage import SavedRRC
            SavedRRC(model)
        if self.stage.kind in ("precomp", "lowpass"):
            from link_sim.algorithm.filter_inverse import validate_lowpass
            validate_lowpass(model)
        self.stage.model = model
        self.stage.settings.pop("model_revision", None)
        self.refresh_model()

    def import_iq(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入基带 I/Q 波形", "", "波形 (*.npy *.npz *.mat *.csv *.txt)")
        if path:
            try:
                imported = import_source(path, self.fields["fs"][0].value()*1e6)
                self.stage.settings.update(imported)
                self.fields["fs"][0].setValue(imported["fs"]/1e6)
                self.refresh_model()
            except Exception as exc:
                QMessageBox.warning(self, "波形导入失败", str(exc))

    def clear_iq(self):
        self.stage.settings.pop("iq", None)
        self.stage.settings.pop("filename", None)
        self.refresh_model()

    def accept(self):
        settings = dict(default_settings(self.stage.kind), **self.stage.settings)
        settings.pop("model_revision", None)
        if self.stage.kind in DEVICE_KINDS:
            settings = {"paired_node": settings["paired_node"]} if settings.get("paired_node") else {}
        elif self.stage.kind in AWGN_CHANNEL_KINDS:
            settings = {}
        for key, (edit, scale, decimals) in self.fields.items():
            value = edit.value()*scale
            settings[key] = int(value) if decimals == 0 else value
        if self.modulation:
            settings["modulation"] = self.modulation.currentText()
        try:
            if self.stage.kind == "source":
                source_wave(settings)
            elif self.stage.kind == "frame_source":
                from link_sim.algorithm.frame_config import frame_summary
                settings.update(self.frame_editor.settings())
                for key in ("fs", "sps", "k", "n", "modulation", "rolloff", "span", "amplitude", "words", "business_frames"):
                    settings.pop(key, None)
                frame_summary(settings)
            elif self.stage.kind in ("rrc_tx", "rrc_rx") and self.stage.model is not None:
                from link_sim.algorithm.rrc_stage import SavedRRC
                SavedRRC(self.stage.model)
            self.stage.settings = settings
            self.stage.enabled = self.enabled.isChecked()
        except Exception as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        super().accept()


class WorkerSignals(QObject):
    done = Signal(object, str)
    progress = Signal(int, int, str)


class ChainWorker(QRunnable):
    def __init__(self, stages, edges, target=None):
        super().__init__()
        self.signals = WorkerSignals()
        self.stages, self.edges = copy.deepcopy(stages), list(edges)
        self.target = target
        self.cancel = threading.Event()
        self.observations, self.observation_errors = {}, {}

    def run(self):
        try:
            result = run_graph(self.stages, self.edges,
                               target=self.target, progress=self.signals.progress.emit, cancelled=self.cancel.is_set)
            from link_sim.algorithm.framed_link import business_output
            for stage in self.stages:
                if stage.kind in OBSERVATION_KINDS and stage.key in result:
                    try:
                        self.observations[stage.key] = business_output(result[stage.key], stage.kind)
                    except (ValueError, RuntimeError) as exc:
                        self.observation_errors[stage.key] = str(exc)
            self.signals.done.emit(result, "")
        except Exception as exc:
            self.signals.done.emit(None, str(exc))


class WaveformWorkspace(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        fonts = {f.name for f in font_manager.fontManager.ttflist}
        for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
            if family in fonts:
                rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
                rcParams["axes.unicode_minus"] = False
                break
        self.setWindowTitle("全链路仿真 · 收发链路")
        self.resize(1300, 860)
        self.outputs = {}
        self.observations, self.observation_errors = {}, {}
        self.worker = None
        self.rebuilding = False
        self._syncing_pair = False
        self._shown_once = False
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)
        self.edit_row = QWidget()
        controls = QHBoxLayout(self.edit_row)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(QLabel("收发模式"))
        self.link_mode = QComboBox()
        self.link_mode.addItems(["单通道", "双极化 A / B"])
        controls.addWidget(self.link_mode)
        self._button(controls, "重建收发链路", self.reset_chain)
        self._button(controls, "适应画布", self.fit_canvas)
        controls.addStretch()
        self._button(controls, "打开方案…", self.open_scheme)
        self._button(controls, "保存方案…", self.save_scheme)
        root.addWidget(self.edit_row)
        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        self.status = QLabel("选择观察点后可计算该处的业务输出")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#8eafc3")
        run_row.addWidget(self.status, 1)
        self.preview_button = self._button(run_row, "计算所选输出", self.run_selected)
        self.preview_button.setToolTip("计算到发射信号、功放或接收信号观察点；后续模块无需配置。")
        self.run_button = self._button(run_row, "运行完整链路", self.run_chain)
        self.run_button.setProperty("class", "primary")
        root.addLayout(run_row)
        self.graph = NodeGraph()
        self.graph.register_nodes(list(NODE_CLASSES.values()))
        self.graph.set_acyclic(True)
        self.graph.set_background_color(8, 21, 34)
        self.graph.set_grid_color(21, 43, 60)
        self.graph.set_pipe_style(1)
        self.pan_filter = CanvasPanFilter(self.graph.viewer())
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.graph.widget.setMinimumHeight(240)
        splitter.addWidget(self.graph.widget)
        observer = QWidget()
        observer_layout = QVBoxLayout(observer)
        observer_layout.setContentsMargins(0, 5, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("观察点"))
        self.point = QComboBox()
        self.point.setMinimumWidth(260)
        row.addWidget(self.point)
        self.export_button = self._button(row, "导出业务 I/Q…", self.export_selected)
        self.export_button.setEnabled(False)
        self.summary = QLabel("选择观察点，计算到该模块即可查看输出")
        self.summary.setWordWrap(True)
        row.addWidget(self.summary, 1)
        observer_layout.addLayout(row)
        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("图形"))
        self.plot_view = QComboBox()
        for label, key in (("全部", "all"), ("频谱", "spectrum"), ("时域波形", "time"), ("I/Q 散点", "iq")):
            self.plot_view.addItem(label, key)
        plot_row.addWidget(self.plot_view)
        self.spectrum_info = QLabel()
        plot_row.addWidget(self.spectrum_info)
        plot_row.addStretch()
        observer_layout.addLayout(plot_row)
        self.figure = Figure(figsize=(12, 2.8), dpi=100, facecolor="#091b29")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(200)
        observer_layout.addWidget(self.canvas, 1)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#8eafc3;padding:3px")
        observer_layout.addWidget(self.note)
        splitter.addWidget(observer)
        splitter.setSizes([360, 330])
        root.addWidget(splitter, 1)
        hint = QLabel("拖动空白处、右键或中键平移画布；滚轮缩放，Shift 拖动框选。\n"
                      "预补偿 → DAC/ADC → 低通滤波；各级可 Bypass。仅观察和导出发射、功放及接收业务信号。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#7896ab;font-size:11px")
        root.addWidget(hint)
        self.graph.node_double_clicked.connect(self.configure_node)
        self.graph.node_selected.connect(self.select_node)
        for sig in (self.graph.port_connected, self.graph.port_disconnected,
                    self.graph.nodes_deleted, self.graph.node_created):
            sig.connect(self.invalidate)
        self.graph.property_changed.connect(self.property_changed)
        self.point.currentIndexChanged.connect(self.draw_output)
        self.plot_view.currentIndexChanged.connect(self.draw_output)
        self.link_mode.currentIndexChanged.connect(self.reset_chain)
        self.reset_chain()

    def property_changed(self, node, name, value):
        # Selection, layout and node colors must not discard computed waveforms.
        if name in ("device_model", "settings", "disabled") and isinstance(node, SignalNode):
            if name == "device_model" and not self._syncing_pair and node.KIND in ("precomp", "lowpass"):
                mate = self.graph.get_node_by_id(node.get_property("settings").get("paired_node"))
                if mate is not None and mate.get_property("device_model") != value:
                    self._syncing_pair = True
                    try:
                        mate.set_property("device_model", copy.deepcopy(value), push_undo=False)
                    finally:
                        self._syncing_pair = False
            self.update_node_caption(node)
        if name in ("settings", "device_model", "disabled"):
            self.invalidate()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._shown_once:
            self._shown_once = True
            QTimer.singleShot(0, self.fit_canvas)

    @staticmethod
    def _button(layout, text, action):
        button = QPushButton(text)
        button.clicked.connect(action)
        layout.addWidget(button)
        return button

    def fit_canvas(self):
        if self.graph.all_nodes():
            self.graph.clear_selection()
            self.graph.select_all()
            self.graph.fit_to_selection()
            self.graph.clear_selection()

    def stage_from_node(self, node):
        return Stage(node.id, node.KIND, self.display_name(node), copy.deepcopy(node.get_property("settings")),
                     model_from_dict(node.get_property("device_model")), not node.disabled())

    def snapshot(self, target=None):
        nodes = self.graph.all_nodes()
        edges = []
        for node in nodes:
            for dst_port, port in enumerate(node.input_ports()):
                for p in port.connected_ports():
                    src_port = p.node().output_ports().index(p)
                    edge = (p.node().id, node.id)
                    edges.append(edge if src_port == dst_port == 0 else (*edge, src_port, dst_port))
        required = ({node.id for node in nodes} if target is None else
                    upstream_graph([node.id for node in nodes], edges, target)[0])
        stages = []
        for node in nodes:
            if node.id in required:
                if not isinstance(node, SignalNode):
                    raise ValueError("计算路径包含不支持的节点，请删除非波形模块")
                stages.append(self.stage_from_node(node))
            elif isinstance(node, SignalNode):
                # Keep IDs/kinds for reproducible noise streams without loading unused models.
                stages.append(Stage(node.id, node.KIND, node.name()))
        return stages, edges

    def create_stage(self, kind, pos=None, stage=None):
        cls = NODE_CLASSES[kind]
        node = self.graph.create_node(cls.type_, name=stage.name if stage else KINDS[kind], pos=pos,
                                      push_undo=False)
        if stage:
            node.set_property("settings", copy.deepcopy(stage.settings), push_undo=False)
            node.set_property("device_model", model_to_dict(stage.model), push_undo=False)
            node.set_property("disabled", not stage.enabled, push_undo=False)
        self.update_node_caption(node)
        return node

    def update_node_caption(self, node):
        label = node.name().split("\n", 1)[0].removesuffix(" · 关闭").removesuffix(" · 旁路")
        data = node.get_property("device_model")
        if data:
            text = data.get('name') or '库模型'
        elif node.KIND == "source" and "iq" in node.get_property("settings"):
            text = "外部 I/Q"
        else:
            text = ""
        if node.name() != label:
            node.set_name(label)
        node.view.text_item.setPlainText(self.display_name(node))
        node.view.set_caption(text, not node.disabled() and (node.KIND not in DEVICE_KINDS or data is not None))

    def display_name(self, node):
        name = node.name().split("\n", 1)[0].removesuffix(" · 关闭").removesuffix(" · 旁路")
        dual = sum(n.KIND in SOURCE_KINDS for n in self.graph.all_nodes() if isinstance(n, SignalNode)) > 1
        prefix = name[:4] if name.startswith(("A · ", "B · ")) else ""
        if prefix and not dual:
            name = name[4:]
        if node.KIND == "lowpass":
            return (prefix if dual else "")+"低通滤波"
        return name

    def remap_pairs(self, nodes):
        for node in nodes.values():
            settings = node.get_property("settings")
            if settings.get("paired_node") in nodes:
                node.set_property("settings", dict(settings, paired_node=nodes[settings['paired_node']].id), push_undo=False)
        for node in nodes.values():
            if node.KIND in ("precomp", "lowpass") and node.get_property("device_model") is not None:
                mate = self.graph.get_node_by_id(node.get_property("settings").get("paired_node"))
                if mate is not None and mate.get_property("device_model") is None:
                    mate.set_property("device_model", copy.deepcopy(node.get_property("device_model")), push_undo=False)

    def configure_node(self, node):
        if self.worker:
            return
        try:
            dialog = StageDialog(self.stage_from_node(node), self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                node.set_property("settings", dialog.stage.settings)
                node.set_property("device_model", model_to_dict(dialog.stage.model))
                node.set_disabled(not dialog.stage.enabled)
                self.update_node_caption(node)
                self.invalidate()
                self.select_node(node)
        except Exception as exc:
            QMessageBox.warning(self, "模块配置失败", str(exc))

    def reset_chain(self):
        if self.worker:
            return
        self.rebuilding = True
        self.graph.clear_session()
        stages, edges, positions = default_link(self.link_mode.currentIndex() == 1)
        nodes = {stage.key: self.create_stage(stage.kind, positions[stage.key], stage) for stage in stages}
        self.remap_pairs(nodes)
        for edge in edges:
            src, dst, src_port, dst_port = edge_parts(edge)
            nodes[src].output(src_port).connect_to(nodes[dst].input(dst_port), push_undo=False)
        self.rebuilding = False
        self.invalidate()
        QTimer.singleShot(0, self.fit_canvas)

    def invalidate(self, *_):
        if self.rebuilding:
            return
        self.outputs = {}
        self.observations, self.observation_errors = {}, {}
        for node in self.graph.all_nodes():
            if isinstance(node, SignalNode):
                self.update_node_caption(node)
        self.refresh_observation_points()
        self.export_button.setEnabled(False)
        self.summary.setText("参数或连接已更新，可选择观察点并计算业务输出")
        self.note.setText("")
        self.spectrum_info.clear()
        self.figure.clear()
        self.canvas.draw_idle()

    def refresh_observation_points(self, preferred=None):
        selected = preferred or self.point.currentData()
        self.point.blockSignals(True)
        self.point.clear()
        for node in self.graph.all_nodes():
            if not isinstance(node, SignalNode) or node.KIND not in OBSERVATION_KINDS:
                continue
            for port in range(2 if node.KIND in DUAL_KINDS else 1):
                key = output_key(node.id, port)
                label = self.display_name(node)
                if node.KIND in DUAL_KINDS:
                    label += " B" if port else " A"
                pending = "（待计算）" if key not in self.outputs else ""
                self.point.addItem(label+" → 输出"+pending, key)
        index = self.point.findData(selected)
        self.point.setCurrentIndex(index if index >= 0 else 0)
        self.point.blockSignals(False)

    def select_node(self, node):
        index = self.point.findData(node.id)
        if index >= 0:
            self.point.setCurrentIndex(index)

    def run_chain(self):
        self.start_run()

    def run_selected(self):
        key = self.point.currentData()
        if key is None:
            self.status.setText("请先选择一个观察点")
            return
        self.start_run(key.removesuffix("::error").removesuffix("::B"))

    def start_run(self, target=None):
        if self.worker:
            return
        try:
            stages, edges = self.snapshot(target)
            order_graph(stages, edges, target=target)
        except Exception as exc:
            QMessageBox.warning(self, "请检查链路", str(exc))
            return
        self.run_observation = self.point.currentData() if target is not None else None
        self.invalidate()
        self.worker = ChainWorker(stages, edges, target)
        self.worker.signals.done.connect(self.run_finished)
        self.worker.signals.progress.connect(self.run_progress)
        self.set_busy(True)
        self.status.setText("正在计算…")
        QThreadPool.globalInstance().start(self.worker)

    def set_busy(self, busy):
        self.edit_row.setEnabled(not busy)
        self.graph.widget.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.preview_button.setEnabled(not busy)
        self.point.setEnabled(not busy)

    def run_progress(self, index, total, name):
        self.status.setText(f"正在计算 {index}/{total} · {name.replace(chr(10), ' ')}")

    def run_finished(self, outputs, error):
        cancelled = self.worker.cancel.is_set()
        observations, observation_errors = self.worker.observations, self.worker.observation_errors
        self.worker = None
        self.set_busy(False)
        if cancelled:
            self.status.setText("仿真已取消")
            return
        if error:
            self.status.setText("仿真失败："+error)
            QMessageBox.warning(self, "仿真失败", error)
            return
        self.outputs = outputs
        self.observations, self.observation_errors = observations, observation_errors
        preferred = self.run_observation or next(reversed(observations), None)
        self.refresh_observation_points(preferred)
        self.status.setText(f"完成 · {len(observations)} 个业务输出"+
                            (f"，{len(observation_errors)} 个提取失败，请切换观察点查看原因" if observation_errors else ""))
        self.draw_output()

    def observed_wave(self):
        key = self.point.currentData()
        if key in self.observation_errors:
            return None
        if key not in self.observations and key in self.outputs:
            from link_sim.algorithm.framed_link import business_output
            node = self.graph.get_node_by_id(key)
            if node is not None and node.KIND in OBSERVATION_KINDS:
                try:
                    self.observations[key] = business_output(self.outputs[key], node.KIND)
                except (ValueError, RuntimeError) as exc:
                    self.observation_errors[key] = str(exc)
        return self.observations.get(key)

    def draw_output(self, *_):
        self.point.setToolTip(self.point.currentText())
        wave = self.observed_wave()
        self.export_button.setEnabled(wave is not None and self.worker is None)
        if wave is None:
            self.figure.clear()
            self.canvas.draw_idle()
            error = self.observation_errors.get(self.point.currentData())
            self.summary.setText("业务信号提取失败："+error if error else "该输出尚未计算，可点击“计算所选输出”查看和导出业务数据")
            self.note.setText("")
            self.spectrum_info.clear()
            return
        self.figure.clear()
        x = wave.valid
        view = self.plot_view.currentData()
        plots = ("time", "spectrum", "iq") if view == "all" else (view,)
        axes = self.figure.subplots(1, len(plots), squeeze=False).ravel()
        self.spectrum_info.clear()
        for kind, ax in zip(plots, axes):
            if kind == "time":
                n = min(1024, len(x))
                t = (np.arange(n)+wave.start)/wave.fs*1e6
                ax.plot(t, x[:n].real, color="#47d7ee", linewidth=.8, label="I")
                ax.plot(t, x[:n].imag, color="#ffbb68", linewidth=.8, label="Q")
                ax.set(title="业务信号时域", xlabel="业务时间 (μs)", ylabel="归一化幅度")
                ax.legend(loc="upper right", fontsize=8)
            elif kind == "spectrum":
                frequency, level = amplitude_spectrum(x, wave.fs)
                ax.plot(frequency/1e6, level, color="#47d7ee", linewidth=.8)
                ax.set(title="业务信号双边幅度频谱", xlabel="基带频率 (MHz)", ylabel="幅度 (dB，参考幅度 1)")
                ax.set_xlim(-wave.fs/2e6, wave.fs/2e6)
                self.spectrum_info.setText(f"频谱频点间隔：{wave.fs/len(x)/1e3:.6g} kHz")
            else:
                scatter = x
                stride = max(1, len(scatter)//4000)
                ax.scatter(scatter[::stride].real, scatter[::stride].imag, s=3, c="#5ee0b2", alpha=.45)
                ax.set(title="业务符号星座", xlabel="I", ylabel="Q")
                ax.set_aspect("equal", adjustable="datalim")
            ax.set_facecolor("#091b29")
            ax.tick_params(colors="#92b4c8", labelsize=8)
            ax.xaxis.label.set_color("#a6c9db")
            ax.yaxis.label.set_color("#a6c9db")
            ax.title.set_color("#d2eef7")
            for spine in ax.spines.values():
                spine.set_color("#28495f")
            ax.grid(alpha=.15, color="#6292a7")
        self.figure.tight_layout(pad=1.8)
        self.canvas.draw_idle()
        stats = signal_statistics(x)
        self.summary.setText(f"{wave.fs/1e6:g} MHz · {len(x):,} 点 · 平均功率 {stats['mean_power']:.4g}（归一化） · "
                             f"RMS {stats['rms']:.4g} · PAPR {stats['papr_db']:.2f} dB")
        if wave.context is not None:
            self.summary.setText(self.summary.text()+f" · 原始业务 {len(wave.context.bits):,} bit")
        self.note.setText(f"有效样本 [{wave.start}, {wave.stop})。"+"；".join(wave.notes))

    def export_selected(self):
        wave = self.observed_wave()
        if wave is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出模块输出", "waveform.npz", "I/Q 波形 (*.npz)")
        if path:
            try:
                export_wave(path if path.lower().endswith(".npz") else path+".npz", wave)
                self.status.setText("已导出 I/Q、频谱、采样率和有效样本区间")
            except Exception as exc:
                QMessageBox.warning(self, "导出失败", str(exc))

    def save_scheme(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存全链路方案", "transceiver.chain.json", "链路方案 (*.json)")
        if not path:
            return
        try:
            stages, edges = self.snapshot()
            positions = {n.id: n.pos() for n in self.graph.all_nodes()}
            save_chain(path if path.lower().endswith(".json") else path+".json", stages, edges,
                       positions)
            self.status.setText("方案已保存，包含模型系数快照和导入 I/Q，可独立重新打开")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def open_scheme(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开全链路方案", "", "链路方案 (*.json)")
        if not path:
            return
        try:
            stages, edges, positions = load_chain(path)
            self.rebuilding = True
            self.graph.clear_session()
            nodes = {s.key: self.create_stage(s.kind, positions[s.key], s) for s in stages}
            self.remap_pairs(nodes)
            for edge in edges:
                src, dst, src_port, dst_port = edge_parts(edge)
                nodes[src].output(src_port).connect_to(nodes[dst].input(dst_port), push_undo=False)
            self.link_mode.blockSignals(True)
            self.link_mode.setCurrentIndex(int(sum(s.kind in SOURCE_KINDS for s in stages) == 2))
            self.link_mode.blockSignals(False)
            self.status.setText("方案已加载，请运行链路")
        except Exception as exc:
            QMessageBox.warning(self, "打开方案失败", str(exc))
        finally:
            self.rebuilding = False
            self.invalidate()
            self.fit_canvas()

    def closeEvent(self, event):
        if self.worker:
            self.worker.cancel.set()
        super().closeEvent(event)
