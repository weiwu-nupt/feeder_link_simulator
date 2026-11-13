"""Finite complex-envelope flowgraphs, independent of Qt and the model editors.

Samples use normalized I/Q units; AD/DA errors come from the original model output.
Displayed/exported frequencies are complex baseband. Frequency-offset blocks
shift sampled spectra within the complex envelope.
"""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import numpy as np
from scipy import signal

from .device_models import DeviceModel, manifest, prepare_model_for_storage, unpack_manifest, to_pa
from .pa_simulation_core import load_waveform, positive_rate, waveform, simulate
from .adc_noise_core import calibrate_model_noise, generate_model_noise
from .mixer_noise_core import generate_phase_noise
from .pa_dpd_core import signature


MAX_SAMPLES = 262144
KINDS = {"source": "基带信号源", "dpd": "数字预失真（DPD）", "dac": "AD/DA", "filter": "滤波器",
         "mixer": "上变频器", "pa": "功率放大器", "gain": "增益",
         "frequency": "频率偏移", "awgn": "加性白噪声"}
KINDS.update(noise="底噪模型", frame_source="按帧发射信号", rrc_tx="发射 RRC", channel="无线信道",
             xpic="交叉极化信道（XPIC）", dual_channel="无线信道（双路）",
             fastica="FastICA 双路分离", downmixer="下变频器",
             rx_adc="接收 ADC", rrc_rx="接收 RRC", sync="同步模块", sink="接收信号")
KINDS.update(precomp="预补偿", lowpass="低通滤波")
DEVICE_KINDS = {"noise": "noise","dpd": "dpd", "dac": "adda", "rx_adc": "adda", "filter": "filter",
                "mixer": "mixer", "downmixer": "mixer", "pa": "pa", "rrc_tx": "filter", "rrc_rx": "filter",
                "precomp": "filter", "lowpass": "filter"}
DUAL_KINDS = ("xpic", "dual_channel", "fastica")
AWGN_CHANNEL_KINDS = ("channel", "dual_channel")
SOURCE_KINDS = ("source", "frame_source")
OBSERVATION_KINDS = (*SOURCE_KINDS, "pa", "sink")


@dataclass
class Wave:
    x: np.ndarray
    fs: float
    carrier_hz: float = 0.
    start: int = 0
    stop: int | None = None
    notes: tuple = ()
    added_error: np.ndarray | None = None
    context: object | None = None
    decoded_bits: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = waveform(self.x)
        self.fs = positive_rate(self.fs)
        if len(self.x) > MAX_SAMPLES:
            raise ValueError(f"单次波形最多 {MAX_SAMPLES:,} 点")
        self.stop = len(self.x) if self.stop is None else self.stop
        if not 0 <= self.start < self.stop <= len(self.x):
            raise ValueError("有效波形区间无效")
        if not np.isfinite(self.carrier_hz) or self.carrier_hz < 0:
            raise ValueError("载频须为有限非负数")
        if self.added_error is not None:
            self.added_error = waveform(self.added_error)
            if len(self.added_error) != len(self.x):
                raise ValueError("本级等效误差与输出长度不一致")

    @property
    def valid(self):
        return self.x[self.start:self.stop]


@dataclass
class Stage:
    key: str
    kind: str
    name: str
    settings: dict = field(default_factory=dict)
    model: DeviceModel | None = None
    enabled: bool = True


def _number(settings, key, default, low=None, high=None, integer=False):
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
        raise ValueError(f"{key} 须为有限数值")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{key} 超出允许范围 {low}–{high}")
    if integer and value != int(value):
        raise ValueError(f"{key} 须为整数")
    return int(value) if integer else float(value)


def default_settings(kind):
    if kind in DEVICE_KINDS:
        return {}
    if kind in AWGN_CHANNEL_KINDS:
        return dict(snr_db=35.)
    return {
        "source": dict(fs=100e6, count=16384, sps=8, amplitude=.2, seed=2025, modulation="QPSK"),
        "frame_source": dict(symbol_rate=700e6, seed=0, lane=0, segments=4,
                             payload_symbols=1024, header_hex="1ACFFC1D"),
        "rrc_tx": {}, "rrc_rx": {}, "sink": {},
        "sync": dict(threshold=.4),
        "xpic": dict(snr_db=35., xpd_a=10., xpd_b=10., delay_a=0., delay_b=0.),
        "fastica": dict(window_offset=4500, window_length=3000),
        "gain": dict(gain_db=0.),
        "frequency": dict(offset_hz=0.),
        "awgn": dict(snr_db=30., seed=2025),
    }[kind].copy()


def default_link(dual=False):
    stages, edges, positions = [], [], {}
    tx = ("frame_source", "rrc_tx", "dpd", "tx_precomp", "dac", "dac_lowpass", "mixer", "pa")
    if not dual:
        tx += ("channel",)
    rx = ("downmixer", "rx_precomp", "rx_adc", "adc_lowpass", "rrc_rx", "sync", "sink")
    special = dict(tx_precomp=("precomp", "发射预补偿"), dac_lowpass=("lowpass", "低通滤波"),
                   rx_precomp=("precomp", "接收预补偿"), adc_lowpass=("lowpass", "低通滤波"))
    lanes = ("A", "B") if dual else ("A",)
    for lane_index, lane in enumerate(lanes):
        previous = None
        for column, suffix in enumerate(tx + rx):
            kind, label = special.get(suffix, (suffix, "发射 DAC" if suffix == "dac" else KINDS.get(suffix)))
            key = f"{lane}_{suffix}"
            settings = default_settings(kind)
            if kind == "frame_source":
                settings.update(lane=lane_index)
            paired = dict(tx_precomp="dac_lowpass", dac_lowpass="tx_precomp", rx_precomp="adc_lowpass", adc_lowpass="rx_precomp")
            if suffix in paired:
                settings["paired_node"] = f"{lane}_{paired[suffix]}"
            stages.append(Stage(key, kind, f"{lane} · {label}" if dual else label, settings))
            if dual and column >= len(tx):
                column += len(DUAL_KINDS)
            positions[key] = [column*180, lane_index*300]
            if previous and not (dual and kind == "downmixer"):
                edges.append((previous, key))
            previous = key
    if dual:
        for column, kind in enumerate(DUAL_KINDS, start=len(tx)):
            settings = default_settings(kind)
            if kind == "xpic":
                settings["snr_db"] = 200.
            label = KINDS["channel"] if kind == "dual_channel" else KINDS[kind]
            stages.append(Stage(kind, kind, label, settings))
            positions[kind] = [column*180, 150]
        for port, lane in enumerate(lanes):
            edges.extend(((f"{lane}_pa", "xpic", 0, port),
                          ("xpic", "dual_channel", port, port),
                          ("dual_channel", "fastica", port, port),
                          ("fastica", f"{lane}_downmixer", port, 0)))
    return stages, edges, positions


def source_wave(settings):
    fs = _number(settings, "fs", 100e6, 1., 1e12)
    if "iq" in settings:
        a = np.asarray(settings["iq"], dtype=float)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("导入波形须为 I/Q 两列")
        return Wave(a[:, 0] + 1j*a[:, 1], fs, notes=("外部 I/Q：保留原始幅度；编码/调制由文件提供",))
    count = _number(settings, "count", 16384, 256, MAX_SAMPLES, True)
    sps = _number(settings, "sps", 8, 2, 256, True)
    amplitude = _number(settings, "amplitude", .2, 1e-6, 1.)
    seed = _number(settings, "seed", 2025, 0, 2**32-1, True)
    mode = settings.get("modulation", "QPSK")
    rng = np.random.default_rng(seed)
    n = (count+sps-1)//sps
    if mode == "BPSK":
        symbols = 2*rng.integers(0, 2, n)-1
    elif mode == "QPSK":
        symbols = ((2*rng.integers(0, 2, n)-1) + 1j*(2*rng.integers(0, 2, n)-1))/np.sqrt(2)
    elif mode == "16QAM":
        symbols = ((2*rng.integers(0, 4, n)-3) + 1j*(2*rng.integers(0, 4, n)-3))/np.sqrt(10)
    else:
        raise ValueError("支持 BPSK、QPSK 和 16QAM")
    return Wave(amplitude*np.repeat(symbols, sps)[:count], fs,
                notes=(f"{mode} 未编码测试信号，矩形脉冲，{sps} 点/符号",))


def import_source(path, fs):
    x, file_rate = load_waveform(path)
    if len(x) > MAX_SAMPLES:
        raise ValueError(f"导入波形最多 {MAX_SAMPLES:,} 点，请先分段")
    return dict(fs=file_rate or positive_rate(fs), iq=np.column_stack((x.real, x.imag)).tolist(),
                filename=Path(path).name)


def edge_parts(edge):
    if not isinstance(edge, (list, tuple)):
        raise ValueError("连线格式无效")
    if len(edge) == 2:
        return edge[0], edge[1], 0, 0
    if len(edge) != 4 or any(type(p) is not int or p < 0 for p in edge[2:]):
        raise ValueError("连线端口无效")
    return tuple(edge)


def output_key(key, port=0):
    return key if port == 0 else key + "::B"


def upstream_graph(node_keys, edges, target):
    """Return only the target and its prerequisites, ignoring downstream wiring."""
    keys = set(node_keys)
    if target not in keys:
        raise ValueError("请选择流程图中已有的输出模块")
    incoming = {}
    for edge in edges:
        if isinstance(edge, (tuple, list)) and len(edge) >= 2:
            incoming.setdefault(edge[1], []).append(edge)
    required, selected_edges, queue = set(), [], [target]
    while queue:
        key = queue.pop()
        if key in required:
            continue
        if key not in keys:
            raise ValueError("所选模块的上游连线指向不存在的模块")
        required.add(key)
        for edge in incoming.get(key, ()):
            src, _, _, _ = edge_parts(edge)
            selected_edges.append(edge)
            queue.append(src)
    return required, selected_edges


def order_graph(stages, edges, *, target=None):
    """Validate one/two-source DAGs with explicit ports for joint polarization nodes."""
    if target is not None:
        required, edges = upstream_graph([stage.key for stage in stages], edges, target)
        stages = [stage for stage in stages if stage.key in required]
    by_id = {stage.key: stage for stage in stages}
    if not by_id or len(by_id) != len(stages) or len(stages) > 64:
        raise ValueError("流程图须包含 1–64 个模块，且模块编号不能重复")
    if any(s.kind not in KINDS for s in stages):
        raise ValueError("流程图包含未知模块")
    sources = [s.key for s in stages if s.kind in SOURCE_KINDS]
    if not 1 <= len(sources) <= 2:
        raise ValueError("请保留一个或两个发射信号源")
    parents, children = {key: {} for key in by_id}, {key: [] for key in by_id}
    for edge in edges:
        src, dst, src_port, dst_port = edge_parts(edge)
        if (src not in by_id or dst not in by_id or dst in sources or dst_port in parents[dst]
                or src_port >= (2 if by_id[src].kind in DUAL_KINDS else 1)
                or dst_port >= (2 if by_id[dst].kind in DUAL_KINDS else 1)):
            raise ValueError("连线无效：请检查输入、输出端口与重复连接")
        parents[dst][dst_port] = (src, src_port)
        children[src].append(dst)
    for stage in stages:
        expected = 0 if stage.kind in SOURCE_KINDS else 2 if stage.kind in DUAL_KINDS else 1
        if len(parents[stage.key]) != expected:
            raise ValueError(f"{stage.name} 需要 {expected} 路输入，请完成连线")
    remaining = {key: len(p) for key, p in parents.items()}
    queue, ordered = list(sources), []
    while queue:
        key = queue.pop(0)
        if key in ordered:
            raise ValueError("流程图不能形成环路")
        ordered.append(key)
        for child in children[key]:
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    if len(ordered) != len(stages):
        raise ValueError("存在未连接的模块或环路，请将所有模块连接到信号源")
    return [by_id[key] for key in ordered], parents


def _check_model(stage, fs):
    if stage.model is None:
        return None
    model = stage.model.validated()
    if model.kind != DEVICE_KINDS.get(stage.kind):
        raise ValueError("所选模型的器件类型不匹配")
    # ADC and mixer noise use spectral densities on the input waveform's sampling grid.
    if model.kind == "filter":
        rate = model.parameters["fs"]
        if not np.isclose(fs, rate, rtol=1e-9, atol=1e-6):
            raise ValueError(f"模型采样率为 {rate/1e6:g} MHz，输入为 {fs/1e6:g} MHz；请修改信号源或更换模型")
    return model


def process_stage(stage, incoming, *, noise_stream=0, oscillator_stream=0):
    if not stage.enabled:
        return Wave(incoming.x.copy(), incoming.fs, incoming.carrier_hz, incoming.start,
                    incoming.stop, ("已旁路，直接传递输入 I/Q",), context=incoming.context, metrics=incoming.metrics)
    if stage.kind in ("rrc_tx", "rrc_rx", "sync", "sink", "channel"):
        from .framed_link import process_framed_stage
        return process_framed_stage(stage, incoming)
    model = _check_model(stage, incoming.fs)
    if stage.kind in DEVICE_KINDS and model is None:
        return Wave(incoming.x.copy(), incoming.fs, incoming.carrier_hz, incoming.start,
                    incoming.stop, ("未加载器件模型，当前为理想直通；双击从模型库加载",), context=incoming.context)

    p = stage.settings
    x, fs = incoming.x, incoming.fs
    start, stop = incoming.start, incoming.stop
    carrier = incoming.carrier_hz
    notes = []
    added_error = None
    if stage.kind == "precomp":
        from .filter_inverse import inverse_taps
        h, delay = inverse_taps(model)
        y = np.convolve(x, h)
        start += len(h)-1
        stop += len(h)-1
        notes.append(f"对应低通：{model.name}；通带逆响应时域卷积，{len(h)} 抽头；"
                     f"目标级联延迟 {delay} 点，正则化 1e-6，最大逆增益 20 dB；深阻带不可恢复")
    elif stage.kind in ("filter", "lowpass"):
        if stage.kind == "lowpass":
            from .filter_inverse import validate_lowpass
            validate_lowpass(model)
        b = model.arrays.get("b")
        a = model.arrays.get("a", np.array([1.]))
        sos = model.arrays.get("sos")
        notes.append(f"模型：{model.name}")
        fir = sos is None and len(a) == 1
        if fir:
            h = model.arrays.get("h")
            if h is None:
                h = b / a[0]
            y = np.convolve(x, h)
            if stage.kind == "filter":
                y = y[:len(x)]
            else:
                stop += len(h)-1
            start += len(h)-1
            notes.append(f"FIR 时域卷积；起始 {len(h)-1} 点为启动瞬态；"+
                         ("保留完整卷积尾部" if stage.kind == "lowpass" else "尾部未补零冲洗"))
        else:
            # The time-domain recurrence accounts for the entire IIR impulse response.
            if stage.kind == "lowpass":
                preview = prepare_model_for_storage(model).arrays["h"]
                flush = len(preview)-1
                x = np.pad(x, (0, flush))
                stop += flush
            y = signal.sosfilt(sos / sos[:, 3:4], x) if sos is not None else signal.lfilter(b, a, x)
            notes.append("IIR 时域递推；输出包含启动瞬态，未自动判定稳定时间")
    elif stage.kind in ("dac", "rx_adc"):
        # Keep the historical node key 'dac' so saved chain diagrams still load.
        seed = 2025
        if noise_stream:
            seed = int(np.random.SeedSequence([seed, noise_stream]).generate_state(1)[0])
        reference = calibrate_model_noise(model)
        noise, power = generate_model_noise(reference, len(x), fs, seed=seed)
        y = x+noise
        added_error = noise
        measured = float(np.mean(abs(noise[start:stop])**2))
        notes.append(f"x=s+n；{model.name or model.parameters['device_profile']}；"
                     f"模型估计噪声基底 {reference.floor_dbfs:.2f} dBFS/频点；"
                     f"Pn={power:.6g}；理论噪声 RMS {np.sqrt(power):.4g}；"
                     f"实测 RMS {np.sqrt(measured):.4g}；自动匹配链路采样率，不叠加杂散")
    elif stage.kind in ("mixer", "downmixer"):
        seed = 2025 if oscillator_stream == 0 else int(np.random.SeedSequence([2025, oscillator_stream]).generate_state(1)[0])
        phi, _, _, _ = generate_phase_noise(fs, len(x), model.parameters["anchors"], seed=seed)
        y = x*np.exp((1j if stage.kind == "mixer" else -1j)*phi)
        notes.append(f"{model.name or '混频器模型'}；按链路 {fs/1e6:g} MHz、{len(x):,} 点生成本振相噪；"
                     f"频点间隔 {fs/len(x):.4g} Hz；相噪 RMS {np.std(phi[start:stop]):.5g} rad；时域逐点旋转")
    elif stage.kind in ("pa", "dpd"):
        pa = to_pa(model)
        _, y, valid = simulate(pa, x, fs)
        start += valid.start
        stop -= len(x) - valid.stop
        notes.append(f"{model.name}；按保存系数直接计算，记忆延迟以当前信号采样点为单位；保留原始幅度")
    elif stage.kind == "noise":
        from .noise_core import generate_noise
        added_error = generate_noise(len(x), fs, model.parameters["density_dbfs_hz"], model.parameters["seed"] + noise_stream)
        y = x + added_error
        notes.append(f"底噪密度 {model.parameters['density_dbfs_hz']:g} dBFS/Hz；按输入采样率生成复高斯白噪声")
    elif stage.kind == "gain":
        gain = 10**(_number(p, "gain_db", 0., -100., 100.)/20)
        y = x*gain
    elif stage.kind == "frequency":
        offset = _number(p, "offset_hz", 0., -fs/2, fs/2)
        y = x*np.exp(2j*np.pi*offset*np.arange(len(x))/fs)
        notes.append("频偏移动采样频谱，请确保信号带宽加频偏未超过 Nyquist")
    elif stage.kind == "awgn":
        snr = _number(p, "snr_db", 30., -40., 200.)
        seed = _number(p, "seed", 2025, 0, 2**32-1, True)
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(np.mean(abs(incoming.valid)**2)*10**(-snr/10)/2)
        y = x+sigma*(rng.standard_normal(len(x))+1j*rng.standard_normal(len(x)))
        notes.append(f"输入有效区间平均功率定义 SNR={snr:g} dB")
    else:
        raise ValueError("未知处理模块")
    if stop-start < 4:
        raise ValueError("波形过短，器件记忆/滤波器启动瞬态后不足 4 个有效点")
    return Wave(y, fs, carrier, start, stop, tuple(notes), added_error, context=incoming.context)


def run_graph(stages, edges, *, target=None, progress=None, cancelled=None):
    ordered, parents = order_graph(stages, edges, target=target)
    by_key = {s.key: s for s in stages}
    outputs, pending_dpd = {}, {}
    # Stable across full and partial runs, including B-only previews.
    noise_streams = {stage.key: i for i, stage in enumerate(s for s in stages if s.kind in ("dac", "rx_adc", "noise"))}
    oscillator_streams = {stage.key: i for i, stage in enumerate(s for s in stages if s.kind in ("mixer", "downmixer"))}
    for index, stage in enumerate(ordered):
        if cancelled and cancelled():
            raise InterruptedError("仿真已取消")
        try:
            if stage.enabled and stage.kind in ("precomp", "lowpass"):
                mate = by_key.get(stage.settings.get("paired_node"))
                if mate is not None and mate.kind in ("precomp", "lowpass"):
                    if stage.model is None and mate.model is not None:
                        stage = replace(stage, model=mate.model)
                    elif stage.model is not None and mate.model is not None:
                        from .filter_inverse import filter_fingerprint
                        if filter_fingerprint(stage.model) != filter_fingerprint(mate.model):
                            raise ValueError("预补偿与配对低通须使用同一个低通模型")
            if stage.kind in SOURCE_KINDS:
                if stage.kind == "frame_source":
                    from .framed_link import frame_source
                    wave = frame_source(stage.settings)
                else:
                    wave = source_wave(stage.settings)
                if not stage.enabled:
                    wave = Wave(np.zeros_like(wave.x), wave.fs, notes=("信号源已关闭，输出零 I/Q",), context=wave.context)
                outputs[stage.key] = wave
            elif stage.kind in DUAL_KINDS:
                from .framed_link import process_dual_stage
                incoming = [outputs[output_key(*parents[stage.key][p])] for p in range(2)]
                values = process_dual_stage(stage, incoming)
                for port, value in enumerate(values):
                    outputs[output_key(stage.key, port)] = value
                    pending_dpd[output_key(stage.key, port)] = pending_dpd.get(output_key(*parents[stage.key][port]))
            else:
                parent, port = parents[stage.key][0]
                dpd = pending_dpd.get(output_key(parent, port))
                if stage.enabled and stage.model is not None:
                    if stage.kind == "dpd":
                        if dpd is not None:
                            raise ValueError("同一路径上的两级数字预失真之间须有已启用的配对功放")
                        dpd = stage.model
                    elif stage.kind == "pa" and dpd is not None:
                        if dpd.parameters["fit_info"]["paSignature"] != signature(to_pa(stage.model)):
                            raise ValueError(f"与上游数字预失真「{dpd.name or '未命名 DPD'}」不配对，请更换模型或旁路 DPD")
                        dpd = None
                pending_dpd[stage.key] = dpd
                outputs[stage.key] = process_stage(stage, outputs[output_key(parent, port)],
                                                  noise_stream=noise_streams.get(stage.key, 0),
                                                  oscillator_stream=oscillator_streams.get(stage.key, 0))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            raise ValueError(f"{stage.name}：{exc}") from exc
        if progress:
            progress(index+1, len(ordered), stage.name)
    return outputs


def model_to_dict(model):
    if model is None:
        return None
    model = prepare_model_for_storage(model)
    data = manifest(model)
    data["arrays"] = {k: dict(real=a.real.tolist(), imag=a.imag.tolist()) for k, a in model.arrays.items()}
    return data


def model_from_dict(data):
    if data is None:
        return None
    arrays = {}
    for key, values in data["arrays"].items():
        real, imag = np.asarray(values["real"]), np.asarray(values["imag"])
        if data.get("kind") in ("pa", "dpd") or np.any(imag):
            # Keep signed zero in PA coefficients so the training fingerprint round-trips.
            array = np.empty(real.shape, dtype=np.complex128)
            array.real, array.imag = real, imag
            arrays[key] = array
        else:
            arrays[key] = real
    return unpack_manifest(data, arrays)


def save_chain(path, stages, edges, positions):
    order_graph(stages, edges)
    def portable_settings(stage):
        if stage.kind in AWGN_CHANNEL_KINDS:
            return dict(snr_db=stage.settings.get("snr_db", 35.))
        if stage.kind != "frame_source":
            return stage.settings
        from .frame_config import frame_plan, portable_code, legacy_frame
        settings = dict(stage.settings)
        settings.pop("amplitude", None)
        rows = frame_plan(settings)
        for group in rows:
            for item in group:
                item["code"] = portable_code(item["code"])
        if legacy_frame(settings):
            settings["business_frames"] = rows
        else:
            settings.pop("business_frames", None)
            settings.pop("words", None)
            settings["business_frame"] = rows[0][0]
        return settings
    payload = dict(format="feeder-waveform-chain", version=1, edges=edges,
                   nodes=[dict(key=s.key, kind=s.kind, name=s.name, settings=portable_settings(s),
                               model=model_to_dict(s.model), enabled=s.enabled,
                               pos=positions.get(s.key, [0, 0])) for s in stages])
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def load_chain(path):
    path = Path(path)
    if path.stat().st_size > 128*1024**2:
        raise ValueError("链路方案超过 128 MiB")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("format") != "feeder-waveform-chain" or data.get("version") != 1:
        raise ValueError("不是受支持的全链路波形方案")
    if not isinstance(data.get("nodes"), list) or not 1 <= len(data["nodes"]) <= 64:
        raise ValueError("模块数量须为 1–64")
    stages, positions = [], {}
    for n in data["nodes"]:
        if not all(isinstance(n.get(k), str) for k in ("key", "kind", "name")) or not isinstance(n.get("settings"), dict):
            raise ValueError("模块说明无效")
        pos = np.asarray(n.get("pos", [0, 0]), dtype=float)
        if pos.shape != (2,) or not np.all(np.isfinite(pos)):
            raise ValueError("画布位置无效")
        enabled = n.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("模块开关状态无效")
        stage = Stage(n["key"], n["kind"], n["name"], n["settings"], model_from_dict(n.get("model")), enabled)
        if stage.kind in AWGN_CHANNEL_KINDS:
            stage.settings = dict(snr_db=stage.settings.get("snr_db", 35.))
        if stage.kind == "source":
            source_wave(stage.settings)
        elif stage.model and stage.model.kind != DEVICE_KINDS.get(stage.kind):
            raise ValueError("器件模型类型与模块不匹配")
        stages.append(stage)
        positions[stage.key] = pos.tolist()
    order_graph(stages, data["edges"])
    # Move legacy source-owned shaping parameters into the independent Tx RRC.
    by_key = {s.key: s for s in stages}
    for edge in data["edges"]:
        src, dst, _, _ = edge_parts(edge)
        source, target = by_key[src], by_key[dst]
        if source.kind == "frame_source" and target.kind == "rrc_tx" and "fs" in source.settings:
            for key in ("sps", "rolloff", "span"):
                target.settings.setdefault(key, source.settings.get(key, {"sps": 4, "rolloff": .25, "span": 8}[key]))
    for stage in stages:
        if stage.kind == "rrc_tx":
            if stage.model is None and any(key in stage.settings for key in ("sps", "rolloff", "span")):
                from .rrc_stage import legacy_rrc_model
                stage.model = legacy_rrc_model(stage.settings)
            stage.settings = {}
    # Historical execution-mode metadata is ignored; all chains run locally.
    return stages, data["edges"], positions


def export_wave(path, wave):
    """NPZ is compatible with the existing PA waveform importer (x/sampleRate)."""
    from .waveform_spectrum import amplitude_spectrum, signal_statistics
    frequency, level = amplitude_spectrum(wave.valid, wave.fs)
    np.savez_compressed(path, x=wave.x, sampleRate=wave.fs,
                        frequencyHz=frequency, spectrumDb=level,
                        validStart=wave.start, validStop=wave.stop, units="normalized IQ",
                        notes=np.asarray(wave.notes, dtype=str),
                        metrics=json.dumps(dict(wave.metrics, **signal_statistics(wave.valid))),
                       **({"transmittedBits": wave.context.bits} if wave.context is not None else {}),
                       **({"addedError": wave.added_error} if wave.added_error is not None else {}),
                       **({"receivedBits": wave.decoded_bits}
                          if wave.decoded_bits is not None else {}))
