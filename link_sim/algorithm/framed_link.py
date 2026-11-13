"""Adapters for the reference project's framed, complex-envelope receiver."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import max_len_seq
import secrets


@dataclass
class BurstContext:
    framer: object
    shaper: object
    bits: np.ndarray
    symbols: np.ndarray
    lane: int
    seed: int = 0


def frame_source(settings):
    import torch
    from .burst_link.modulation import create
    from .frame_config import frame_plan, frame_summary, legacy_frame, header_bits, payload_size
    from .mixed_burst import MixedBurstFramer
    from .waveform_chain import Wave, _number

    # Small burst kernels avoid CPU thread oversubscription in the GUI worker.
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    lane = _number(settings, "lane", 0, 0, 1, True)
    seed = _number(settings, "seed", 0, 0, 2**32-1, True)
    if seed == 0:
        seed = secrets.randbelow(2**32-1)+1
    sync = max_len_seq(7, taps=[6] if lane == 0 else [4], state=[1]*7)[0]
    header = (np.r_[max_len_seq(6, taps=[5] if lane == 0 else [1], state=[1]*6)[0], 0]
              if legacy_frame(settings) else header_bits(settings))
    bpsk = create("BPSK")
    sync_symbols = bpsk.modulate(torch.tensor(np.tile(np.r_[sync, 0], 8), dtype=torch.uint8))
    header_symbols = bpsk.modulate(torch.tensor(header, dtype=torch.uint8))
    from .waveform_chain import MAX_SAMPLES
    summary = frame_summary(settings)
    if summary["total_symbols"] > MAX_SAMPLES:
        raise ValueError("完整突发超过最大符号数，请减少每突发帧数或业务数据符号数")
    plan = frame_plan(settings)
    framer = MixedBurstFramer(sync_symbols, header_symbols, plan, payload_size(settings))
    bits = np.random.default_rng(seed).integers(0, 2, framer.capacity_bits, dtype=np.uint8)
    symbols = framer(torch.tensor(bits)).numpy()
    context = BurstContext(framer, None, bits, symbols, lane, seed)
    summary["actual_seed"] = seed
    return Wave(symbols, summary["symbol_rate"], context=context, metrics=summary,
                notes=(f"同步头 1024 符号；每突发 {len(plan)} 帧，每帧 {len(header_symbols)} 符号帧头；"
                       f"业务补零共 {summary['zero_padding_symbols']:,} 符号；实际随机种子 {seed}；"
                       f"原始业务 {len(bits):,} bit，编码后 {summary['coded_bits']:,} bit，"
                       f"总计 {len(symbols):,} 符号；码元速率 {summary['symbol_rate']/1e6:g} Mbaud；"
                       "采用单位平均能量星座，未额外缩放发射幅度",))


def tensor(x):
    import torch
    return torch.as_tensor(np.asarray(x), dtype=torch.complex128)


def business_output(wave, kind):
    """Three observation points expose payload symbols, never training headers.

    PA observation uses a hidden matched-filter and timing branch. It does not
    decode, normalize the PA gain, or change the signal passed down the graph.
    """
    from .waveform_chain import Wave
    c = wave.context
    if c is None:
        return wave
    if kind not in ("frame_source", "source", "pa", "sink"):
        raise ValueError("只支持发射信号、功放及接收信号业务观测")
    symbols, fs = wave.x, wave.fs
    notes = []
    if kind == "pa":
        from .burst_link.receive_filtering import RRCMatchedFilter
        from .burst_link.interpolation import FractionalInterpolator
        if c.shaper is None:
            raise ValueError("功放业务信号观测需要上游已加载的发射 RRC 模型")
        matched = RRCMatchedFilter(c.shaper)(tensor(wave.x)).numpy()
        start, _ = synchronize(Wave(matched, wave.fs, context=c), .25)
        symbols = FractionalInterpolator()(tensor(matched), start=start,
            step=c.shaper.samples_per_symbol, count=c.framer.total_symbols).numpy()
        fs /= c.shaper.samples_per_symbol
        notes.append("功放观测支路：内部接收 RRC 匹配滤波＋同步及小数定时插值；保留功放幅度和相位")
    if len(symbols) != c.framer.total_symbols:
        raise ValueError("业务观测需要完成符号同步，请检查接收 RRC 和同步模块是否已启用")
    spans = [span for group in c.framer.codeword_spans for span in group]
    data = np.concatenate([symbols[a:b] for a, b in spans])
    skipped = sum(b-a for a, b in c.framer.payload_spans)-len(data)
    notes.append(f"仅提取实际业务码字，去除同步头、所有帧头和 {skipped:,} 个帧尾补零符号；按发送顺序拼接")
    metrics = dict(wave.metrics, business_symbols=len(data), omitted_padding_symbols=skipped, actual_seed=c.seed)
    return Wave(data, fs, wave.carrier_hz, context=c, decoded_bits=wave.decoded_bits,
                metrics=metrics, notes=tuple(notes)+wave.notes)


def synchronize(wave, threshold):
    from .burst_link.synchronization import CorrelationSynchronizer
    if wave.context is None:
        raise ValueError("同步需要带有约定帧结构的信号源")
    c = wave.context
    if c.shaper is None:
        raise ValueError("同步前请连接并启用独立的发射 RRC 模块")
    sync = CorrelationSynchronizer(sync_symbols=c.framer.sync_symbols, shaper=c.shaper, threshold=threshold)
    result = sync(tensor(wave.x))
    if not result.success:
        raise ValueError(f"同步失败：{result.reason}；相关峰 {result.peak_value}")
    start = result.fractional_start_sample if result.fractional_valid else float(result.start_sample)
    return start, result


def process_framed_stage(stage, wave):
    from .waveform_chain import Wave, _number
    from .burst_link.receive_filtering import RRCMatchedFilter
    from .burst_link.interpolation import FractionalInterpolator

    c, p = wave.context, stage.settings
    if stage.kind in ("rrc_tx", "rrc_rx", "sync", "sink") and c is None:
        raise ValueError("请连接按帧生成的发射信号")
    if stage.kind == "rrc_tx":
        from .rrc_stage import transmit_shaper
        from .waveform_chain import MAX_SAMPLES
        shaper = transmit_shaper(stage)
        if shaper.output_length(len(wave.x)) > MAX_SAMPLES:
            raise ValueError("RRC 成形后超过最大采样点数，请减少业务帧数或每符号采样点")
        x = shaper(tensor(wave.x)).numpy()
        return Wave(x, wave.fs*shaper.samples_per_symbol, wave.carrier_hz,
                    context=replace(c, shaper=shaper), metrics=wave.metrics,
                    notes=(f"独立 RRC 时域成形：sps={shaper.samples_per_symbol}，α={shaper.rolloff:g}，"
                           f"span={shaper.span_symbols}；保留完整卷积尾部",))
    if stage.kind == "rrc_rx":
        if c.shaper is None:
            raise ValueError("接收 RRC 需要上游已启用的发射 RRC")
        if stage.model is not None:
            from .rrc_stage import SavedRRC
            rx = SavedRRC(stage.model)
            if (rx.samples_per_symbol != c.shaper.samples_per_symbol or rx.tap_count != c.shaper.tap_count
                    or not np.allclose(rx.coefficients().numpy(), c.shaper.coefficients().numpy())):
                raise ValueError("接收 RRC 模型应与发射 RRC 的抽头和每符号采样点一致")
        x = RRCMatchedFilter(c.shaper)(tensor(wave.x)).numpy()
        return Wave(x, wave.fs, wave.carrier_hz, context=c, notes=("接收 RRC 匹配滤波，保留完整卷积尾部",))
    if stage.kind == "sync":
        start, result = synchronize(wave, _number(p, "threshold", .4, .01, 1.))
        x = FractionalInterpolator()(tensor(wave.x), start=start, step=c.shaper.samples_per_symbol,
                                    count=c.framer.total_symbols).numpy()
        return Wave(x, wave.fs/c.shaper.samples_per_symbol, wave.carrier_hz, context=c,
                    metrics=dict(sync_sample=start, correlation=result.peak_value),
                    notes=(f"同步位置 {start:.5f} 采样；相关峰 {result.peak_value:.5f}；小数定时插值",))
    if stage.kind == "sink":
        bits = c.framer.decode(tensor(wave.x)).numpy()
        tracked = c.framer.last_phase_result.symbols.numpy()
        spans = [span for group in c.framer.codeword_spans for span in group]
        data = np.concatenate([tracked[a:b] for a, b in spans])
        reference = np.concatenate([c.symbols[a:b] for a, b in spans])
        errors = int(np.count_nonzero(bits != c.bits))
        evm = float(np.linalg.norm(data-reference)/np.linalg.norm(reference))
        return Wave(tracked, wave.fs, wave.carrier_hz, context=c, decoded_bits=bits,
                    metrics=dict(wave.metrics, bit_errors=errors, bit_count=len(bits), ber=errors/len(bits), evm=evm),
                    notes=(f"帧头相位/幅度恢复，各业务帧按各自调制编码软解调及译码；误码 {errors}/{len(bits)}，"
                           f"BER={errors/len(bits):.4g}；业务符号 EVM={100*evm:.3f}%",))
    if stage.kind == "channel":
        return wireless_channel(wave, p, seed=2026 + (c.lane if c is not None else 0))
    raise ValueError("未知收发模块")


def wireless_channel(wave, settings, *, seed):
    # MATLAB awgn(x, snr, 'measured') uses the mean power of the input samples.
    return replace(wave, x=add_noise(wave.x, settings, seed), added_error=None,
                   notes=("AWGN：按输入波形平均功率叠加复高斯白噪声",))


def add_noise(x, settings, seed):
    from .waveform_chain import _number
    snr = _number(settings, "snr_db", 35., -40., 200.)
    power = np.mean(abs(x)**2)*10**(-snr/10)
    rng = np.random.default_rng(seed)
    return x + np.sqrt(power/2)*(rng.standard_normal(len(x))+1j*rng.standard_normal(len(x)))


def process_dual_stage(stage, inputs):
    from .waveform_chain import Wave, _number
    from .burst_link.channel.dual_polarization import DualPolarizationChannel
    from .burst_link.receive_filtering import RRCMatchedFilter
    from .burst_link.fastica_separator import FastICAPolarizationSeparator
    a, b = inputs
    if not np.isclose(a.fs, b.fs) or len(a.x) != len(b.x):
        raise ValueError("双路输入必须等长、采样率相同并共享时间轴")
    if not stage.enabled:
        return [Wave(w.x.copy(), w.fs, w.carrier_hz, w.start, w.stop, ("双路模块已旁路",), context=w.context)
                for w in inputs]
    p = stage.settings
    if stage.kind == "dual_channel":
        return [wireless_channel(wave, p, seed=2026+port) for port, wave in enumerate(inputs)]
    if stage.kind == "xpic":
        channel = DualPolarizationChannel(sample_rate_hz=a.fs, filter_taps=[1.],
            xpd_db=(_number(p, "xpd_a", 10., 0., 100.), _number(p, "xpd_b", 10., 0., 100.)),
            cross_phase_rad=(.3, -.4),
            source_delay_samples=(_number(p, "delay_a", 0., 0., 4096.), _number(p, "delay_b", 0., 0., 4096.)))
        x, y = channel(tensor(a.x), tensor(b.x))
        return [Wave(add_noise(z.numpy(), p, 3030+i), w.fs, w.carrier_hz, context=w.context,
                     notes=("双极化信道：两发射源经 2×2 交叉耦合进入共同接收窗口",))
                for i, (z, w) in enumerate(zip((x, y), inputs))]
    if any(w.context is None for w in inputs):
        raise ValueError("FastICA 需要两路带有已知同步头的帧信号")
    starts = []
    for w in inputs:
        matched = RRCMatchedFilter(w.context.shaper)(tensor(w.x)).numpy()
        start, _ = synchronize(Wave(matched, w.fs, context=w.context), .25)
        starts.append(start-w.context.shaper.delay_samples)
    separator = FastICAPolarizationSeparator(
        window_offset_samples=_number(p, "window_offset", 4500, 0, 200000, True),
        window_length_samples=_number(p, "window_length", 3000, 128, 200000, True))
    training_mask = None
    if any(w.context.framer.payload_symbols is not None for w in inputs):
        # Use receiver-known layout and measured timing only; no transmitted bits.
        # The shared zero envelope must not become an ICA training component.
        joint = np.ones(len(a.x), dtype=bool)
        for w, start in zip(inputs, starts):
            mask = np.zeros(len(w.x), dtype=bool)
            shaper = w.context.shaper
            for group in w.context.framer.codeword_spans:
                for lo, hi in group:
                    first = max(0, int(np.ceil(start+lo*shaper.samples_per_symbol+shaper.tap_count)))
                    last = min(len(mask), int(np.floor(start+(hi-1)*shaper.samples_per_symbol-shaper.tap_count)))
                    mask[first:max(first, last)] = True
            joint &= mask
        import torch
        training_mask = torch.as_tensor(joint)
    result = separator(tensor(a.x), tensor(b.x), start_a=starts[0], start_b=starts[1], training_mask=training_mask)
    note = (f"复数 FastICA 联合分离；迭代 {result.fastica.iterations} 次；"
            f"训练窗口 [{result.fit_start}, {result.fit_end})；路序区分度 {result.coefficient_margin:.4f}")
    if training_mask is not None:
        note += f"；跳过帧头及补零，使用 {int(training_mask[result.fit_start:result.fit_end].sum())} 个有效业务采样"
    return [Wave(z.numpy(), w.fs, w.carrier_hz, context=w.context, notes=(note,))
            for z, w in zip((result.waveform_a, result.waveform_b), inputs)]
