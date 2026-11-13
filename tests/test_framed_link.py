"""End-to-end framed reception and two-polarization graph regressions."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import copy
import json

from link_sim.algorithm.waveform_chain import (
    Stage, default_link, export_wave, load_chain, order_graph, process_stage, run_graph, save_chain,
)
from link_sim.algorithm.framed_link import process_dual_stage
from link_sim.algorithm.framed_link import frame_source
from link_sim.algorithm.frame_config import default_business_frame, frame_summary
from waveform_fixtures import configured_link


class FramedLinkTests(unittest.TestCase):
    def test_three_observers_only_contain_business_symbols_and_pa_gain_is_preserved(self):
        from link_sim.algorithm.framed_link import business_output
        from link_sim.algorithm.device_models import from_pa
        from link_sim.algorithm.pa_simulation_core import PAModel
        stages, edges, _ = configured_link()
        gain = 1.7*np.exp(.4j)
        next(s for s in stages if s.kind == "pa").model = from_pa(
            PAModel([[gain]], "memPoly", 1, sample_rate=2.8e9), "复增益功放")
        next(s for s in stages if s.kind == "channel").settings["snr_db"] = 200.
        out = run_graph(stages, edges)
        raw_source = out["A_frame_source"]
        spans = [span for group in raw_source.context.framer.codeword_spans for span in group]
        expected = np.concatenate([raw_source.x[a:b] for a, b in spans])
        source = business_output(raw_source, "frame_source")
        pa = business_output(out["A_pa"], "pa")
        sink = business_output(out["A_sink"], "sink")
        self.assertEqual(len(raw_source.x), 5248)
        for wave in (source, pa, sink):
            self.assertEqual(len(wave.x), 2048)
            self.assertEqual(wave.fs, 700e6)
        np.testing.assert_array_equal(source.x, expected)
        # Matched RRC has finite-length ISI, but may not remove PA gain or phase.
        self.assertLess(np.linalg.norm(pa.x-gain*expected)/np.linalg.norm(gain*expected), .01)
        self.assertEqual(len(out), len(stages), "The observation branch must not add visible nodes")
        np.testing.assert_array_equal(sink.decoded_bits, raw_source.context.bits)
        np.testing.assert_allclose(sink.x, source.x, atol=.04)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"pa.npz"
            export_wave(path, pa)
            with np.load(path) as data:
                np.testing.assert_array_equal(data["x"], pa.x)
                self.assertEqual(len(data["spectrumDb"]), 2048)

    def test_source_rate_changes_use_pa_sample_delays_directly_without_changing_grid(self):
        from link_sim.algorithm.device_models import from_pa
        from link_sim.algorithm.pa_simulation_core import PAModel
        stages, edges, _ = configured_link()
        amplifier = next(s for s in stages if s.kind == "pa")
        pa = PAModel([[1], [.03]], "memPoly", 1, sample_rate=860.16e6)
        amplifier.model = from_pa(pa, "直接计算功放")
        for symbol_rate in (25e6, 100e6, 215.04e6, 700e6, 1e9):
            with self.subTest(symbol_rate=symbol_rate):
                stages[0].settings["symbol_rate"] = symbol_rate
                outputs = run_graph(stages, edges, target="A_pa")
                before, after = outputs["A_mixer"], outputs["A_pa"]
                self.assertEqual(after.fs, 4*symbol_rate)
                self.assertEqual(len(before.x), len(after.x))
                self.assertTrue(np.isfinite(after.x).all())
                np.testing.assert_array_equal(after.x, pa.predict(before.x))
                self.assertEqual(after.start-before.start, 1)
                self.assertIn("按保存系数直接计算", "；".join(after.notes))
                self.assertEqual(amplifier.model.parameters["sample_rate"], 860.16e6)

    def test_rrc_requires_a_loaded_model_and_source_can_run_independently(self):
        stages, edges, _ = default_link()
        self.assertEqual(set(run_graph(stages, edges, target="A_frame_source")), {"A_frame_source"})
        with self.assertRaisesRegex(ValueError, "加载 RRC"):
            run_graph(stages, edges, target="A_rrc_tx")

    def test_legacy_rrc_settings_migrate_to_saved_model(self):
        stages, edges, positions = default_link()
        next(s for s in stages if s.kind == "rrc_tx").settings.update(sps=8, rolloff=.35, span=10)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"legacy.json"
            save_chain(path, stages, edges, positions)
            loaded, edges, _ = load_chain(path)
        rrc = next(s for s in loaded if s.kind == "rrc_tx")
        self.assertEqual(rrc.settings, {})
        self.assertEqual(rrc.model.parameters["sps"], 8)
        self.assertEqual(rrc.model.parameters["beta"], .35)
        output = run_graph(loaded, edges, target="A_rrc_tx")["A_rrc_tx"]
        self.assertEqual(output.fs, 5.6e9)

    def test_joint_preview_requires_both_inputs_but_no_receiver_configuration(self):
        stages, edges, _ = configured_link(True)
        next(s for s in stages if s.kind == "fastica").settings["window_length"] = "invalid"
        edges = [e for e in edges if not (len(e) == 4 and e[1] == "fastica" and e[3] == 1)]
        outputs = run_graph(stages, edges, target="xpic")
        self.assertEqual(len(outputs), 18)
        self.assertIn("A_frame_source", outputs)
        self.assertIn("B_frame_source", outputs)
        self.assertIn("xpic::B", outputs)
        self.assertNotIn("dual_channel", outputs)
        self.assertNotIn("dual_channel::B", outputs)
        self.assertNotIn("fastica", outputs)
        with self.assertRaises(ValueError):
            run_graph(stages, [e for e in edges if not (len(e) == 4 and e[1] == "xpic" and e[3] == 1)], target="xpic")

    def test_rrc_uses_saved_filter_taps_at_source_symbol_rate(self):
        from link_sim.algorithm.device_models import DeviceModel, FILTER_DEFAULTS
        from link_sim.algorithm.burst_link.pulse_shaping import RRC
        source = frame_source({})
        taps = RRC(samples_per_symbol=4, rolloff=.25, span_symbols=8).coefficients().numpy()*1.5
        model = DeviceModel("filter", dict(FILTER_DEFAULTS, impl="RRC 成形", fs=100e6,
                                          sps=4, beta=.25, span=8), {"h": taps}).validated()
        stage = Stage("rrc", "rrc_tx", "独立 RRC", model=model)
        output = process_stage(stage, source)
        impulses = np.zeros((len(source.x)-1)*4+1, dtype=complex)
        impulses[::4] = source.x
        np.testing.assert_allclose(output.x, np.convolve(impulses, taps), atol=1e-14)
        self.assertEqual(output.fs, 2.8e9)
        self.assertIsNone(source.context.shaper)
        np.testing.assert_array_equal(output.context.shaper.coefficients().numpy(), taps)

    def test_source_mixed_business_frames_and_separate_rrc(self):
        from link_budget.algorithm.ber_core.catalog import session_catalog
        mods = {m["name"]: m for m in session_catalog()["modulations"]}
        frames = [default_business_frame() for _ in range(3)]
        for frame, modulation in zip(frames, ("BPSK", "QPSK", "16QAM")):
            frame["mod"] = mods[modulation]
        stages, edges, _ = configured_link()
        stages[0].settings = dict(symbol_rate=700e6, seed=2026, segments=1, words=3, business_frames=[frames])
        out = run_graph(stages, edges)
        source = out["A_frame_source"]
        self.assertEqual(source.fs, 700e6)
        self.assertEqual(len(source.x), 2880)
        self.assertIsNone(source.context.shaper, "Source must not design or apply RRC")
        self.assertEqual(source.metrics["information_bits"], 1536)
        self.assertEqual(source.metrics["coded_bits"], 3072)
        self.assertEqual([b-a for group in source.context.framer.codeword_spans for a, b in group], [1024, 512, 256])
        self.assertEqual(out["A_rrc_tx"].fs, 2.8e9)
        np.testing.assert_array_equal(out["A_sink"].decoded_bits, source.context.bits)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"source.npz"
            export_wave(path, source)
            with np.load(path) as data:
                np.testing.assert_array_equal(data["transmittedBits"], source.context.bits)
                self.assertEqual(json.loads(str(data["metrics"]))["information_bits"], 1536)

    def test_shared_catalog_codec_families_and_modulation_padding(self):
        import torch
        from link_budget.algorithm.ber_core.catalog import session_catalog, apsk_definition
        from link_budget.algorithm.ber_core.simulation import make_codec
        catalog = session_catalog()
        codes = [next(c for c in catalog["codes"] if c["family"] == f) for f in ("nr", "dvb", "qc", "none")]
        codes += [dict(id="h", name="测试 H", family="h", format="h", text="1 1 1 0\n0 1 1 1", k=2, n=4),
                  dict(id="conv", name="测试卷积码", family="conv", k=128, length=3, polys=["7", "5"], terminate=True),
                  dict(id="turbo", name="测试 Turbo", family="turbo", k=128, length=3, polys=["7", "5"],
                       terminate=True, rate="1/3", interleaver="qpp")]
        for code in codes:
            if "n" not in code and code["family"] in ("conv", "turbo"):
                _, _, _, code["n"], _ = make_codec(code)
        frames = []
        mods = catalog["modulations"] + [apsk_definition(16, [4, 12], [1., 2.85], [0., 15.])]
        for index, code in enumerate(codes):
            frames.append(dict(mod=mods[index % len(mods)], code=code))
        frames.append(dict(mod=mods[-1], code=codes[0]))
        settings = dict(segments=1, words=len(frames), business_frames=[frames])
        wave = frame_source(settings)
        decoded = wave.context.framer.decode(torch.as_tensor(wave.x)).numpy()
        np.testing.assert_array_equal(decoded, wave.context.bits)
        self.assertGreater(wave.metrics["modulation_padding_bits"], 0)
        self.assertEqual(wave.metrics["information_bits"], sum(f["information_bits"] for f in wave.metrics["frames"]))

    def test_saved_matrix_frame_is_independent_of_original_file(self):
        from link_budget.algorithm.ber_core.catalog import session_catalog
        code = copy.deepcopy(next(c for c in session_catalog()["codes"] if c["family"] == "qc"))
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder)/"matrix.base"
            original.write_text(Path(code["path"]).read_text(), encoding="utf-8")
            code["path"] = str(original)
            stages, edges, positions = configured_link()
            stages[0].settings.update(segments=1, payload_symbols=100000,
                                      business_frame=dict(mod=default_business_frame()["mod"], code=code))
            path = Path(folder)/"portable.json"
            save_chain(path, stages, edges, positions)
            original.unlink()
            loaded, _, _ = load_chain(path)
            self.assertNotIn("path", loaded[0].settings["business_frame"]["code"])
            actual = frame_source(loaded[0].settings)
            self.assertEqual(len(actual.context.bits), actual.metrics["information_bits"])

    def test_single_channel_recovers_bits_and_receiver_uses_no_payload_truth(self):
        stages, edges, _ = configured_link()
        out = run_graph(stages, edges)
        received = out["A_sink"]
        np.testing.assert_array_equal(received.decoded_bits, out["A_frame_source"].context.bits)
        self.assertLess(received.metrics["evm"], .08)
        synchronized = out["A_sync"]
        synchronized.context.bits = 1 - synchronized.context.bits
        second = process_stage(stages[-1], synchronized)
        np.testing.assert_array_equal(second.decoded_bits, received.decoded_bits)
        self.assertEqual(second.metrics["ber"], 1.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"received.npz"
            export_wave(path, received)
            with np.load(path) as data:
                np.testing.assert_array_equal(data["receivedBits"], received.decoded_bits)

    def test_awgn_uses_measured_power_and_preserves_sample_grid(self):
        from link_sim.algorithm.waveform_chain import Wave

        signal = 2.5 * np.exp(2j*np.pi*np.arange(65536)/32)
        inputs = [Wave(signal, 2.8e9, start=4, stop=len(signal)-4),
                  Wave(signal/5, 2.8e9)]
        settings = dict(snr_db=20., gain_db=-6., offset_hz=100., delay=10.)
        single = process_stage(Stage("single", "channel", "无线信道", settings), inputs[0])
        pair = process_dual_stage(Stage("dual", "dual_channel", "无线信道", settings), inputs)
        for source, output in zip([inputs[0], *inputs], [single, *pair]):
            noise = output.x - source.x
            measured_snr = 10*np.log10(np.mean(abs(source.x)**2)/np.mean(abs(noise)**2))
            self.assertAlmostEqual(measured_snr, 20., delta=.15)
            self.assertEqual((output.fs, output.start, output.stop), (source.fs, source.start, source.stop))
        noise_a, noise_b = (output.x - source.x for source, output in zip(inputs, pair))
        correlation = abs(np.vdot(noise_a, noise_b))/(np.linalg.norm(noise_a)*np.linalg.norm(noise_b))
        self.assertLess(correlation, .02)

    def test_dual_channel_ports_scheme_and_joint_bypass(self):
        stages, edges, positions = configured_link(True)
        next(s for s in stages if s.kind == "xpic").settings["delay_b"] = .5
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"dual.json"
            save_chain(path, stages, edges, positions)
            stages, edges, positions = load_chain(path)
        _, parents = order_graph(stages, edges)
        channel = "dual_channel"
        self.assertEqual([s.key for s in stages if s.kind in ("channel", "dual_channel")], [channel])
        for port, lane in enumerate(("A", "B")):
            with self.subTest(lane=lane):
                self.assertEqual(parents["xpic"][port], (f"{lane}_pa", 0))
                self.assertEqual(parents[channel][port], ("xpic", port))
                self.assertEqual(parents["fastica"][port], (channel, port))
                self.assertEqual(parents[f"{lane}_downmixer"][0], ("fastica", port))
                self.assertLess(positions["xpic"][0], positions[channel][0])
                self.assertLess(positions[channel][0], positions["fastica"][0])
        out = run_graph(stages, edges)
        for lane in ("A", "B"):
            with self.subTest(lane=lane):
                np.testing.assert_array_equal(out[lane+"_sink"].decoded_bits,
                                              out[lane+"_frame_source"].context.bits)
        self.assertFalse(np.array_equal(out["A_sink"].decoded_bits, out["B_sink"].decoded_bits))
        bypass = process_dual_stage(Stage("f", "fastica", "旁路", enabled=False),
                                    [out[channel], out[channel+"::B"]])
        for actual, key in zip(bypass, (channel, channel+"::B")):
            np.testing.assert_array_equal(actual.x, out[key].x)
        # Both input ports are required, and a two-port round-trip must preserve B.
        with self.assertRaises(ValueError):
            order_graph(stages, [e for e in edges if not (len(e) == 4 and e[1] == "fastica" and e[3] == 1)])
        with self.assertRaises((ValueError, RuntimeError)):
            process_dual_stage(Stage("f", "fastica", "退化混合"), [out["xpic"], out["xpic"]])


if __name__ == "__main__":
    unittest.main()
