"""Qt integration: actual worker, selection and embedded model configuration."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox, QPushButton, QStyle, QStyleOptionSpinBox, QToolButton

from link_sim.algorithm.device_models import from_pa
from link_sim.algorithm.pa_dpd import PADPDWidget
from link_sim.algorithm.pa_simulation_core import PAModel
from link_sim.algorithm.waveform_chain import Stage, Wave
from link_sim.algorithm.waveform_chain import default_settings
from link_sim.ui.waveform_workspace import StageDialog, WaveformWorkspace
from link_sim.ui.workspace import EMBEDDED_STYLE, LazyDialogPage
from waveform_fixtures import configure_workspace_rrc, rrc_model


class WaveformWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.workspace = WaveformWorkspace()
        configure_workspace_rrc(self.workspace)

    def tearDown(self):
        self.workspace.close()
        self.workspace.deleteLater()
        self.app.processEvents()

    def wait_for_worker(self, expected_outputs=16):
        deadline = time.monotonic()+15
        while self.workspace.worker and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(self.workspace.worker, "Background computation did not finish")
        self.assertEqual(len(self.workspace.outputs), expected_outputs)

    def test_source_dialog_first_show_fits_scaled_screen_and_keeps_actions_visible(self):
        for area, font_size in ((QRect(40, 30, 1280, 720), 10), (QRect(40, 30, 800, 520), 14)):
            with self.subTest(area=area, font_size=font_size):
                dialog = StageDialog(Stage("source", "frame_source", "发射信号", default_settings("frame_source")), self.workspace)
                dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
                dialog.setStyleSheet(EMBEDDED_STYLE)
                dialog.setFont(QFont("Microsoft YaHei", font_size))
                with patch.object(dialog, "available_area", return_value=area):
                    dialog.show()
                    self.app.processEvents()
                    try:
                        self.assertTrue(area.contains(dialog.frameGeometry()), str(dialog.frameGeometry()))
                        table = dialog.frame_editor.table
                        for column in (0, 1, 2):
                            self.assertLessEqual(table.cellWidget(0, column).width(), table.columnWidth(column))
                        actions = dialog.findChild(QDialogButtonBox)
                        self.assertIs(actions.parentWidget(), dialog)
                        self.assertTrue(dialog.rect().contains(actions.geometry()))
                        for button in actions.buttons():
                            bounds = QRect(button.mapTo(dialog, QPoint()), button.size())
                            self.assertTrue(dialog.rect().contains(bounds))
                        scroll = dialog.content_scroll
                        scroll.ensureWidgetVisible(dialog.frame_editor.summary, 0, 0)
                        self.app.processEvents()
                        bounds = QRect(dialog.frame_editor.summary.mapTo(scroll.viewport(), QPoint()),
                                       dialog.frame_editor.summary.size())
                        self.assertTrue(scroll.viewport().rect().contains(bounds), str(bounds))
                    finally:
                        dialog.close()
                        dialog.deleteLater()
                        self.app.processEvents()

    def test_payload_source_and_pa_preview_without_valid_receiver(self):
        w = self.workspace
        source = next(n for n in w.graph.all_nodes() if n.KIND == "frame_source")
        rrc = next(n for n in w.graph.all_nodes() if n.KIND == "rrc_tx")
        pa = next(n for n in w.graph.all_nodes() if n.KIND == "pa")
        # A broken receiver model and disconnected receiver must not be inspected.
        adc = next(n for n in w.graph.all_nodes() if n.KIND == "rx_adc")
        adc.set_property("device_model", {"kind": "adda"}, push_undo=False)
        w.graph.delete_nodes([n for n in w.graph.all_nodes() if n.KIND == "sync"])
        self.assertGreaterEqual(w.point.findData(source.id), 0)
        self.assertEqual(w.point.findData(rrc.id), -1)
        self.assertEqual(w.point.count(), 3)
        w.select_node(pa)
        with patch("link_sim.ui.waveform_workspace.QMessageBox.warning") as warning:
            w.preview_button.click()
            self.wait_for_worker(8)
            warning.assert_not_called()
        self.assertIn(source.id, w.outputs)
        self.assertEqual(w.point.currentData(), pa.id)
        with tempfile.TemporaryDirectory() as folder:
            for node, fs in ((source, 700e6), (pa, 700e6)):
                w.select_node(node)
                self.assertTrue(w.export_button.isEnabled())
                self.assertEqual(len(w.figure.axes), 3)
                path = str(Path(folder)/(node.KIND+".npz"))
                with patch("link_sim.ui.waveform_workspace.QFileDialog.getSaveFileName", return_value=(path, "")):
                    w.export_selected()
                with np.load(path) as data:
                    np.testing.assert_array_equal(data["x"], w.observed_wave().x)
                    self.assertEqual(len(data["x"]), 2048)
                    self.assertEqual(float(data["sampleRate"]), fs)
                    self.assertEqual(len(data["frequencyHz"]), len(w.observed_wave().valid))
                    self.assertEqual(len(data["spectrumDb"]), len(data["frequencyHz"]))
                    self.assertNotIn("carrierHz", data)
                    if node is source:
                        import json
                        self.assertAlmostEqual(json.loads(str(data["metrics"]))["mean_power"], 1., places=6)
                if node is source:
                    self.assertIn("平均功率 1（归一化）", w.summary.text())
                    time_axis = w.figure.axes[0]
                    self.assertEqual([line.get_label() for line in time_axis.lines], ["I", "Q"])
                    np.testing.assert_allclose(abs(time_axis.lines[0].get_ydata()), 1/np.sqrt(2), atol=1e-6)
        w.select_node(next(n for n in w.graph.all_nodes() if n.KIND == "sink"))
        self.assertIsNone(w.observed_wave())
        self.assertFalse(w.export_button.isEnabled())
        self.assertFalse(w.figure.axes, "Do not show the previous module's signal for an uncomputed output")

    def test_spectrum_view_uses_baseband_frequency_and_valid_samples(self):
        w = self.workspace
        node = next(n for n in w.graph.all_nodes() if n.KIND == "frame_source")
        x = np.zeros(4096, dtype=complex)
        x[128:2176] = .25*np.exp(2j*np.pi*.125*np.arange(2048))
        w.outputs[node.id] = Wave(x, 100e6, carrier_hz=12e9, start=128, stop=2176)
        w.refresh_observation_points(node.id)
        w.plot_view.setCurrentIndex(w.plot_view.findData("spectrum"))
        self.assertEqual(len(w.figure.axes), 1)
        ax = w.figure.axes[0]
        frequencies, levels = ax.lines[0].get_data()
        self.assertEqual(frequencies[np.argmax(levels)], 12.5)
        self.assertAlmostEqual(max(levels), 20*np.log10(.25), places=10)
        self.assertNotIn("载频", w.summary.text())
        self.assertIn("基带频率", ax.get_xlabel())
        w.plot_view.setCurrentIndex(w.plot_view.findData("all"))
        self.assertEqual(len(w.figure.axes), 3)

    def test_device_configuration_only_loads_models(self):
        for kind in ("dpd", "filter", "mixer", "pa", "rrc_tx", "rrc_rx", "precomp", "lowpass", "noise"):
            with self.subTest(kind=kind):
                dialog = StageDialog(Stage(kind, kind, kind, dict(cutoff=1., gain_db=6.)), self.workspace)
                self.assertFalse(dialog.findChildren(QDoubleSpinBox))
                self.assertTrue(dialog.details.isReadOnly())
                self.assertCountEqual([b.text() for b in dialog.findChildren(QPushButton)],
                                      ["从模型库加载…", "应用", "取消"])
                self.assertIn("理想直通" if kind not in ("rrc_tx", "rrc_rx") else "RRC", dialog.model_label.text())
                dialog.enabled.setChecked(False)
                dialog.accept()
                self.assertEqual(dialog.stage.settings, {})
                self.assertFalse(dialog.stage.enabled)
                dialog.deleteLater()

    def test_rrc_library_model_parameters_without_input_controls(self):
        dialog = StageDialog(Stage("r", "rrc_tx", "发射 RRC"), self.workspace)
        first, second = rrc_model(4), rrc_model(8, .35, 10)
        class Library(QDialog):
            selected_model = first
            def __init__(self, *args):
                super().__init__()
            def exec(self):
                return QDialog.DialogCode.Accepted
        with patch("link_sim.ui.waveform_workspace.ModelLibraryDialog", Library):
            dialog.choose_model()
        self.assertEqual(dialog.stage.model.parameters["sps"], 4)
        Library.selected_model = second
        with patch("link_sim.ui.waveform_workspace.ModelLibraryDialog", Library):
            dialog.choose_model()
        self.assertFalse(dialog.findChildren(QDoubleSpinBox))
        self.assertEqual(dialog.stage.model.parameters["sps"], 8)
        np.testing.assert_array_equal(dialog.stage.model.arrays["h"], second.arrays["h"])
        dialog.accept()
        self.assertEqual(dialog.stage.settings, {})
        dialog.deleteLater()

    def test_frame_source_shared_modcod_padding_header_and_live_bit_count(self):
        from link_budget.algorithm.ber_core.catalog import apsk_definition, session_catalog, update_session_catalog
        saved = session_catalog()
        custom = apsk_definition(16, [4, 12], [1., 2.85], [0., 15.])
        code = dict(id="test_conv", name="已新增卷积码", family="conv", k=128, n=260,
                    length=3, polys=["7", "5"], terminate=True)
        update_session_catalog(saved["modulations"]+[custom], saved["codes"]+[code])
        try:
            dialog = StageDialog(Stage("source", "frame_source", "发射信号", default_settings("frame_source")), self.workspace)
            self.assertEqual(dialog.fields["symbol_rate"][0].value(), 700.)
            self.assertEqual(dialog.fields["seed"][0].value(), 0)
            self.assertTrue({"words", "sps", "rolloff", "span", "k", "n"}.isdisjoint(dialog.fields))
            dialog.fields["segments"][0].setValue(3)
            editor = dialog.frame_editor
            self.assertEqual(editor.table.rowCount(), 1)
            self.assertEqual(editor.header.text(), "1ACFFC1D")
            self.assertIn("1,536 bit", editor.summary.text())
            self.assertIn("512", editor.summary.text())
            mod, family, coding = editor.combos[0]
            mod.setCurrentText("BPSK")
            dialog.fields["payload_symbols"][0].setValue(512)
            self.assertIn("超过", editor.error)
            with patch("link_sim.ui.waveform_workspace.QMessageBox.warning") as warning:
                dialog.accept()
                warning.assert_called_once()
            self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
            dialog.fields["payload_symbols"][0].setValue(1024)
            self.assertIsNone(editor.error)
            self.assertGreaterEqual(mod.findText(custom["name"]), 0)
            mod.setCurrentText("16QAM")
            family.setCurrentIndex(family.findData("conv"))
            self.assertEqual(coding.currentData()["id"], "test_conv")
            self.assertIn("384 bit", editor.summary.text())
            self.assertIn("959", editor.summary.text())
            editor.header.setText("0x0001")
            self.assertIn("16 符号 BPSK", editor.summary.text())
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            self.assertEqual(dialog.stage.settings["business_frame"]["code"]["id"], "test_conv")
            self.assertEqual(dialog.stage.settings["business_frame"]["mod"]["name"], "16QAM")
            self.assertEqual(dialog.stage.settings["header_hex"], "0001")
            self.assertNotIn("business_frames", dialog.stage.settings)
            self.assertNotIn("sps", dialog.stage.settings)
            dialog.deleteLater()
        finally:
            update_session_catalog(saved["modulations"], saved["codes"])

    def test_dpd_numeric_controls_keep_editable_fields_without_arrows(self):
        panel = PADPDWidget(self.workspace)
        panel.setStyleSheet(EMBEDDED_STYLE)
        for field in (panel.memory, panel.degree, panel.cross):
            self.assertFalse(field.findChildren(QToolButton))
            option = QStyleOptionSpinBox()
            field.initStyleOption(option)
            for control in (QStyle.SubControl.SC_SpinBoxUp, QStyle.SubControl.SC_SpinBoxDown):
                self.assertEqual(field.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, control, field).width(), 0)
        edit = panel.memory.lineEdit()
        edit.selectAll()
        QTest.keyClicks(edit, "8")
        panel.memory.interpretText()
        self.assertEqual(panel.memory.value(), 8)

    def test_dpd_50mhz_default_and_one_click_channel_settings_fit_sampling_band(self):
        panel = PADPDWidget(self.workspace)
        panel.set_model(PAModel([[2.]], "memPoly", 1, sample_rate=860.16e6))
        panel.source.setCurrentIndex(1)
        self.assertEqual(panel.bandwidth.currentText(), "50")
        self.assertEqual(panel.oversampling.value(), 3)
        self.assertAlmostEqual(panel.rate.value(), 184.32)
        self.assertEqual([f.value() for f in (panel.main_bw, panel.adj_bw, panel.adj_offset)], [50.,50.,50.])
        self.assertEqual(panel.configuration(True)["rate"], 184.32e6)
        self.assertIn("设置有效", panel.channel_hint.text())
        panel.oversampling.setValue(1)
        self.assertIn("至少为 3", panel.channel_hint.text())
        with self.assertRaisesRegex(ValueError, "150 MHz"):
            panel.configuration(True)
        panel.channel_defaults.click()
        self.assertEqual(panel.oversampling.value(), 3)
        panel.oversampling.setValue(6)
        panel.channel_defaults.click()
        self.assertEqual(panel.oversampling.value(), 6)
        panel.bandwidth.setCurrentText("100")
        self.assertEqual(panel.oversampling.value(), 7)
        self.assertEqual([f.value() for f in (panel.main_bw, panel.adj_bw, panel.adj_offset)], [100.,100.,100.])
        self.assertEqual(panel.configuration(True)["rate"], 860.16e6)
        panel.source.setCurrentIndex(0)
        self.assertFalse(panel.channel_defaults.isEnabled())

    def test_embedded_pa_fields_have_a_mouse_editable_text_area(self):
        page = LazyDialogPage("link_sim.algorithm.pa_model", "PAModelDialog")
        page.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        page.resize(1320, 900)
        page.show()
        try:
            deadline = time.monotonic()+5
            while not page.loaded and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertTrue(page.loaded, "Background editor import did not finish")
            dialog = page.dialog
            for field, value in ((dialog.e_memlen, "8"), (dialog.e_deglen, "7")):
                self.assertGreater(field.lineEdit().width(), field.width() / 2)
                QTest.mouseClick(field.lineEdit(), Qt.MouseButton.LeftButton, pos=field.lineEdit().rect().center())
                QTest.keyClick(field.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
                QTest.keyClicks(field.lineEdit(), value)
                QTest.keyClick(field.lineEdit(), Qt.Key.Key_Return)
                self.assertEqual(field.value(), int(value))
            dialog.combo_model.setCurrentIndex(1)
            self.assertTrue(dialog.e_crossdepth.isEnabled())
        finally:
            page.close()
            page.deleteLater()

    def test_zoomed_dual_graph_keeps_all_titles_and_lane_labels(self):
        w = self.workspace
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        w.resize(1300, 850)
        w.show()
        for node in w.graph.all_nodes():
            self.assertNotRegex(node.view.text_item.toPlainText(), r"^[AB] · ")
            if node.KIND == "lowpass":
                self.assertEqual(node.view.text_item.toPlainText(), "低通滤波")
        w.link_mode.setCurrentIndex(1)
        self.app.processEvents()
        w.fit_canvas()
        w.graph.widget.grab()
        self.assertEqual(len(w.graph.all_nodes()), 33)
        channel, = [node for node in w.graph.all_nodes() if node.KIND in ("channel", "dual_channel")]
        self.assertEqual(channel.name(), "无线信道")
        self.assertEqual((len(channel.input_ports()), len(channel.output_ports())), (2, 2))
        for node in w.graph.all_nodes():
            self.assertTrue(node.view.text_item.isVisible(), node.name())
            if node.KIND not in ("xpic", "dual_channel", "fastica"):
                self.assertRegex(node.name(), r"^[AB] · ")
            if node.KIND == "lowpass":
                self.assertRegex(node.view.text_item.toPlainText(), r"^[AB] · 低通滤波$")
        self.assertEqual(w.point.count(), 6)
        labels = {button.text() for button in w.findChildren(QPushButton) if button.isVisibleTo(w)}
        self.assertFalse(labels & {"添加模块", "配置所选…", "删除所选", "1:1", "取消"})
        self.assertTrue({"重建收发链路", "适应画布", "打开方案…", "保存方案…",
                         "计算所选输出", "运行完整链路"} <= labels)
        for kind in ("channel", "dual_channel"):
            dialog = StageDialog(Stage("ch", kind, "无线信道", dict(snr_db=12., delay=4.)), w)
            try:
                self.assertEqual(set(dialog.fields), {"snr_db"})
                dialog.accept()
                self.assertEqual(dialog.stage.settings, {"snr_db": 12.})
            finally:
                dialog.deleteLater()

    def test_node_caption_pair_and_bypass_preserve_module_name(self):
        from test_filter_inverse import lowpass
        from link_sim.algorithm.waveform_chain import model_to_dict
        w = self.workspace
        pre = next(n for n in w.graph.all_nodes() if n.KIND == "precomp")
        original_name = pre.name()
        mate = w.graph.get_node_by_id(pre.get_property("settings")["paired_node"])
        model = lowpass()
        model.name = "这是一个很长的已加载低通滤波器模型名称用于验证模块下方自动换行"
        pre.set_property("device_model", model_to_dict(model))
        self.assertEqual(pre.name(), original_name)
        self.assertEqual(mate.get_property("device_model")["name"], model.name)
        self.assertEqual(pre.view.model_caption.toPlainText(), model.name)
        self.assertLessEqual(pre.view.model_caption.boundingRect().width(), pre.view.WIDTH)
        self.assertGreater(pre.view.model_caption.boundingRect().height(), 30)
        self.assertGreaterEqual(pre.view.model_caption.pos().y(), pre.view.BODY_HEIGHT)
        title = pre.view.text_item
        self.assertAlmostEqual(title.pos().x()+title.boundingRect().width()/2, pre.view.WIDTH/2)
        self.assertAlmostEqual(title.pos().y()+title.boundingRect().height()/2, pre.view.BODY_HEIGHT/2)
        self.assertEqual(title.font().pointSize(), 16)
        self.assertEqual(pre.view.model_caption.font().pointSize(), 14)
        self.assertTrue(pre.view.applied)
        pre.set_disabled(True)
        self.assertEqual(pre.name(), original_name)
        self.assertFalse(pre.view.applied)
        self.assertIn("Bypass", pre.view.toolTip())
        self.assertFalse(pre.view._x_item.isVisible())
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/"paired.json")
            with patch("link_sim.ui.waveform_workspace.QFileDialog.getSaveFileName", return_value=(path, "")):
                w.save_scheme()
            with patch("link_sim.ui.waveform_workspace.QFileDialog.getOpenFileName", return_value=(path, "")):
                w.open_scheme()
        pre = next(n for n in w.graph.all_nodes() if n.KIND == "precomp")
        mate = w.graph.get_node_by_id(pre.get_property("settings")["paired_node"])
        self.assertIsNotNone(mate)
        self.assertEqual(mate.get_property("settings")["paired_node"], pre.id)

    def test_zoomed_canvas_pans_with_blank_left_and_right_drag(self):
        w = self.workspace
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        w.resize(1300, 850)
        w.show()
        self.app.processEvents()
        viewer = w.graph.viewer()
        viewer.reset_zoom()
        positions = [node.pos() for node in w.graph.all_nodes()]
        for button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            blank = next(QPoint(50, y) for y in range(25, viewer.viewport().height(), 25)
                         if viewer.itemAt(QPoint(50, y)) is None)
            before = viewer.scene_rect()[:]
            zoom = viewer.transform().m11()
            QTest.mousePress(viewer.viewport(), button, pos=blank)
            QTest.mouseMove(viewer.viewport(), blank+QPoint(80, 30))
            QTest.mouseRelease(viewer.viewport(), button, pos=blank+QPoint(80, 30))
            self.assertNotEqual(before[:2], viewer.scene_rect()[:2])
            self.assertAlmostEqual(viewer.transform().m11(), zoom)
            self.assertEqual(positions, [node.pos() for node in w.graph.all_nodes()])

    def test_async_run_selection_and_model_edit(self):
        w = self.workspace
        self.assertFalse(hasattr(w, "backend"))
        w.run_chain()
        self.wait_for_worker()
        self.assertTrue(w.export_button.isEnabled())
        w.fit_canvas()
        self.assertEqual(len(w.outputs), 16, "Selecting/repositioning nodes must preserve results")
        pa_node = next(n for n in w.graph.all_nodes() if n.KIND == "pa")
        w.select_node(pa_node)
        self.assertEqual(w.point.currentData(), pa_node.id)
        stage = w.stage_from_node(pa_node)
        model = from_pa(PAModel(np.array([[2.]]), "memPoly", 1,
                               sample_rate=w.outputs[pa_node.id].fs), "两倍增益")
        stage.model = model
        # Exercise the same accepted-dialog route as double-click + library selection.
        class AppliedDialog(QDialog):
            def __init__(self, *args):
                super().__init__()
                self.stage = stage

            def exec(self):
                return QDialog.DialogCode.Accepted

        with patch("link_sim.ui.waveform_workspace.StageDialog", AppliedDialog):
            w.configure_node(pa_node)
        self.assertFalse(w.outputs)
        self.assertFalse(w.export_button.isEnabled())
        w.run_chain()
        self.wait_for_worker()
        mixer = next(n for n in w.graph.all_nodes() if n.KIND == "mixer")
        np.testing.assert_allclose(w.outputs[pa_node.id].x, 2*w.outputs[mixer.id].x)

    def test_standalone_pa_accepts_unmatched_rate_and_keeps_user_input_rate(self):
        from link_sim.algorithm.pa_simulation import PASimulationWidget
        panel = PASimulationWidget(self.workspace)
        panel.rate.setValue(2800.)
        panel.set_model(PAModel([[2.]], "memPoly", 1, sample_rate=1e6))
        self.assertEqual(panel.rate.value(), 2800.)
        panel.source.setCurrentIndex(1)
        rng = np.random.default_rng(7)
        x = rng.normal(size=256)+1j*rng.normal(size=256)
        with patch.object(panel, "input_signal", return_value=(x, 2.8e9, {})), \
             patch("link_sim.algorithm.pa_simulation.QMessageBox.warning") as warning:
            panel.run_simulation()
            warning.assert_not_called()
        self.assertEqual(panel.result["sampleRate"], 2.8e9)
        np.testing.assert_array_equal(panel.result["y"], 2*x)

    def test_save_open_scheme_rebuilds_connections(self):
        w = self.workspace
        dpd = next(n for n in w.graph.all_nodes() if n.KIND == "dpd")
        dpd.set_disabled(True)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/"transmitter.json")
            with patch("link_sim.ui.waveform_workspace.QFileDialog.getSaveFileName", return_value=(path, "")):
                w.save_scheme()
            w.graph.delete_nodes(w.graph.all_nodes())
            with patch("link_sim.ui.waveform_workspace.QFileDialog.getOpenFileName", return_value=(path, "")):
                w.open_scheme()
            stages, edges = w.snapshot()
            self.assertEqual(len(stages), 16)
            self.assertEqual(len(edges), 15)
            dpd = next(s for s in stages if s.kind == "dpd")
            dac = next(s for s in stages if s.kind == "dac")
            self.assertFalse(dpd.enabled)
            precomp = next(s for s in stages if s.kind == "precomp" and "发射" in s.name)
            self.assertIn((dpd.key, precomp.key), edges)
            self.assertIn((precomp.key, dac.key), edges)
            w.run_chain()
            self.wait_for_worker()
            self.assertIn("已旁路", w.outputs[dpd.key].notes[0])


if __name__ == "__main__":
    unittest.main()
