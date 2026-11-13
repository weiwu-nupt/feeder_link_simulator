"""Measured imports, absolute noise power and responsive lazy page loading."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QTabWidget
from link_sim.algorithm.device_models import DeviceModel, read_model_file, write_model_file
from link_sim.algorithm.measured_data import load_measured_filter, load_measured_mixer
from link_sim.algorithm.noise_core import generate_noise
from link_sim.algorithm.waveform_chain import Stage, Wave, process_stage
from link_sim.ui.workspace import LazyDialogPage, simulation_pages


class ChannelUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_order(self):
        pages = simulation_pages()
        self.assertEqual([title for title, _ in pages], ["信道仿真", "模型库", "全链路仿真"])
        tabs = pages[0][1].findChild(QTabWidget)
        self.assertEqual([tabs.tabText(i) for i in range(tabs.count())],
                         ["功放", "滤波器", "混频器", "ADC/DAC", "底噪模型"])
        for _, page in pages:
            page.deleteLater()

    def test_background_import_keeps_event_loop_alive_and_constructs_on_gui(self):
        gate = threading.Event()
        entered = threading.Event()
        constructed = []
        class Editor(QDialog):
            def __init__(self, parent=None):
                constructed.append(QThread.currentThread())
                super().__init__(parent)
        def slow_import(*_):
            entered.set()
            gate.wait(5)
            return Editor
        page = LazyDialogPage("test", "Editor")
        try:
            with patch("link_sim.ui.workspace._load_class", side_effect=slow_import):
                page.show()
                self.assertTrue(entered.wait(1))
                fired = []
                QTimer.singleShot(0, lambda: fired.append(True))
                self.app.processEvents()
                self.assertTrue(fired)
                self.assertTrue(page.loading)
                gate.set()
                deadline = time.monotonic()+5
                while not page.loaded and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(.01)
            self.assertTrue(page.loaded)
            self.assertEqual(constructed, [self.app.thread()])
        finally:
            gate.set()
            page.close()
            page.deleteLater()
            self.app.processEvents()

    def test_measured_imports_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            impulse = root/"impulse.mat"
            savemat(impulse, dict(h=[.25, .5, .25], fs=20e6))
            model = load_measured_filter(impulse, 1e6)
            np.testing.assert_array_equal(model.arrays["h"], [.25, .5, .25])
            self.assertEqual(model.parameters["fs"], 20e6)
            table = root/"pn.csv"
            table.write_text("offset_Hz,level_dBc_Hz\n1000,-60\n10000,-80\n", encoding="utf-8")
            mixer = load_measured_mixer(table, dict(fs=1e6, sim_time=.001, f_if_in=1e5))
            for snapshot in (model, mixer):
                target = root/(snapshot.kind+".npz")
                write_model_file(target, snapshot)
                loaded = read_model_file(target)
                self.assertEqual(loaded.parameters, snapshot.parameters)
            table.write_text("1000,-60\n1000,-80\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_measured_mixer(table, mixer.parameters)

    def test_frequency_response_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"response.csv"
            np.savetxt(path, [[0, 0, 0], [500000, 0, 0]], delimiter=",")
            model = load_measured_filter(path, 1e6)
            np.testing.assert_allclose(model.arrays["h"], np.r_[1., np.zeros(4095)], atol=1e-12)

    def test_editor_import_actions_and_noise_library_binding(self):
        from link_sim.algorithm.filter_model import FilterModelDialog
        from link_sim.algorithm.mixer_model import MixerModelDialog
        from link_sim.algorithm.model_store import ModelStore
        from link_sim.algorithm.noise_model import NoiseModelDialog
        from link_sim.ui.device_model_bindings import capture_device, apply_device
        from link_sim.ui.waveform_workspace import StageDialog
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            impulse = root/"impulse.mat"
            savemat(impulse, dict(h=[.25, .5, .25], fs=100e6))
            pn = root/"pn.csv"
            np.savetxt(pn, [[1000, -60], [100000, -100]], delimiter=",")
            for cls, path in ((FilterModelDialog, impulse), (MixerModelDialog, pn)):
                editor = cls()
                try:
                    parameter_model = capture_device(editor)
                    self.assertEqual([editor.tabs.tabText(i) for i in range(editor.tabs.count())],
                                     ["参数建模", "实测数据建模与导出"])
                    self.assertIs(editor.tabs.currentWidget(), editor.parameter_page)
                    editor.tabs.setCurrentWidget(editor.measured_page)
                    page = editor.measured_page
                    self.assertFalse(page.export_button.isEnabled())
                    with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(path), "")), \
                         patch("PySide6.QtWidgets.QMessageBox.warning", side_effect=AssertionError):
                        page.browse_button.click()
                        self.assertIsNone(page.model)
                        page.build_button.click()
                    self.assertEqual(page.model.name, path.stem)
                    self.assertTrue(page.canvas.figure.axes)
                    target = root/(path.stem+".npz")
                    with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(target), "")), \
                         patch("PySide6.QtWidgets.QMessageBox.warning", side_effect=AssertionError):
                        page.export_button.click()
                    self.assertEqual(read_model_file(target).parameters, page.model.parameters)
                    store = ModelStore(root/"measured.sqlite3")
                    with patch("link_sim.ui.model_library.current_store", return_value=store), \
                         patch("link_sim.ui.model_library.SaveModelDialog.exec", return_value=QDialog.DialogCode.Accepted), \
                         patch("PySide6.QtWidgets.QMessageBox.warning", side_effect=AssertionError):
                        page.save_button.click()
                    [entry] = store.list(page.model.kind)
                    self.assertEqual(store.get(entry["id"]).parameters, page.model.parameters)
                    editor.tabs.setCurrentWidget(editor.parameter_page)
                    self.assertEqual(capture_device(editor).parameters, parameter_model.parameters)
                    page.fs.setText("50")
                    self.assertFalse(page.export_button.isEnabled())
                    self.assertFalse(page.save_button.isEnabled())
                finally:
                    editor.close()
                    editor.deleteLater()
            noise = NoiseModelDialog()
            stage = StageDialog(Stage("floor", "noise", "底噪"))
            try:
                model = DeviceModel("noise", dict(fs=20e6, density_dbfs_hz=-140., seed=2026), name="测试底噪").validated()
                apply_device(noise, model)
                self.assertEqual(capture_device(noise).parameters, model.parameters)
                stage.stage.model = model
                stage.refresh_model()
                stage.accept()
                self.assertEqual(stage.result(), QDialog.DialogCode.Accepted)
            finally:
                noise.close()
                noise.deleteLater()
                stage.close()
                stage.deleteLater()
                self.app.processEvents()

    def test_absolute_noise_power_and_chain_addition(self):
        fs = 20e6
        noise = generate_noise(300000, fs, -140)
        self.assertAlmostEqual(np.mean(abs(noise)**2)/(fs*1e-14), 1, delta=.01)
        np.testing.assert_array_equal(noise, generate_noise(len(noise), fs, -140))
        model = DeviceModel("noise", dict(fs=100e6, density_dbfs_hz=-140., seed=2025)).validated()
        x = np.ones(4096, dtype=complex)
        output = process_stage(Stage("n", "noise", "底噪", {}, model), Wave(x, fs))
        np.testing.assert_allclose(output.x, x+output.added_error)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"noise.json"
            write_model_file(path, model)
            self.assertEqual(read_model_file(path).parameters, model.parameters)


if __name__ == "__main__":
    unittest.main()
