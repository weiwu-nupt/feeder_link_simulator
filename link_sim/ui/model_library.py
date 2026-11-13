"""Shared model actions and a searchable SQLite model-library browser."""

from pathlib import Path
import os
from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialogButtonBox, QFileDialog, QMessageBox,
)
from link_sim.algorithm.model_store import ModelStore, default_database_path, format_beijing_time
from link_sim.algorithm.device_models import KINDS, from_pa, json_text, read_model_file, write_model_file
from link_sim.ui.styles import APP_STYLE, light_palette


def current_store():
    if "FEEDER_MODEL_DB" in os.environ:
        return ModelStore(default_database_path())
    path = QSettings("FeederLinkSim", "ModelLibrary").value("databasePath", str(default_database_path()))
    return ModelStore(path)


class SaveModelDialog(QDialog):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setPalette(light_palette())
        self.setStyleSheet(APP_STYLE)
        self.setWindowTitle("保存模型")
        self.setMinimumWidth(440)
        box = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(model.name or KINDS[model.kind]+"模型")
        self.name.setMaxLength(120)
        form.addRow("模型名称", self.name)
        box.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box.addWidget(buttons)

    def accept(self):
        if not self.name.text().strip():
            QMessageBox.warning(self, "模型名称", "请填写模型名称")
            return
        super().accept()


def save_named(model, parent=None, store=None, *, paired_pa=None):
    dialog = SaveModelDialog(model, parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    model.name = dialog.name.text().strip()
    return (store or current_store()).save(model, paired_pa=paired_pa)


def choose_export(parent, model):
    path, chosen = QFileDialog.getSaveFileName(parent, "导出器件模型", (model.name or model.kind)+".json",
                                             "JSON 模型 (*.json);;NumPy 模型 (*.npz)")
    if not path:
        return
    if not Path(path).suffix:
        path += ".npz" if "npz" in chosen else ".json"
    write_model_file(path, model)
    return path


class ModelLibraryDialog(QDialog):
    def __init__(self, parent=None, kind=None, store=None):
        super().__init__(parent)
        self.setPalette(light_palette())
        self.setStyleSheet(APP_STYLE)
        self.setWindowTitle("器件模型库")
        self.resize(1000, 660)
        self.store = store or current_store()
        self.expected_kind = kind
        self.selected_model = None
        self.selected_id = None
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItem("全部器件", None)
        for key, label in KINDS.items():
            self.kind.addItem(label, key)
        if kind:
            self.kind.setCurrentIndex(self.kind.findData(kind))
            self.kind.setEnabled(False)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索模型编号或名称")
        row.addWidget(self.kind)
        row.addWidget(self.search, 1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        root.addLayout(row)
        self.location = QLabel(str(self.store.path))
        self.location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.location.setWordWrap(True)
        root.addWidget(self.location)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["编号", "名称", "器件类型", "保存时间（北京时间）"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        root.addWidget(self.table, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(145)
        root.addWidget(self.details)
        actions = QHBoxLayout()
        self.import_button = QPushButton("外部文件入库")
        self.export_button = QPushButton("导出所选模型")
        self.delete_button = QPushButton("删除所选模型")
        switch = QPushButton("选择数据库…")
        actions.addWidget(self.import_button)
        actions.addWidget(self.export_button)
        actions.addWidget(self.delete_button)
        actions.addWidget(switch)
        actions.addStretch()
        self.load_button = QPushButton("加载模型" if kind else "查看详情")
        close = QPushButton("关闭")
        actions.addWidget(self.load_button)
        actions.addWidget(close)
        root.addLayout(actions)
        self.kind.currentIndexChanged.connect(self.refresh)
        self.search.textChanged.connect(self.refresh)
        self.table.itemSelectionChanged.connect(self.show_details)
        self.table.itemDoubleClicked.connect(self.load_selected)
        self.import_button.clicked.connect(self.import_file)
        self.export_button.clicked.connect(self.export_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        self.load_button.clicked.connect(self.load_selected)
        switch.clicked.connect(self.switch_database)
        close.clicked.connect(self.reject)
        self.refresh()

    def refresh(self, *_):
        try:
            entries = self.store.list(self.kind.currentData(), self.search.text())
            self.table.setRowCount(0)
            self.details.clear()
            self.selected_model = None
            self.selected_id = None
            self.load_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            for row, entry in enumerate(entries):
                self.table.insertRow(row)
                for col, text in enumerate((str(entry["id"]), entry["name"], KINDS[entry["kind"]],
                                            format_beijing_time(entry["created_at"]))):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, entry["id"])
                    self.table.setItem(row, col, item)
            if entries:
                self.table.selectRow(0)
        except Exception as exc:
            QMessageBox.warning(self, "读取模型库失败", str(exc))

    def show_details(self):
        self.selected_model, self.selected_id = None, None
        self.load_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        items = self.table.selectedItems()
        if not items:
            return
        try:
            key = items[0].data(Qt.ItemDataRole.UserRole)
            self.selected_id = key
            self.delete_button.setEnabled(True)
            model = self.store.get(key)
            self.selected_model, self.selected_id = model, key
            arrays = "\n".join(f"{name}: {value.dtype}, 形状 {value.shape}" for name, value in model.arrays.items())
            pair_text = ""
            if model.kind == "dpd":
                pa = self.store.paired_power_amplifier(key)
                pair_text = f"\n配对功放：{pa.name if pa else '当前库中未找到'}"
            self.details.setPlainText(f"#{key} {model.name} · {KINDS[model.kind]}{pair_text}\n{arrays}\n参数：{json_text(model.parameters)}")
            self.load_button.setEnabled(True)
            self.export_button.setEnabled(True)
        except Exception as exc:
            self.details.setPlainText(f"无法读取模型：{exc}")

    def load_selected(self, *_):
        if self.selected_id is not None:
            try:
                # Read again to avoid applying a record changed outside this dialog.
                self.selected_model = self.store.get(self.selected_id)
                if self.expected_kind is not None and self.isWindow():
                    self.accept()
                else:
                    self.show_details()
            except Exception as exc:
                QMessageBox.warning(self, "加载失败", str(exc))

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "外部模型入库", "", "器件模型 (*.json *.npz *.csv *.txt)")
        if path:
            try:
                model = read_model_file(path, self.expected_kind)
                if save_named(model, self, self.store) is not None:
                    self.refresh()
            except Exception as exc:
                QMessageBox.warning(self, "模型入库失败", str(exc))

    def export_selected(self):
        if self.selected_id is not None:
            try:
                choose_export(self, self.store.get(self.selected_id))
            except Exception as exc:
                QMessageBox.warning(self, "导出失败", str(exc))

    def delete_selected(self):
        if self.selected_id is None:
            return
        key = self.selected_id
        item = self.table.item(self.table.currentRow(), 1)
        name = item.text() if item is not None else ""
        answer = QMessageBox.question(self, "删除模型", f"删除模型 #{key}「{name}」及其系数？此操作不可撤销。",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete(key)
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "删除模型失败", str(exc))

    def switch_database(self):
        path, _ = QFileDialog.getSaveFileName(self, "选择现有或新建模型数据库", str(self.store.path),
                                            "SQLite 数据库 (*.sqlite3 *.sqlite *.db)",
                                            options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        try:
            if not Path(path).suffix:
                path += ".sqlite3"
            store = ModelStore(path)
            self.store = store
            if "FEEDER_MODEL_DB" not in os.environ:
                QSettings("FeederLinkSim", "ModelLibrary").setValue("databasePath", str(store.path))
            self.location.setText(str(store.path))
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "数据库不可用", str(exc))


class ModelActions(QWidget):
    def __init__(self, owner, *, file_actions=True):
        super().__init__(owner)
        self.owner = owner
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 6)
        actions = [("save_button", "保存到模型库", self.save),
                   ("load_button", "从模型库加载", self.load)]
        if file_actions:
            actions.extend((("import_button", "导入模型文件", self.import_file),
                            ("export_button", "导出模型文件", self.export_file)))
        for name, label, action in actions:
            button = QPushButton(label)
            button.setMinimumHeight(30)
            button.clicked.connect(action)
            row.addWidget(button)
            setattr(self, name, button)
        self.status = QLabel("模型文件与仿真波形分别保存")
        self.status.setStyleSheet("color:#648A9C;font-size:9pt;")
        row.addWidget(self.status, 1)

    def check_idle(self):
        page = getattr(self.owner, "dpd_page", None)
        if page is not None and page.busy:
            raise ValueError("请等待 DPD 训练完成后再操作模型")

    def save(self):
        try:
            self.check_idle()
            from link_sim.ui.device_model_bindings import capture_device
            model = capture_device(self.owner)
            pa = self.owner.dpd_page.model if model.kind == "dpd" else None
            paired_pa = from_pa(pa, f"功放 {pa.name}") if pa is not None else None
            key = save_named(model, self, paired_pa=paired_pa)
            if key is not None:
                self.status.setText(f"已入库：#{key} {model.name}")
        except Exception as exc:
            QMessageBox.warning(self, "保存模型失败", str(exc))

    def load(self):
        try:
            self.check_idle()
            dialog = ModelLibraryDialog(self, self.owner.device_model_kind())
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.apply(dialog.selected_model)
        except Exception as exc:
            QMessageBox.warning(self, "加载模型失败", str(exc))

    def apply(self, model):
        self.check_idle()
        from link_sim.ui.device_model_bindings import apply_device
        apply_device(self.owner, model)
        self.owner._library_identity = (model.kind, model.name, model.description)
        self.status.setText(f"已加载：{model.name or KINDS[model.kind]}")

    def import_file(self):
        try:
            self.check_idle()
            path, _ = QFileDialog.getOpenFileName(self, "导入器件模型", "", "器件模型 (*.json *.npz *.csv *.txt)")
            if path:
                self.apply(read_model_file(path, self.owner.device_model_kind()))
        except Exception as exc:
            QMessageBox.warning(self, "导入模型失败", str(exc))

    def export_file(self):
        try:
            self.check_idle()
            from link_sim.ui.device_model_bindings import capture_device
            model = capture_device(self.owner)
            path = choose_export(self, model)
            if path:
                self.status.setText(f"已导出：{Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self, "导出模型失败", str(exc))
