"""
模型管理对话框模块
用于添加、编辑和删除ASR模型配置
"""
import os
import json
import traceback
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                            QTableWidgetItem, QPushButton, QLabel, QHeaderView,
                            QMessageBox, QComboBox, QLineEdit, QFileDialog,
                            QFormLayout, QSpinBox, QCheckBox, QGroupBox, QWidget,
                            QTabWidget, QMenu, QAction)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCursor

from src.utils.config_manager import config_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelManagerDialog(QDialog):
    """模型管理对话框"""

    models_changed = pyqtSignal()  # 模型配置变更信号

    def __init__(self, parent=None):
        """初始化模型管理对话框"""
        super().__init__(parent)
        self.setWindowTitle("模型管理")
        self.resize(800, 600)

        # 加载模型配置
        self.models_config = config_manager.get_config("models", {})
        if not self.models_config:
            # 初始化默认配置
            self.models_config = {
                "vosk": [],
                "sherpa_onnx": []
            }

        self._init_ui()
        self._load_models()

    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 创建选项卡
        self.tab_widget = QTabWidget(self)

        # Vosk模型选项卡
        self.vosk_tab = QWidget()
        self.vosk_layout = QVBoxLayout(self.vosk_tab)
        self.vosk_table = QTableWidget(0, 5, self.vosk_tab)
        self.vosk_table.setHorizontalHeaderLabels(["ID", "名称", "路径", "采样率", "单词时间戳"])
        self.vosk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vosk_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.vosk_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.vosk_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.vosk_table.customContextMenuRequested.connect(lambda pos: self._show_context_menu(pos, "vosk"))
        self.vosk_layout.addWidget(self.vosk_table)

        # Vosk按钮
        vosk_button_layout = QHBoxLayout()
        self.add_vosk_button = QPushButton("添加", self.vosk_tab)
        self.add_vosk_button.clicked.connect(lambda: self._add_model("vosk"))
        vosk_button_layout.addWidget(self.add_vosk_button)
        vosk_button_layout.addStretch()
        self.vosk_layout.addLayout(vosk_button_layout)

        # Sherpa-ONNX模型选项卡
        self.sherpa_tab = QWidget()
        self.sherpa_layout = QVBoxLayout(self.sherpa_tab)
        self.sherpa_table = QTableWidget(0, 5, self.sherpa_tab)
        self.sherpa_table.setHorizontalHeaderLabels(["ID", "名称", "路径", "采样率", "持久流"])
        self.sherpa_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sherpa_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sherpa_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sherpa_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sherpa_table.customContextMenuRequested.connect(lambda pos: self._show_context_menu(pos, "sherpa_onnx"))
        self.sherpa_layout.addWidget(self.sherpa_table)

        # Sherpa按钮
        sherpa_button_layout = QHBoxLayout()
        self.add_sherpa_button = QPushButton("添加", self.sherpa_tab)
        self.add_sherpa_button.clicked.connect(lambda: self._add_model("sherpa_onnx"))
        sherpa_button_layout.addWidget(self.add_sherpa_button)
        sherpa_button_layout.addStretch()
        self.sherpa_layout.addLayout(sherpa_button_layout)

        # 添加选项卡
        self.tab_widget.addTab(self.vosk_tab, "Vosk模型")
        self.tab_widget.addTab(self.sherpa_tab, "Sherpa-ONNX模型")

        layout.addWidget(self.tab_widget)

        # 创建底部按钮
        button_layout = QHBoxLayout()

        self.close_button = QPushButton("关闭", self)
        self.close_button.clicked.connect(self.accept)

        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _load_models(self):
        """加载模型配置"""
        try:
            # 加载Vosk模型
            self.vosk_table.setRowCount(0)
            vosk_models = self.models_config.get("vosk", [])
            for i, model in enumerate(vosk_models):
                self.vosk_table.insertRow(i)
                self.vosk_table.setItem(i, 0, QTableWidgetItem(model.get("id", "")))
                self.vosk_table.setItem(i, 1, QTableWidgetItem(model.get("name", "")))
                self.vosk_table.setItem(i, 2, QTableWidgetItem(model.get("path", "")))
                self.vosk_table.setItem(i, 3, QTableWidgetItem(str(model.get("sample_rate", 16000))))
                self.vosk_table.setItem(i, 4, QTableWidgetItem("是" if model.get("use_words", True) else "否"))

            # 加载Sherpa-ONNX模型
            self.sherpa_table.setRowCount(0)
            sherpa_models = self.models_config.get("sherpa_onnx", [])
            for i, model in enumerate(sherpa_models):
                self.sherpa_table.insertRow(i)
                self.sherpa_table.setItem(i, 0, QTableWidgetItem(model.get("id", "")))
                self.sherpa_table.setItem(i, 1, QTableWidgetItem(model.get("name", "")))
                self.sherpa_table.setItem(i, 2, QTableWidgetItem(model.get("path", "")))
                self.sherpa_table.setItem(i, 3, QTableWidgetItem(str(model.get("sample_rate", 16000))))
                self.sherpa_table.setItem(i, 4, QTableWidgetItem("是" if model.get("use_persistent_stream", True) else "否"))

            logger.info("已加载模型配置")
        except Exception as e:
            logger.error(f"加载模型配置时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def _show_context_menu(self, pos, model_type):
        """显示上下文菜单"""
        table = self.vosk_table if model_type == "vosk" else self.sherpa_table
        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        model_id = table.item(row, 0).text()

        menu = QMenu(self)

        # 编辑操作
        edit_action = QAction("编辑", self)
        edit_action.triggered.connect(lambda: self._edit_model(model_type, row))
        menu.addAction(edit_action)

        # 删除操作
        delete_action = QAction("删除", self)
        delete_action.triggered.connect(lambda: self._delete_model(model_type, row))
        menu.addAction(delete_action)

        menu.exec_(QCursor.pos())

    def _add_model(self, model_type):
        """添加模型"""
        try:
            dialog = ModelConfigDialog(model_type, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                model_data = dialog.get_model_data()

                # 检查ID是否已存在
                models = self.models_config.get(model_type, [])
                for model in models:
                    if model.get("id") == model_data.get("id"):
                        QMessageBox.warning(self, "添加失败", f"ID '{model_data.get('id')}' 已存在")
                        return

                # 添加模型
                models.append(model_data)
                self.models_config[model_type] = models

                # 保存配置
                config_manager.update_and_save("models", self.models_config)

                # 重新加载模型列表
                self._load_models()

                # 发送信号
                self.models_changed.emit()

                logger.info(f"已添加{model_type}模型: {model_data.get('id')}")
        except Exception as e:
            logger.error(f"添加模型时出错: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "添加失败", f"添加模型时出错: {str(e)}")

    def _edit_model(self, model_type, row):
        """编辑模型"""
        try:
            # 获取模型数据
            table = self.vosk_table if model_type == "vosk" else self.sherpa_table
            model_id = table.item(row, 0).text()

            # 查找模型
            models = self.models_config.get(model_type, [])
            model_data = None
            for model in models:
                if model.get("id") == model_id:
                    model_data = model
                    break

            if not model_data:
                QMessageBox.warning(self, "编辑失败", f"未找到ID为 '{model_id}' 的模型")
                return

            # 打开编辑对话框
            dialog = ModelConfigDialog(model_type, model_data, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                updated_data = dialog.get_model_data()

                # 更新模型
                for i, model in enumerate(models):
                    if model.get("id") == model_id:
                        models[i] = updated_data
                        break

                self.models_config[model_type] = models

                # 保存配置
                config_manager.update_and_save("models", self.models_config)

                # 重新加载模型列表
                self._load_models()

                # 发送信号
                self.models_changed.emit()

                logger.info(f"已更新{model_type}模型: {model_id}")
        except Exception as e:
            logger.error(f"编辑模型时出错: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "编辑失败", f"编辑模型时出错: {str(e)}")

    def _delete_model(self, model_type, row):
        """删除模型"""
        try:
            # 获取模型数据
            table = self.vosk_table if model_type == "vosk" else self.sherpa_table
            model_id = table.item(row, 0).text()

            # 确认删除
            result = QMessageBox.question(
                self,
                "确认删除",
                f"确定要删除模型 '{model_id}' 吗?",
                QMessageBox.Yes | QMessageBox.No
            )
            if result != QMessageBox.Yes:
                return

            # 删除模型
            models = self.models_config.get(model_type, [])
            for i, model in enumerate(models):
                if model.get("id") == model_id:
                    del models[i]
                    break

            self.models_config[model_type] = models

            # 保存配置
            config_manager.update_and_save("models", self.models_config)

            # 重新加载模型列表
            self._load_models()

            # 发送信号
            self.models_changed.emit()

            logger.info(f"已删除{model_type}模型: {model_id}")
        except Exception as e:
            logger.error(f"删除模型时出错: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "删除失败", f"删除模型时出错: {str(e)}")


class ModelConfigDialog(QDialog):
    """模型配置对话框"""

    def __init__(self, model_type, model_data=None, parent=None):
        """
        初始化模型配置对话框

        Args:
            model_type: 模型类型 (vosk 或 sherpa_onnx)
            model_data: 模型数据 (编辑模式)
            parent: 父窗口
        """
        super().__init__(parent)
        self.model_type = model_type
        self.model_data = model_data or {}
        self.is_edit_mode = bool(model_data)

        self._init_ui()
        self._load_data()

    def _init_ui(self):
        """初始化UI"""
        if self.is_edit_mode:
            self.setWindowTitle(f"编辑{self._get_model_type_display()}模型")
        else:
            self.setWindowTitle(f"添加{self._get_model_type_display()}模型")

        self.resize(500, 400)

        layout = QVBoxLayout(self)

        # 创建表单
        form_layout = QFormLayout()

        # 模型ID
        self.id_edit = QLineEdit(self)
        if self.is_edit_mode:
            self.id_edit.setEnabled(False)  # 编辑模式下不允许修改ID
        form_layout.addRow("模型ID:", self.id_edit)

        # 模型名称
        self.name_edit = QLineEdit(self)
        form_layout.addRow("模型名称:", self.name_edit)

        # 模型路径
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        self.browse_button = QPushButton("浏览...", self)
        self.browse_button.clicked.connect(self._browse_path)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.browse_button)
        form_layout.addRow("模型路径:", path_layout)

        # 采样率
        self.sample_rate_spin = QSpinBox(self)
        self.sample_rate_spin.setRange(8000, 48000)
        self.sample_rate_spin.setSingleStep(1000)
        self.sample_rate_spin.setValue(16000)
        form_layout.addRow("采样率:", self.sample_rate_spin)

        # 根据模型类型添加特定选项
        if self.model_type == "vosk":
            # 使用单词时间戳
            self.use_words_check = QCheckBox("启用", self)
            form_layout.addRow("使用单词时间戳:", self.use_words_check)
        elif self.model_type == "sherpa_onnx":
            # 使用持久流
            self.use_persistent_stream_check = QCheckBox("启用", self)
            form_layout.addRow("使用持久流:", self.use_persistent_stream_check)

        layout.addLayout(form_layout)

        # 创建按钮
        button_layout = QHBoxLayout()

        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.clicked.connect(self.reject)

        self.save_button = QPushButton("保存", self)
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self._save_model)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

    def _load_data(self):
        """加载模型数据"""
        if not self.is_edit_mode:
            return

        self.id_edit.setText(self.model_data.get("id", ""))
        self.name_edit.setText(self.model_data.get("name", ""))
        self.path_edit.setText(self.model_data.get("path", ""))
        self.sample_rate_spin.setValue(self.model_data.get("sample_rate", 16000))

        if self.model_type == "vosk":
            self.use_words_check.setChecked(self.model_data.get("use_words", True))
        elif self.model_type == "sherpa_onnx":
            self.use_persistent_stream_check.setChecked(self.model_data.get("use_persistent_stream", True))

    def _browse_path(self):
        """浏览模型路径"""
        path = QFileDialog.getExistingDirectory(self, "选择模型目录", self.path_edit.text())
        if path:
            self.path_edit.setText(path)

    def _save_model(self):
        """保存模型配置"""
        # 验证输入
        model_id = self.id_edit.text().strip()
        if not model_id:
            QMessageBox.warning(self, "输入错误", "模型ID不能为空")
            return

        # 验证ID格式
        if not model_id.isalnum() and "_" not in model_id:
            QMessageBox.warning(self, "输入错误", "模型ID只能包含字母、数字和下划线")
            return

        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "输入错误", "模型名称不能为空")
            return

        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "输入错误", "模型路径不能为空")
            return

        # 检查路径是否存在
        if not os.path.exists(path):
            result = QMessageBox.question(
                self,
                "路径不存在",
                f"路径 '{path}' 不存在，是否继续?",
                QMessageBox.Yes | QMessageBox.No
            )
            if result == QMessageBox.No:
                return

        # 创建模型数据
        model_data = {
            "id": model_id,
            "name": name,
            "path": path,
            "sample_rate": self.sample_rate_spin.value()
        }

        # 添加特定选项
        if self.model_type == "vosk":
            model_data["use_words"] = self.use_words_check.isChecked()
        elif self.model_type == "sherpa_onnx":
            model_data["use_persistent_stream"] = self.use_persistent_stream_check.isChecked()

        # 返回模型数据
        self.model_data = model_data
        self.accept()

    def get_model_data(self):
        """获取模型数据"""
        return self.model_data

    def _get_model_type_display(self):
        """获取模型类型显示名称"""
        if self.model_type == "vosk":
            return "Vosk"
        elif self.model_type == "sherpa_onnx":
            return "Sherpa-ONNX"
        return self.model_type
