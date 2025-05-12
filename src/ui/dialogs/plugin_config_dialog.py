"""
插件配置对话框模块
用于显示和编辑插件配置
"""
import os
import json
import traceback
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                            QLabel, QLineEdit, QCheckBox, QComboBox, QSpinBox,
                            QPushButton, QTabWidget, QWidget, QFileDialog,
                            QMessageBox, QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

from src.utils.config_manager import config_manager
from src.utils.logger import get_logger
from src.core.plugins import PluginManager

logger = get_logger(__name__)

class ConfigField(QWidget):
    """配置字段基类"""

    value_changed = pyqtSignal(str, object)  # 字段名, 新值

    def __init__(self, name, label, value=None, description=None, parent=None):
        super().__init__(parent)
        self.name = name
        self.label_text = label
        self.description = description
        self.initial_value = value

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建标签
        self.label = QLabel(self.label_text)
        layout.addWidget(self.label)

        # 创建控件（由子类实现）
        self.field = self._create_field()
        layout.addWidget(self.field)

        # 如果有描述，添加描述标签
        if self.description:
            desc_label = QLabel(self.description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(desc_label)

    def _create_field(self):
        """创建具体的控件（由子类实现）"""
        raise NotImplementedError("子类必须实现此方法")

    def get_value(self):
        """获取当前值（由子类实现）"""
        raise NotImplementedError("子类必须实现此方法")

    def set_value(self, value):
        """设置当前值（由子类实现）"""
        raise NotImplementedError("子类必须实现此方法")

    def reset(self):
        """重置为初始值"""
        self.set_value(self.initial_value)

class StringField(ConfigField):
    """字符串配置字段"""

    def _create_field(self):
        field = QLineEdit(self)
        if self.initial_value is not None:
            field.setText(str(self.initial_value))
        field.textChanged.connect(lambda text: self.value_changed.emit(self.name, text))
        return field

    def get_value(self):
        return self.field.text()

    def set_value(self, value):
        self.field.setText(str(value) if value is not None else "")

class BooleanField(ConfigField):
    """布尔配置字段"""

    def _create_field(self):
        field = QCheckBox(self)
        if self.initial_value is not None:
            field.setChecked(bool(self.initial_value))
        field.stateChanged.connect(lambda state: self.value_changed.emit(self.name, bool(state)))
        return field

    def get_value(self):
        return self.field.isChecked()

    def set_value(self, value):
        self.field.setChecked(bool(value) if value is not None else False)

class NumberField(ConfigField):
    """数字配置字段"""

    def __init__(self, name, label, value=None, description=None, min_value=0, max_value=1000000, parent=None):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__(name, label, value, description, parent)

    def _create_field(self):
        field = QSpinBox(self)
        field.setMinimum(self.min_value)
        field.setMaximum(self.max_value)
        if self.initial_value is not None:
            field.setValue(int(self.initial_value))
        field.valueChanged.connect(lambda value: self.value_changed.emit(self.name, value))
        return field

    def get_value(self):
        return self.field.value()

    def set_value(self, value):
        self.field.setValue(int(value) if value is not None else 0)

class ChoiceField(ConfigField):
    """选择配置字段"""

    def __init__(self, name, label, value=None, description=None, choices=None, parent=None):
        self.choices = choices or []
        super().__init__(name, label, value, description, parent)

    def _create_field(self):
        field = QComboBox(self)
        for choice in self.choices:
            if isinstance(choice, (list, tuple)) and len(choice) == 2:
                field.addItem(choice[1], choice[0])
            else:
                field.addItem(str(choice), choice)

        if self.initial_value is not None:
            index = field.findData(self.initial_value)
            if index >= 0:
                field.setCurrentIndex(index)

        field.currentIndexChanged.connect(lambda: self.value_changed.emit(self.name, field.currentData()))
        return field

    def get_value(self):
        return self.field.currentData()

    def set_value(self, value):
        index = self.field.findData(value)
        if index >= 0:
            self.field.setCurrentIndex(index)

class PathField(ConfigField):
    """路径配置字段"""

    def __init__(self, name, label, value=None, description=None, is_dir=False, parent=None):
        self.is_dir = is_dir
        super().__init__(name, label, value, description, parent)

    def _create_field(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        field = QLineEdit(self)
        if self.initial_value is not None:
            field.setText(str(self.initial_value))
        field.textChanged.connect(lambda text: self.value_changed.emit(self.name, text))

        browse_button = QPushButton("浏览...", self)
        browse_button.clicked.connect(self._browse)

        layout.addWidget(field)
        layout.addWidget(browse_button)

        container = QWidget(self)
        container.setLayout(layout)
        self.text_field = field

        return container

    def _browse(self):
        if self.is_dir:
            path = QFileDialog.getExistingDirectory(self, "选择目录", self.text_field.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", self.text_field.text())

        if path:
            self.text_field.setText(path)
            self.value_changed.emit(self.name, path)

    def get_value(self):
        return self.text_field.text()

    def set_value(self, value):
        self.text_field.setText(str(value) if value is not None else "")

class PluginConfigDialog(QDialog):
    """插件配置对话框"""

    def __init__(self, plugin_id, parent=None):
        super().__init__(parent)
        self.plugin_id = plugin_id
        self.plugin_manager = PluginManager()
        self.config_manager = config_manager
        self.fields = {}
        self.changes = {}

        self._init_ui()
        self._load_config()

    def _init_ui(self):
        self.setWindowTitle(f"插件配置 - {self.plugin_id}")
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # 创建滚动区域
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        # 创建内容容器
        self.content_widget = QWidget(scroll_area)
        self.form_layout = QFormLayout(self.content_widget)

        scroll_area.setWidget(self.content_widget)
        layout.addWidget(scroll_area)

        # 创建按钮
        button_layout = QHBoxLayout()

        self.reset_button = QPushButton("重置", self)
        self.reset_button.clicked.connect(self._reset_fields)

        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.clicked.connect(self.reject)

        self.save_button = QPushButton("保存", self)
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self._save_config)

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

    def _load_config(self):
        """加载插件配置"""
        try:
            logger.info(f"开始加载插件配置: {self.plugin_id}")

            # 获取插件配置
            plugin_config = self.config_manager.get_plugin_config(self.plugin_id)
            if not plugin_config:
                logger.warning(f"未找到插件配置: {self.plugin_id}")
                plugin_config = {}
            else:
                logger.info(f"插件配置: {plugin_config}")

            # 获取插件元数据
            plugin_meta = self.plugin_manager.get_plugin_metadata(self.plugin_id)
            if not plugin_meta:
                logger.warning(f"未找到插件元数据: {self.plugin_id}")
                return
            else:
                logger.info(f"插件元数据: {plugin_meta}")

            # 获取配置架构
            config_schema = plugin_meta.get('config_schema', {})
            if not config_schema:
                logger.warning(f"插件 {self.plugin_id} 没有配置架构")
                # 添加一个提示标签
                self.form_layout.addRow(QLabel("此插件没有可配置的选项"))
                return
            else:
                logger.info(f"配置架构: {config_schema}")

            # 创建配置字段
            for field_name, field_schema in config_schema.items():
                field_type = field_schema.get('type', 'string')
                field_label = field_schema.get('label', field_name)
                field_desc = field_schema.get('description', '')
                field_default = field_schema.get('default')
                field_value = plugin_config.get(field_name, field_default)

                logger.debug(f"创建字段: {field_name}, 类型: {field_type}, 值: {field_value}")

                field = None

                if field_type == 'string':
                    field = StringField(field_name, field_label, field_value, field_desc)
                elif field_type == 'boolean':
                    field = BooleanField(field_name, field_label, field_value, field_desc)
                elif field_type == 'number':
                    min_val = field_schema.get('min', 0)
                    max_val = field_schema.get('max', 1000000)
                    field = NumberField(field_name, field_label, field_value, field_desc, min_val, max_val)
                elif field_type == 'choice':
                    choices = field_schema.get('choices', [])
                    field = ChoiceField(field_name, field_label, field_value, field_desc, choices)
                elif field_type == 'path':
                    is_dir = field_schema.get('is_dir', False)
                    field = PathField(field_name, field_label, field_value, field_desc, is_dir)

                if field:
                    field.value_changed.connect(self._on_field_changed)
                    self.fields[field_name] = field
                    self.form_layout.addRow(field)
                    logger.debug(f"已添加字段: {field_name}")
                else:
                    logger.warning(f"未知的字段类型: {field_type}")

            if not self.fields:
                logger.warning(f"插件 {self.plugin_id} 没有创建任何配置字段")
                # 添加一个提示标签
                self.form_layout.addRow(QLabel("此插件没有可配置的选项"))
            else:
                logger.info(f"已加载 {len(self.fields)} 个配置字段")

        except Exception as e:
            logger.error(f"加载插件配置时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def _on_field_changed(self, field_name, value):
        """字段值变更处理"""
        self.changes[field_name] = value
        logger.debug(f"字段值变更: {field_name} = {value}")

    def _reset_fields(self):
        """重置所有字段"""
        for field in self.fields.values():
            field.reset()
        self.changes.clear()
        logger.debug("已重置所有字段")

    def _save_config(self):
        """保存配置"""
        try:
            if not self.changes:
                logger.info("没有配置变更，无需保存")
                self.accept()
                return

            # 获取当前配置
            plugin_config = self.config_manager.get_plugin_config(self.plugin_id) or {}

            # 应用变更
            for field_name, value in self.changes.items():
                plugin_config[field_name] = value

            # 保存配置
            if self.config_manager.register_plugin(self.plugin_id, plugin_config):
                logger.info(f"已保存插件配置: {self.plugin_id}")
                QMessageBox.information(self, "保存成功", "插件配置已保存")
                self.accept()
            else:
                logger.error(f"保存插件配置失败: {self.plugin_id}")
                QMessageBox.critical(self, "保存失败", "保存插件配置时出错")

        except Exception as e:
            logger.error(f"保存插件配置时出错: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "保存失败", f"保存插件配置时出错: {str(e)}")
