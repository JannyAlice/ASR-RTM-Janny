"""
插件管理对话框模块
用于管理所有插件
"""
import os
import json
import traceback
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                            QTableWidgetItem, QPushButton, QLabel, QHeaderView,
                            QMessageBox, QCheckBox, QMenu, QAction)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QCursor

from src.utils.config_manager import config_manager
from src.utils.logger import get_logger
from src.core.plugins import PluginManager
from .plugin_config_dialog import PluginConfigDialog

logger = get_logger(__name__)

class PluginManagerDialog(QDialog):
    """插件管理对话框"""

    plugin_status_changed = pyqtSignal(str, bool)  # 插件ID, 是否启用

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plugin_manager = PluginManager()
        self.config_manager = config_manager

        self._init_ui()
        self._load_plugins()

    def _init_ui(self):
        self.setWindowTitle("插件管理")
        self.resize(800, 500)

        layout = QVBoxLayout(self)

        # 创建插件列表
        self.plugin_table = QTableWidget(0, 5, self)
        self.plugin_table.setHorizontalHeaderLabels(["启用", "名称", "版本", "类型", "描述"])
        self.plugin_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.plugin_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.plugin_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.plugin_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.plugin_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.plugin_table.verticalHeader().setVisible(False)
        self.plugin_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.plugin_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.plugin_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plugin_table.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.plugin_table)

        # 创建按钮
        button_layout = QHBoxLayout()

        self.refresh_button = QPushButton("刷新", self)
        self.refresh_button.clicked.connect(self._load_plugins)

        self.config_button = QPushButton("配置", self)
        self.config_button.clicked.connect(self._configure_selected_plugin)

        self.close_button = QPushButton("关闭", self)
        self.close_button.setDefault(True)
        self.close_button.clicked.connect(self.accept)

        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.config_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _load_plugins(self):
        """加载插件列表"""
        try:
            # 清空表格
            self.plugin_table.setRowCount(0)

            # 获取所有插件
            plugins = self.plugin_manager.get_all_plugins()

            # 填充表格
            for i, (plugin_id, plugin_info) in enumerate(plugins.items()):
                self.plugin_table.insertRow(i)

                # 启用复选框
                enabled = plugin_info.get('enabled', False)
                checkbox = QCheckBox(self)
                checkbox.setChecked(enabled)
                checkbox.stateChanged.connect(lambda state, pid=plugin_id: self._toggle_plugin(pid, bool(state)))
                self.plugin_table.setCellWidget(i, 0, checkbox)

                # 名称
                name_item = QTableWidgetItem(plugin_info.get('name', plugin_id))
                name_item.setData(Qt.UserRole, plugin_id)  # 存储插件ID
                self.plugin_table.setItem(i, 1, name_item)

                # 版本
                version_item = QTableWidgetItem(plugin_info.get('version', ''))
                self.plugin_table.setItem(i, 2, version_item)

                # 类型
                type_item = QTableWidgetItem(plugin_info.get('type', ''))
                self.plugin_table.setItem(i, 3, type_item)

                # 描述
                desc_item = QTableWidgetItem(plugin_info.get('description', ''))
                self.plugin_table.setItem(i, 4, desc_item)

            logger.info(f"已加载 {len(plugins)} 个插件")

        except Exception as e:
            logger.error(f"加载插件列表时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def _toggle_plugin(self, plugin_id, enabled):
        """切换插件启用状态"""
        try:
            # 获取插件配置
            plugin_config = self.config_manager.get_plugin_config(plugin_id) or {}

            # 更新启用状态
            plugin_config['enabled'] = enabled

            # 保存配置
            if self.config_manager.register_plugin(plugin_id, plugin_config):
                logger.info(f"已{('启用' if enabled else '禁用')}插件: {plugin_id}")
                self.plugin_status_changed.emit(plugin_id, enabled)
            else:
                logger.error(f"保存插件配置失败: {plugin_id}")
                QMessageBox.critical(self, "操作失败", f"无法{('启用' if enabled else '禁用')}插件: {plugin_id}")

        except Exception as e:
            logger.error(f"切换插件状态时出错: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "操作失败", f"切换插件状态时出错: {str(e)}")

    def _configure_selected_plugin(self):
        """配置选中的插件"""
        selected_rows = self.plugin_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "提示", "请先选择一个插件")
            return

        row = selected_rows[0].row()
        name_item = self.plugin_table.item(row, 1)
        plugin_id = name_item.data(Qt.UserRole)

        logger.info(f"配置插件: {plugin_id}")

        dialog = PluginConfigDialog(plugin_id, self)
        dialog.exec_()

        # 刷新插件列表
        self._load_plugins()

    def _show_context_menu(self, pos):
        """显示上下文菜单"""
        selected_rows = self.plugin_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        name_item = self.plugin_table.item(row, 1)
        plugin_id = name_item.data(Qt.UserRole)

        logger.info(f"显示插件上下文菜单: {plugin_id}")

        menu = QMenu(self)

        # 配置操作
        config_action = QAction("配置", self)
        config_action.triggered.connect(lambda: self._configure_plugin(plugin_id))
        menu.addAction(config_action)

        # 启用/禁用操作
        checkbox = self.plugin_table.cellWidget(row, 0)
        if checkbox.isChecked():
            enable_action = QAction("禁用", self)
            enable_action.triggered.connect(lambda: checkbox.setChecked(False))
        else:
            enable_action = QAction("启用", self)
            enable_action.triggered.connect(lambda: checkbox.setChecked(True))
        menu.addAction(enable_action)

        # 显示菜单
        menu.exec_(QCursor.pos())

    def _configure_plugin(self, plugin_id):
        """配置指定的插件"""
        logger.info(f"配置插件: {plugin_id}")

        dialog = PluginConfigDialog(plugin_id, self)
        dialog.exec_()

        # 刷新插件列表
        self._load_plugins()
