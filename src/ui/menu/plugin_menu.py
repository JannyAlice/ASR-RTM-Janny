"""
插件菜单模块
负责插件管理相关菜单
"""
import traceback
from PyQt5.QtWidgets import QMenu, QAction
from PyQt5.QtCore import pyqtSignal

from src.utils.logger import get_logger
from src.core.plugins import PluginManager

logger = get_logger(__name__)

class PluginMenu(QMenu):
    """插件菜单类"""

    plugin_manager_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("插件(&P)", parent)
        self.plugin_manager = PluginManager()

        self._init_actions()

    def _init_actions(self):
        """初始化菜单项"""
        # 插件管理
        self.manage_action = QAction("插件管理(&M)...", self)
        self.manage_action.triggered.connect(lambda: self.plugin_manager_requested.emit())
        self.addAction(self.manage_action)

        # 分隔线
        self.addSeparator()

        # 刷新插件
        self.refresh_action = QAction("刷新插件(&R)", self)
        self.refresh_action.triggered.connect(self._refresh_plugins)
        self.addAction(self.refresh_action)

    def _refresh_plugins(self):
        """刷新插件"""
        try:
            self.plugin_manager.reload_plugins()
            logger.info("已刷新插件")
        except Exception as e:
            logger.error(f"刷新插件时出错: {str(e)}")
            logger.error(traceback.format_exc())
