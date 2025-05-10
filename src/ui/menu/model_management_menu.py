"""
模型管理菜单模块
负责创建和管理模型管理相关的菜单项
"""
import traceback
from PyQt5.QtWidgets import QMenu, QAction
from PyQt5.QtCore import pyqtSignal

from src.utils.logger import get_logger
from src.core.plugins import PluginManager

logger = get_logger(__name__)

class ModelManagementMenu(QMenu):
    """模型管理菜单类"""

    model_manager_requested = pyqtSignal()  # 请求打开模型管理器

    def __init__(self, parent=None):
        """
        初始化模型管理菜单

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("模型管理(&A)")  # 设置菜单标题，带有快捷键

        # 初始化插件管理器
        self.plugin_manager = PluginManager()

        # 创建菜单项
        self._create_actions()

    def _create_actions(self):
        """创建菜单项"""
        self.actions = {}

        # 添加模型管理菜单项
        self.manage_action = QAction("模型管理(&M)...", self)
        self.manage_action.triggered.connect(lambda: self.model_manager_requested.emit())
        self.addAction(self.manage_action)

        # 添加刷新模型列表菜单项
        self.refresh_action = QAction("刷新模型列表(&R)", self)
        self.refresh_action.triggered.connect(self.update_models)
        self.addAction(self.refresh_action)

        # 添加分隔线
        self.addSeparator()

        # 创建模型管理动作
        self.actions['show_sys_info'] = QAction("显示系统信息", self.parent())
        self.actions['check_model_dir'] = QAction("检查模型目录", self.parent())
        self.actions['search_model_doc'] = QAction("搜索模型文档", self.parent())

        # 将动作添加到菜单
        self.addAction(self.actions['show_sys_info'])
        self.addAction(self.actions['check_model_dir'])
        self.addAction(self.actions['search_model_doc'])

    def update_models(self):
        """更新模型列表"""
        try:
            # 重新加载插件
            self.plugin_manager.reload_plugins()

            logger.info("已刷新模型列表")

            # 发送通知，让其他组件知道模型列表已更新
            # 这里可以添加一个信号，但目前我们只是记录日志
        except Exception as e:
            logger.error(f"更新模型列表时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数

        Args:
            main_window: 主窗口实例
        """
        try:
            # 连接模型管理信号
            self.model_manager_requested.connect(main_window._show_model_manager)

            # 连接其他模型管理信号
            self.actions['show_sys_info'].triggered.connect(main_window.show_system_info)
            self.actions['check_model_dir'].triggered.connect(main_window.check_model_directory)
            self.actions['search_model_doc'].triggered.connect(main_window.search_model_documentation)

            logger.info("模型管理菜单信号连接完成")
        except Exception as e:
            logger.error(f"连接模型管理菜单信号时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def update_menu_state(self, is_recording=False):
        """
        更新菜单状态

        Args:
            is_recording: 是否正在录音
        """
        # 在录音时禁用刷新按钮
        self.refresh_action.setEnabled(not is_recording)

        # 其他模型管理菜单项始终可用，不受录音状态影响