"""
主菜单模块
负责创建和管理应用程序的主菜单栏
"""
from PyQt5.QtWidgets import QMenuBar

from src.utils.config_manager import config_manager
from src.utils.logger import get_logger
from src.ui.menu.transcription_menu import TranscriptionMenu
from src.ui.menu.model_menu import ModelMenu
from src.ui.menu.background_menu import BackgroundMenu
from src.ui.menu.font_menu import FontMenu
from src.ui.menu.model_management_menu import ModelManagementMenu
from src.ui.menu.plugin_menu import PluginMenu
from src.ui.menu.extra_menu import ExtraMenu
from src.ui.menu.help_menu import HelpMenu

logger = get_logger(__name__)

class MainMenu(QMenuBar):
    """主菜单类，整合所有子菜单"""

    def __init__(self, parent=None):
        """
        初始化主菜单

        Args:
            parent: 父窗口
        """
        super().__init__(parent)

        # 加载配置
        self.config = config_manager

        # 设置样式
        self.setStyleSheet("""
            QMenuBar {
                background-color: rgba(60, 60, 60, 255);
                color: white;
                border: none;
                padding: 2px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background: rgba(80, 80, 80, 255);
                border-radius: 4px;
            }
            QMenuBar::item:pressed {
                background: rgba(100, 100, 100, 255);
                border-radius: 4px;
            }
            /* 添加菜单项选中样式 */
            QMenu::item:checked {
                background-color: rgba(74, 144, 226, 180);
            }
            QMenu::indicator:checked {
                image: url(:/images/check.png);
                position: absolute;
                left: 7px;
            }
        """)

        # 创建子菜单
        self.create_menus()

    def create_menus(self):
        """创建所有子菜单"""
        try:
            # 转录模式菜单
            self.transcription_menu = TranscriptionMenu(self.parent())
            self.addMenu(self.transcription_menu)

            # 模型选择菜单
            self.model_menu = ModelMenu(self.parent())
            self.addMenu(self.model_menu)

            # 背景模式菜单
            self.background_menu = BackgroundMenu(self.parent())
            self.addMenu(self.background_menu)

            # 字体菜单
            self.font_menu = FontMenu(self.parent())
            self.addMenu(self.font_menu)

            # 模型管理菜单
            self.model_management_menu = ModelManagementMenu(self.parent())
            self.addMenu(self.model_management_menu)

            # 插件菜单
            self.plugin_menu = PluginMenu(self.parent())
            self.addMenu(self.plugin_menu)

            # 附加功能菜单
            self.extra_menu = ExtraMenu(self.parent())
            self.addMenu(self.extra_menu)

            # 帮助菜单
            self.help_menu = HelpMenu(self.parent())
            self.addMenu(self.help_menu)

            logger.info("所有菜单创建完成")
        except Exception as e:
            logger.error(f"创建菜单时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def connect_signals(self, main_window):
        """
        连接菜单信号到主窗口槽函数

        Args:
            main_window: 主窗口实例
        """
        try:
            # 转录模式菜单信号
            self.transcription_menu.connect_signals(main_window)

            # 模型选择菜单信号
            self.model_menu.connect_signals(main_window)

            # 背景模式菜单信号
            self.background_menu.connect_signals(main_window)

            # 字体菜单信号
            self.font_menu.connect_signals(main_window)

            # 模型管理菜单信号
            self.model_management_menu.connect_signals(main_window)

            # 插件菜单信号
            if hasattr(self, 'plugin_menu'):
                self.plugin_menu.plugin_manager_requested.connect(main_window._show_plugin_manager)

            # 附加功能菜单信号
            self.extra_menu.connect_signals(main_window)

            # 帮助菜单信号
            self.help_menu.connect_signals(main_window)

            logger.info("所有菜单信号连接完成")
        except Exception as e:
            logger.error(f"连接菜单信号时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def update_menu_state(self, is_recording=False):
        """
        更新菜单状态

        Args:
            is_recording: 是否正在录音
        """
        try:
            # 在录音时禁用某些菜单
            self.transcription_menu.setEnabled(not is_recording)
            self.model_menu.setEnabled(not is_recording)

            # 插件菜单在录音时禁用
            if hasattr(self, 'plugin_menu'):
                self.plugin_menu.setEnabled(not is_recording)

            # 更新子菜单状态
            self.transcription_menu.update_menu_state(is_recording)
            self.model_menu.update_menu_state(is_recording)
            self.background_menu.update_menu_state(is_recording)
            self.font_menu.update_menu_state(is_recording)
            self.model_management_menu.update_menu_state(is_recording)
            self.extra_menu.update_menu_state(is_recording)

            logger.debug(f"菜单状态已更新，录音状态: {is_recording}")
        except Exception as e:
            logger.error(f"更新菜单状态时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())