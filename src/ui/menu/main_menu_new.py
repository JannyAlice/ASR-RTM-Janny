"""
主菜单模块
负责创建和管理应用程序的主菜单栏
"""
from PyQt5.QtWidgets import QMenuBar

from src.utils.config_manager import config_manager
from src.utils.logger import get_logger
from src.ui.menu.transcription_menu_new import TranscriptionMenu
from src.ui.menu.extension_menu import ExtensionMenu
from src.ui.menu.ui_settings_menu import UISettingsMenu
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
        self._set_style()

        # 创建子菜单
        self.create_menus()

    def _set_style(self):
        """设置菜单样式"""
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

    def create_menus(self):
        """创建所有子菜单"""
        try:
            # 转录模式菜单
            self.transcription_menu = TranscriptionMenu(self.parent())
            self.addMenu(self.transcription_menu)

            # 扩展管理菜单
            self.extension_menu = ExtensionMenu(self.parent())
            self.addMenu(self.extension_menu)

            # UI设置菜单
            self.ui_settings_menu = UISettingsMenu(self.parent())
            self.addMenu(self.ui_settings_menu)

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

            # 扩展管理菜单信号
            self.extension_menu.connect_signals(main_window)

            # UI设置菜单信号
            self.ui_settings_menu.connect_signals(main_window)

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
            # 更新子菜单状态
            self.transcription_menu.update_menu_state(is_recording)
            
            # 在录音时禁用扩展管理菜单
            self.extension_menu.update_menu_state(is_recording)
            
            # UI设置和帮助菜单始终可用
            self.ui_settings_menu.update_menu_state(is_recording)
            self.help_menu.update_menu_state(is_recording)
            
            logger.debug(f"菜单状态已更新，录音状态: {is_recording}")
        except Exception as e:
            logger.error(f"更新菜单状态时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
