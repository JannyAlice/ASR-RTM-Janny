"""
帮助菜单模块
负责创建和管理帮助相关的菜单项
"""
from PyQt5.QtWidgets import QMenu, QAction

class HelpMenu(QMenu):
    """帮助菜单类"""

    def __init__(self, parent=None):
        """
        初始化帮助菜单

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("帮助(&H)")  # 设置菜单标题，带有快捷键

        # 创建菜单项
        self._create_actions()

    def _create_actions(self):
        """创建菜单项"""
        self.actions = {}

        # 创建帮助动作
        self.actions['usage'] = QAction("使用说明(&F1)", self.parent())
        self.actions['usage'].setShortcut("F1")

        self.actions['about'] = QAction("关于(&A)", self.parent())

        # 将动作添加到菜单
        self.addAction(self.actions['usage'])
        self.addAction(self.actions['about'])

    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数

        Args:
            main_window: 主窗口实例
        """
        # 帮助信号
        self.actions['usage'].triggered.connect(main_window.show_usage)
        self.actions['about'].triggered.connect(main_window.show_about)

    def update_menu_state(self, _=False):
        """
        更新菜单状态

        Args:
            _: 忽略参数，保持接口一致性
        """
        # 帮助菜单始终可用，不受录音状态影响
        pass