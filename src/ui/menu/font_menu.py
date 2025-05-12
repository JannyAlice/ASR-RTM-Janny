"""
字体菜单模块
负责创建和管理字体相关的菜单项
"""
from PyQt5.QtWidgets import QMenu, QAction, QActionGroup

class FontMenu(QMenu):
    """字体菜单类"""

    def __init__(self, parent=None):
        """
        初始化字体菜单

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("字体大小(&F)")  # 设置菜单标题，带有快捷键

        # 创建菜单项
        self._create_actions()

        # 设置默认选中项
        self.actions['medium'].setChecked(True)

    def _create_actions(self):
        """创建菜单项"""
        self.actions = {}

        # 创建字体大小选择动作组
        self.font_group = QActionGroup(self.parent())

        # 创建字体大小选择动作
        self.actions['small'] = QAction("小", self.parent(), checkable=True)
        self.actions['medium'] = QAction("中", self.parent(), checkable=True)
        self.actions['large'] = QAction("大", self.parent(), checkable=True)

        # 将动作添加到组
        self.font_group.addAction(self.actions['small'])
        self.font_group.addAction(self.actions['medium'])
        self.font_group.addAction(self.actions['large'])

        # 将动作添加到菜单
        self.addAction(self.actions['small'])
        self.addAction(self.actions['medium'])
        self.addAction(self.actions['large'])

    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数

        Args:
            main_window: 主窗口实例
        """
        # 字体大小选择信号
        self.actions['small'].triggered.connect(
            lambda: main_window.set_font_size("small")
        )
        self.actions['medium'].triggered.connect(
            lambda: main_window.set_font_size("medium")
        )
        self.actions['large'].triggered.connect(
            lambda: main_window.set_font_size("large")
        )

    def update_menu_state(self, _=False):
        """
        更新菜单状态

        Args:
            _: 忽略参数，保持接口一致性
        """
        # 字体菜单始终可用，不受录音状态影响
        pass