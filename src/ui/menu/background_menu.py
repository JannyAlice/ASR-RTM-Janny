"""
背景模式菜单模块
负责创建和管理背景模式相关的菜单项
"""
from PyQt5.QtWidgets import QMenu, QAction, QActionGroup

class BackgroundMenu(QMenu):
    """背景模式菜单类"""

    def __init__(self, parent=None):
        """
        初始化背景模式菜单

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("背景模式(&B)")  # 设置菜单标题，带有快捷键

        # 创建菜单项
        self._create_actions()

        # 设置默认选中项
        self.actions['translucent'].setChecked(True)

    def _create_actions(self):
        """创建菜单项"""
        self.actions = {}

        # 创建背景模式选择动作组
        self.bg_group = QActionGroup(self.parent())

        # 创建背景模式选择动作
        self.actions['opaque'] = QAction("不透明", self.parent(), checkable=True)
        self.actions['translucent'] = QAction("半透明", self.parent(), checkable=True)
        self.actions['transparent'] = QAction("全透明", self.parent(), checkable=True)

        # 将动作添加到组
        self.bg_group.addAction(self.actions['opaque'])
        self.bg_group.addAction(self.actions['translucent'])
        self.bg_group.addAction(self.actions['transparent'])

        # 将动作添加到菜单
        self.addAction(self.actions['opaque'])
        self.addAction(self.actions['translucent'])
        self.addAction(self.actions['transparent'])

    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数

        Args:
            main_window: 主窗口实例
        """
        # 背景模式选择信号
        self.actions['opaque'].triggered.connect(
            lambda: main_window.set_background_mode("opaque")
        )
        self.actions['translucent'].triggered.connect(
            lambda: main_window.set_background_mode("translucent")
        )
        self.actions['transparent'].triggered.connect(
            lambda: main_window.set_background_mode("transparent")
        )

    def update_menu_state(self, _=False):
        """
        更新菜单状态

        Args:
            _: 忽略参数，保持接口一致性
        """
        # 背景模式菜单始终可用，不受录音状态影响
        pass