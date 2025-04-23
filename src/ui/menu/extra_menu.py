"""
附加功能菜单模块
负责创建和管理附加功能相关的菜单项
"""
from PyQt5.QtWidgets import QMenu, QAction

class ExtraMenu(QMenu):
    """附加功能菜单类"""

    def __init__(self, parent=None):
        """
        初始化附加功能菜单

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("附加功能(&E)")  # 设置菜单标题，带有快捷键

        # 创建菜单项
        self._create_actions()

    def _create_actions(self):
        """创建菜单项"""
        self.actions = {}

        # 创建附加功能动作
        self.actions['speaker_id'] = QAction("启用说话人识别", self.parent(), checkable=True)

        # 将动作添加到菜单
        self.addAction(self.actions['speaker_id'])

    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数

        Args:
            main_window: 主窗口实例
        """
        # 附加功能信号
        self.actions['speaker_id'].triggered.connect(
            lambda checked: main_window.toggle_speaker_identification(checked)
        )

    def update_menu_state(self, is_recording=False):
        """
        更新菜单状态

        Args:
            is_recording: 是否正在录音
        """
        # 在录音时禁用说话人识别切换
        self.actions['speaker_id'].setEnabled(not is_recording)