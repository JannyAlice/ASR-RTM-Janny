"""
模型管理菜单模块
负责创建和管理模型管理相关的菜单项
"""
from PyQt5.QtWidgets import QMenu, QAction

class ModelManagementMenu(QMenu):
    """模型管理菜单类"""

    def __init__(self, parent=None):
        """
        初始化模型管理菜单

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("模型管理(&A)")  # 设置菜单标题，带有快捷键

        # 创建菜单项
        self._create_actions()

    def _create_actions(self):
        """创建菜单项"""
        self.actions = {}

        # 创建模型管理动作
        self.actions['show_sys_info'] = QAction("显示系统信息", self.parent())
        self.actions['check_model_dir'] = QAction("检查模型目录", self.parent())
        self.actions['search_model_doc'] = QAction("搜索模型文档", self.parent())

        # 将动作添加到菜单
        self.addAction(self.actions['show_sys_info'])
        self.addAction(self.actions['check_model_dir'])
        self.addAction(self.actions['search_model_doc'])

    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数

        Args:
            main_window: 主窗口实例
        """
        # 模型管理信号
        self.actions['show_sys_info'].triggered.connect(main_window.show_system_info)
        self.actions['check_model_dir'].triggered.connect(main_window.check_model_directory)
        self.actions['search_model_doc'].triggered.connect(main_window.search_model_documentation)

    def update_menu_state(self, _=False):
        """
        更新菜单状态

        Args:
            _: 忽略参数，保持接口一致性
        """
        # 模型管理菜单始终可用，不受录音状态影响
        pass