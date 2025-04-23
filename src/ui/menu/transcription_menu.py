"""
转录模式菜单模块
负责创建和管理转录模式相关的菜单项
"""
from PyQt5.QtWidgets import QMenu, QAction, QActionGroup

class TranscriptionMenu(QMenu):
    """转录模式菜单类"""

    def __init__(self, parent=None):
        """
        初始化转录模式菜单

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("转录模式(&T)")  # 设置菜单标题，带有快捷键

        # 创建菜单项
        self._create_actions()

        # 设置默认选中项
        self.actions['en_rec'].setChecked(True) # 默认选英文识别

    def _create_actions(self):
        """创建菜单项"""
        self.actions = {}

        # 系统音频子菜单
        self.system_audio_menu = QMenu("系统音频模式", self.parent())
        self.addMenu(self.system_audio_menu)

        # 创建语言选择动作组
        self.lang_group = QActionGroup(self.parent())

        # 创建语言选择动作
        self.actions['en_rec'] = QAction("英文识别", self.parent(), checkable=True)
        self.actions['cn_rec'] = QAction("中文识别", self.parent(), checkable=True)
        self.actions['auto_rec'] = QAction("自动识别", self.parent(), checkable=True)

        # 将动作添加到组
        self.lang_group.addAction(self.actions['en_rec'])
        self.lang_group.addAction(self.actions['cn_rec'])
        self.lang_group.addAction(self.actions['auto_rec'])

        # 将动作添加到系统音频子菜单
        self.system_audio_menu.addAction(self.actions['en_rec'])
        self.system_audio_menu.addAction(self.actions['cn_rec'])
        self.system_audio_menu.addAction(self.actions['auto_rec'])

        # 添加分隔线
        self.addSeparator()

        # 文件选择动作
        self.actions['select_file'] = QAction("选择音频/视频文件", self.parent())
        self.actions['select_file'].setShortcut("Ctrl+O")
        self.addAction(self.actions['select_file'])

        # 添加分隔线
        self.addSeparator()

        # 开始/停止转录动作
        self.actions['start_transcription'] = QAction("开始转录", self.parent())
        self.actions['start_transcription'].setShortcut("F5")
        self.addAction(self.actions['start_transcription'])

        self.actions['stop_transcription'] = QAction("停止转录", self.parent())
        self.actions['stop_transcription'].setShortcut("F6")
        self.addAction(self.actions['stop_transcription'])

    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数

        Args:
            main_window: 主窗口实例
        """
        # 语言选择信号
        self.actions['en_rec'].triggered.connect(
            lambda: main_window.set_recognition_language("en")
        )
        self.actions['cn_rec'].triggered.connect(
            lambda: main_window.set_recognition_language("zh")
        )
        self.actions['auto_rec'].triggered.connect(
            lambda: main_window.set_recognition_language("auto")
        )

        # 文件选择信号
        self.actions['select_file'].triggered.connect(main_window.select_file)

        # 开始/停止转录信号
        self.actions['start_transcription'].triggered.connect(
            main_window._on_start_clicked
        )
        self.actions['stop_transcription'].triggered.connect(
            main_window._on_stop_clicked
        )

    def update_menu_state(self, is_recording=False):
        """
        更新菜单状态

        Args:
            is_recording: 是否正在录音
        """
        # 在录音时禁用某些动作
        self.actions['en_rec'].setEnabled(not is_recording)
        self.actions['cn_rec'].setEnabled(not is_recording)
        self.actions['auto_rec'].setEnabled(not is_recording)
        self.actions['select_file'].setEnabled(not is_recording)

        # 根据录音状态启用/禁用开始/停止动作
        self.actions['start_transcription'].setEnabled(not is_recording)
        self.actions['stop_transcription'].setEnabled(is_recording)