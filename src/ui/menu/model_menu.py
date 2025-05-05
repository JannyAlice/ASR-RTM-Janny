"""
模型选择菜单模块
负责创建和管理模型选择相关的菜单项
"""
from PyQt5.QtWidgets import QMenu, QAction, QActionGroup

class ModelMenu(QMenu):
    """模型选择菜单类"""

    def __init__(self, parent=None):
        """
        初始化模型选择菜单

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("模型选择(&M)")  # 设置菜单标题，带有快捷键

        # 创建菜单项
        self._create_actions()

        # 设置默认选中项
        self.actions['vosk_small'].setChecked(True)
        self.actions['argos'].setChecked(True)

    def _create_actions(self):
        """创建菜单项"""
        self.actions = {}

        # 创庯ASR转录模型子菜单
        self.asr_menu = QMenu("ASR 转录模型", self.parent())
        self.addMenu(self.asr_menu)

        # 创庯ASR模型选择动作组
        self.asr_group = QActionGroup(self.parent())

        # 创庯ASR模型选择动作
        self.actions['vosk_small'] = QAction("VOSK Small 模型", self.parent(), checkable=True)
        self.actions['sherpa_onnx_int8'] = QAction("Sherpa-ONNX int8量化模型", self.parent(), checkable=True)
        self.actions['sherpa_onnx_std'] = QAction("Sherpa-ONNX 标准模型", self.parent(), checkable=True)
        self.actions['sherpa_0626_int8'] = QAction("Sherpa-ONNX 2023-06-26 int8 模型", self.parent(), checkable=True)
        self.actions['sherpa_0626_std'] = QAction("Sherpa-ONNX 2023-06-26 标准模型", self.parent(), checkable=True)

        # 将动作添加到组
        self.asr_group.addAction(self.actions['vosk_small'])
        self.asr_group.addAction(self.actions['sherpa_onnx_int8'])
        self.asr_group.addAction(self.actions['sherpa_onnx_std'])
        self.asr_group.addAction(self.actions['sherpa_0626_int8'])
        self.asr_group.addAction(self.actions['sherpa_0626_std'])

        # 将动作添加到ASR子菜单
        self.asr_menu.addAction(self.actions['vosk_small'])
        self.asr_menu.addAction(self.actions['sherpa_onnx_int8'])
        self.asr_menu.addAction(self.actions['sherpa_onnx_std'])
        self.asr_menu.addAction(self.actions['sherpa_0626_int8'])
        self.asr_menu.addAction(self.actions['sherpa_0626_std'])

        # 创庯RTM翻译模型子菜单
        self.rtm_menu = QMenu("RTM 翻译模型", self.parent())
        self.addMenu(self.rtm_menu)

        # 创庯RTM模型选择动作组
        self.rtm_group = QActionGroup(self.parent())

        # 创庯RTM模型选择动作
        self.actions['argos'] = QAction("Argostranslate 模型", self.parent(), checkable=True)
        self.actions['opus'] = QAction("Opus-Mt-ONNX 模型", self.parent(), checkable=True)

        # 将动作添加到组
        self.rtm_group.addAction(self.actions['argos'])
        self.rtm_group.addAction(self.actions['opus'])

        # 将动作添加到RTM子菜单
        self.rtm_menu.addAction(self.actions['argos'])
        self.rtm_menu.addAction(self.actions['opus'])

    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数

        Args:
            main_window: 主窗口实例
        """
        # ASR模型选择信号
        self.actions['vosk_small'].triggered.connect(
            lambda: main_window.set_asr_model("vosk_small")
        )
        self.actions['sherpa_onnx_int8'].triggered.connect(
            lambda: main_window.set_asr_model("sherpa_onnx_int8")
        )
        self.actions['sherpa_onnx_std'].triggered.connect(
            lambda: main_window.set_asr_model("sherpa_onnx_std")
        )
        self.actions['sherpa_0626_int8'].triggered.connect(
            lambda: main_window.set_asr_model("sherpa_0626_int8")
        )
        self.actions['sherpa_0626_std'].triggered.connect(
            lambda: main_window.set_asr_model("sherpa_0626_std")
        )

        # RTM模型选择信号
        self.actions['argos'].triggered.connect(
            lambda: main_window.set_rtm_model("argos")
        )
        self.actions['opus'].triggered.connect(
            lambda: main_window.set_rtm_model("opus")
        )

    def update_menu_state(self, is_recording=False):
        """
        更新菜单状态

        Args:
            is_recording: 是否正在录音
        """
        # 在录音时禁用所有模型选择动作
        for action in self.actions.values():
            action.setEnabled(not is_recording)