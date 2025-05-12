"""
模型选择菜单模块
负责创建和管理模型选择相关的菜单项
"""
import traceback
from PyQt5.QtWidgets import QMenu, QAction, QActionGroup
from PyQt5.QtCore import pyqtSignal

from src.utils.logger import get_logger
from src.core.plugins import PluginManager

logger = get_logger(__name__)

class ModelMenu(QMenu):
    """模型选择菜单类"""

    model_selected = pyqtSignal(str)  # 模型ID
    rtm_model_selected = pyqtSignal(str)  # RTM模型ID

    def __init__(self, parent=None):
        """
        初始化模型选择菜单

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("模型选择(&M)")  # 设置菜单标题，带有快捷键

        self.plugin_manager = PluginManager()
        self.actions = {}

        # 创建ASR转录模型子菜单
        self.asr_menu = QMenu("ASR 转录模型", self)
        self.addMenu(self.asr_menu)

        # 创建ASR模型选择动作组
        self.asr_group = QActionGroup(self)
        self.asr_group.setExclusive(True)

        # 创建RTM翻译模型子菜单
        self.rtm_menu = QMenu("RTM 翻译模型", self)
        self.addMenu(self.rtm_menu)

        # 创建RTM模型选择动作组
        self.rtm_group = QActionGroup(self)
        self.rtm_group.setExclusive(True)

        # 创建菜单项
        self._create_actions()

    def _create_actions(self):
        """创建菜单项"""
        try:
            # 清空现有菜单项
            self.asr_menu.clear()
            self.rtm_menu.clear()
            self.actions.clear()

            # 获取可用的ASR模型插件
            asr_models = self.plugin_manager.get_available_models()

            if not asr_models:
                # 如果没有可用模型，添加一个禁用的菜单项
                action = QAction("无可用ASR模型", self)
                action.setEnabled(False)
                self.asr_menu.addAction(action)
            else:
                # 添加ASR模型菜单项
                for model_id in asr_models:
                    # 获取模型元数据
                    metadata = self.plugin_manager.get_plugin_metadata(model_id)
                    if not metadata:
                        continue

                    # 创建菜单项
                    action = QAction(metadata.get('name', model_id), self)
                    action.setCheckable(True)
                    action.setData(model_id)
                    action.triggered.connect(lambda checked, mid=model_id: self._on_asr_model_selected(mid))

                    # 添加到菜单
                    self.asr_menu.addAction(action)
                    self.asr_group.addAction(action)
                    self.actions[model_id] = action

                # 设置默认选中项
                if 'vosk_small' in self.actions:
                    self.actions['vosk_small'].setChecked(True)
                elif asr_models:
                    self.actions[asr_models[0]].setChecked(True)

            # 添加RTM模型菜单项（保持原有的RTM模型）
            rtm_models = {
                'argos': "Argostranslate 模型",
                'opus': "Opus-Mt-ONNX 模型"
            }

            for model_id, model_name in rtm_models.items():
                action = QAction(model_name, self)
                action.setCheckable(True)
                action.setData(model_id)
                action.triggered.connect(lambda checked, mid=model_id: self._on_rtm_model_selected(mid))

                self.rtm_menu.addAction(action)
                self.rtm_group.addAction(action)
                self.actions[model_id] = action

            # 设置默认RTM选中项
            if 'argos' in self.actions:
                self.actions['argos'].setChecked(True)

            logger.info("已更新模型菜单")

        except Exception as e:
            logger.error(f"创建模型菜单项时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def _on_asr_model_selected(self, model_id):
        """ASR模型选择处理"""
        logger.info(f"已选择ASR模型: {model_id}")
        self.model_selected.emit(model_id)

    def _on_rtm_model_selected(self, model_id):
        """RTM模型选择处理"""
        logger.info(f"已选择RTM模型: {model_id}")
        self.rtm_model_selected.emit(model_id)

    def update_models(self):
        """更新模型列表"""
        try:
            # 重新加载插件
            self.plugin_manager.reload_plugins()

            # 重新创建菜单项
            self._create_actions()

            logger.info("已刷新模型列表")
        except Exception as e:
            logger.error(f"更新模型列表时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def set_current_model(self, model_id):
        """设置当前选中的模型"""
        if model_id in self.actions:
            self.actions[model_id].setChecked(True)

    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数

        Args:
            main_window: 主窗口实例
        """
        try:
            # 连接ASR模型选择信号
            self.model_selected.connect(main_window.set_asr_model)

            # 连接RTM模型选择信号
            self.rtm_model_selected.connect(main_window.set_rtm_model)

            logger.info("模型菜单信号连接完成")
        except Exception as e:
            logger.error(f"连接模型菜单信号时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def update_menu_state(self, is_recording=False):
        """
        更新菜单状态

        Args:
            is_recording: 是否正在录音
        """
        # 在录音时禁用所有模型选择动作
        for action in self.actions.values():
            action.setEnabled(not is_recording)