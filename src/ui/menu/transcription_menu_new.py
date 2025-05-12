"""
转录模式菜单模块
负责创建和管理转录模式相关的菜单项
"""
import traceback
from PyQt5.QtWidgets import QMenu, QAction, QActionGroup
from PyQt5.QtCore import pyqtSignal

from src.utils.logger import get_logger
from src.core.plugins import PluginManager

logger = get_logger(__name__)

class TranscriptionMenu(QMenu):
    """转录模式菜单类"""
    
    model_selected = pyqtSignal(str)  # 模型ID
    rtm_model_selected = pyqtSignal(str)  # RTM模型ID
    
    def __init__(self, parent=None):
        """
        初始化转录模式菜单
        
        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("转录模式(&T)")  # 设置菜单标题，带有快捷键
        
        self.plugin_manager = PluginManager()
        self.actions = {}
        
        # 创建子菜单
        self._create_language_mode_submenu()
        self._create_asr_model_submenu()
        self._create_rtm_model_submenu()
        self._create_audio_mode_submenu()
        
    def _create_language_mode_submenu(self):
        """创建语言模式子菜单"""
        self.language_menu = QMenu("语言模式(&L)", self)
        self.addMenu(self.language_menu)
        
        # 创建语言选择动作组
        self.lang_group = QActionGroup(self)
        self.lang_group.setExclusive(True)
        
        # 创建语言选择动作
        self.actions['en_rec'] = QAction("英文识别(&E)", self, checkable=True)
        self.actions['cn_rec'] = QAction("中文识别(&C)", self, checkable=True)
        self.actions['auto_rec'] = QAction("自动识别(&A)", self, checkable=True)
        
        # 将动作添加到组
        self.lang_group.addAction(self.actions['en_rec'])
        self.lang_group.addAction(self.actions['cn_rec'])
        self.lang_group.addAction(self.actions['auto_rec'])
        
        # 将动作添加到菜单
        self.language_menu.addAction(self.actions['en_rec'])
        self.language_menu.addAction(self.actions['cn_rec'])
        self.language_menu.addAction(self.actions['auto_rec'])
        
        # 设置默认选中项
        self.actions['en_rec'].setChecked(True)
        
    def _create_asr_model_submenu(self):
        """创建ASR模型子菜单"""
        self.asr_menu = QMenu("ASR语音识别模型(&A)", self)
        self.addMenu(self.asr_menu)
        
        # 创建ASR模型选择动作组
        self.asr_group = QActionGroup(self)
        self.asr_group.setExclusive(True)
        
        # 创建Vosk系列子菜单
        self.vosk_menu = QMenu("Vosk系列(&V)", self)
        self.asr_menu.addMenu(self.vosk_menu)
        
        # 创建Vosk模型选择动作
        self.actions['vosk_small'] = QAction("Vosk Small模型(&S)", self, checkable=True)
        self.actions['vosk_medium'] = QAction("Vosk Medium模型(&M)", self, checkable=True)
        self.actions['vosk_large'] = QAction("Vosk Large模型(&L)", self, checkable=True)
        
        # 将Vosk动作添加到组
        self.asr_group.addAction(self.actions['vosk_small'])
        self.asr_group.addAction(self.actions['vosk_medium'])
        self.asr_group.addAction(self.actions['vosk_large'])
        
        # 将Vosk动作添加到菜单
        self.vosk_menu.addAction(self.actions['vosk_small'])
        self.vosk_menu.addAction(self.actions['vosk_medium'])
        self.vosk_menu.addAction(self.actions['vosk_large'])
        
        # 创建Sherpa-ONNX系列子菜单
        self.sherpa_menu = QMenu("Sherpa-ONNX系列(&S)", self)
        self.asr_menu.addMenu(self.sherpa_menu)
        
        # 创建Sherpa-ONNX模型选择动作
        self.actions['sherpa_0220_int8'] = QAction("Sherpa-ONNX 2023-02-20 Int8模型(&1)", self, checkable=True)
        self.actions['sherpa_0220_std'] = QAction("Sherpa-ONNX 2023-02-20 标准模型(&2)", self, checkable=True)
        self.actions['sherpa_0621_int8'] = QAction("Sherpa-ONNX 2023-06-21 Int8模型(&3)", self, checkable=True)
        self.actions['sherpa_0621_std'] = QAction("Sherpa-ONNX 2023-06-21 标准模型(&4)", self, checkable=True)
        self.actions['sherpa_0626_int8'] = QAction("Sherpa-ONNX 2023-06-26 Int8模型(&5)", self, checkable=True)
        self.actions['sherpa_0626_std'] = QAction("Sherpa-ONNX 2023-06-26 标准模型(&6)", self, checkable=True)
        
        # 将Sherpa-ONNX动作添加到组
        self.asr_group.addAction(self.actions['sherpa_0220_int8'])
        self.asr_group.addAction(self.actions['sherpa_0220_std'])
        self.asr_group.addAction(self.actions['sherpa_0621_int8'])
        self.asr_group.addAction(self.actions['sherpa_0621_std'])
        self.asr_group.addAction(self.actions['sherpa_0626_int8'])
        self.asr_group.addAction(self.actions['sherpa_0626_std'])
        
        # 将Sherpa-ONNX动作添加到菜单
        self.sherpa_menu.addAction(self.actions['sherpa_0220_int8'])
        self.sherpa_menu.addAction(self.actions['sherpa_0220_std'])
        self.sherpa_menu.addAction(self.actions['sherpa_0621_int8'])
        self.sherpa_menu.addAction(self.actions['sherpa_0621_std'])
        self.sherpa_menu.addAction(self.actions['sherpa_0626_int8'])
        self.sherpa_menu.addAction(self.actions['sherpa_0626_std'])
        
        # 设置默认选中项
        self.actions['vosk_small'].setChecked(True)
        
    def _create_rtm_model_submenu(self):
        """创建RTM模型子菜单"""
        self.rtm_menu = QMenu("RTM实时翻译模型(&R)", self)
        self.addMenu(self.rtm_menu)
        
        # 创建RTM模型选择动作组
        self.rtm_group = QActionGroup(self)
        self.rtm_group.setExclusive(True)
        
        # 创建RTM模型选择动作
        self.actions['opus_mt'] = QAction("Opus-MT模型(&O)", self, checkable=True)
        self.actions['argos'] = QAction("ArgosTranslate模型(&A)", self, checkable=True)
        
        # 将RTM动作添加到组
        self.rtm_group.addAction(self.actions['opus_mt'])
        self.rtm_group.addAction(self.actions['argos'])
        
        # 将RTM动作添加到菜单
        self.rtm_menu.addAction(self.actions['opus_mt'])
        self.rtm_menu.addAction(self.actions['argos'])
        
        # 设置默认选中项
        self.actions['opus_mt'].setChecked(True)
        
    def _create_audio_mode_submenu(self):
        """创建音频模式子菜单"""
        # 添加分隔线
        self.addSeparator()
        
        # 创建音频模式动作组
        self.audio_group = QActionGroup(self)
        self.audio_group.setExclusive(True)
        
        # 创建音频模式动作
        self.actions['system_audio'] = QAction("系统音频模式(&S)", self, checkable=True)
        self.actions['file_audio'] = QAction("文件音频模式(&F)", self, checkable=True)
        
        # 将动作添加到组
        self.audio_group.addAction(self.actions['system_audio'])
        self.audio_group.addAction(self.actions['file_audio'])
        
        # 将动作添加到菜单
        self.addAction(self.actions['system_audio'])
        self.addAction(self.actions['file_audio'])
        
        # 设置默认选中项
        self.actions['system_audio'].setChecked(True)
        
    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数
        
        Args:
            main_window: 主窗口实例
        """
        try:
            # 语言模式选择信号
            self.actions['en_rec'].triggered.connect(
                lambda: main_window.set_recognition_language("en")
            )
            self.actions['cn_rec'].triggered.connect(
                lambda: main_window.set_recognition_language("zh")
            )
            self.actions['auto_rec'].triggered.connect(
                lambda: main_window.set_recognition_language("auto")
            )
            
            # ASR模型选择信号
            # Vosk系列
            self.actions['vosk_small'].triggered.connect(
                lambda: self._on_asr_model_selected("vosk_small")
            )
            self.actions['vosk_medium'].triggered.connect(
                lambda: self._on_asr_model_selected("vosk_medium")
            )
            self.actions['vosk_large'].triggered.connect(
                lambda: self._on_asr_model_selected("vosk_large")
            )
            
            # Sherpa-ONNX系列
            self.actions['sherpa_0220_int8'].triggered.connect(
                lambda: self._on_asr_model_selected("sherpa_0220_int8")
            )
            self.actions['sherpa_0220_std'].triggered.connect(
                lambda: self._on_asr_model_selected("sherpa_0220_std")
            )
            self.actions['sherpa_0621_int8'].triggered.connect(
                lambda: self._on_asr_model_selected("sherpa_0621_int8")
            )
            self.actions['sherpa_0621_std'].triggered.connect(
                lambda: self._on_asr_model_selected("sherpa_0621_std")
            )
            self.actions['sherpa_0626_int8'].triggered.connect(
                lambda: self._on_asr_model_selected("sherpa_0626_int8")
            )
            self.actions['sherpa_0626_std'].triggered.connect(
                lambda: self._on_asr_model_selected("sherpa_0626_std")
            )
            
            # RTM模型选择信号
            self.actions['opus_mt'].triggered.connect(
                lambda: self._on_rtm_model_selected("opus_mt")
            )
            self.actions['argos'].triggered.connect(
                lambda: self._on_rtm_model_selected("argos")
            )
            
            # 音频模式选择信号
            self.actions['system_audio'].triggered.connect(
                lambda: main_window.set_audio_mode("system")
            )
            self.actions['file_audio'].triggered.connect(
                lambda: main_window.select_file()
            )
            
            # 连接模型选择信号
            self.model_selected.connect(main_window.set_asr_model)
            self.rtm_model_selected.connect(main_window.set_rtm_model)
            
            logger.info("转录模式菜单信号连接完成")
        except Exception as e:
            logger.error(f"连接转录模式菜单信号时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
    def _on_asr_model_selected(self, model_id):
        """ASR模型选择处理"""
        logger.info(f"已选择ASR模型: {model_id}")
        self.model_selected.emit(model_id)
        
    def _on_rtm_model_selected(self, model_id):
        """RTM模型选择处理"""
        logger.info(f"已选择RTM模型: {model_id}")
        self.rtm_model_selected.emit(model_id)
        
    def update_menu_state(self, is_recording=False):
        """
        更新菜单状态
        
        Args:
            is_recording: 是否正在录音
        """
        # 在录音时禁用模型选择和音频模式选择
        self.language_menu.setEnabled(not is_recording)
        self.asr_menu.setEnabled(not is_recording)
        self.rtm_menu.setEnabled(not is_recording)
        self.actions['system_audio'].setEnabled(not is_recording)
        self.actions['file_audio'].setEnabled(not is_recording)
