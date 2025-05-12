"""
主窗口模块
负责创建和管理应用程序的主窗口
"""
import os
import sys
import logging
import subprocess
import json
import traceback
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot, QTimer

# 导入新的菜单类
from src.ui.menu.main_menu_new import MainMenu
from src.ui.widgets.subtitle_widget import SubtitleWidget
from src.ui.widgets.control_panel import ControlPanel
from src.ui.dialogs.plugin_manager_dialog import PluginManagerDialog
from src.ui.dialogs.model_manager_dialog import ModelManagerDialog  # type: ignore
from src.core.signals import TranscriptionSignals
from src.core.asr.model_manager import ASRModelManager
from src.core.audio.audio_processor import AudioProcessor
from src.utils.config_manager import config_manager  # type: ignore
from src.utils.com_handler import com_handler  # type: ignore

# 条件导入 FileTranscriber
try:
    from src.core.audio.file_transcriber import FileTranscriber
    HAS_FILE_TRANSCRIBER = True
except ImportError:
    HAS_FILE_TRANSCRIBER = False

class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self, model_manager=None, config_manager=None):
        """初始化主窗口

        Args:
            model_manager: ASR模型管理器实例
            config_manager: 配置管理器实例
        """
        super().__init__()

        # 获取日志记录器
        from src.utils.logger import get_logger
        self.logger = get_logger(__name__)
        self.logger.info("初始化MainWindow")

        # 设置配置管理器
        from src.utils.config_manager import config_manager as global_config_manager
        self.config_manager = config_manager if config_manager else global_config_manager
        self.logger.debug(f"配置管理器类型: {type(self.config_manager)}")

        # 创建信号
        self.signals = TranscriptionSignals()
        self.logger.debug("创建转录信号")

        # 设置模型管理器
        self.model_manager = model_manager if model_manager else ASRModelManager()
        self.logger.debug(f"设置模型管理器: {type(self.model_manager).__name__}")

        # 创建音频处理器
        self.audio_processor = AudioProcessor(self.signals)
        self.logger.debug("创建音频处理器")

        # 创建文件转录器（如果可用）
        self.file_transcriber = None
        if HAS_FILE_TRANSCRIBER:
            self.file_transcriber = FileTranscriber(self.signals)
            self.logger.info("文件转录器创建成功")
        else:
            self.logger.warning("文件转录器不可用")

        # 转录模式标志
        self.is_file_mode = False
        self.file_path = None

        # 初始化UI
        self._init_ui()

        # 连接信号
        self._connect_signals()

        # 加载窗口状态
        self.load_window_state()

        # 加载默认模型
        self._load_default_model()

        # 加载音频设备
        self._load_audio_devices()

        self.logger.info("MainWindow初始化完成")

    def _init_ui(self):
        """初始化UI"""
        # 设置窗口属性
        self.setWindowTitle("实时字幕")
        self.resize(800, 400)

        # 创建中央控件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建布局
        layout = QVBoxLayout(central_widget)

        # 创建字幕控件
        self.subtitle_widget = SubtitleWidget(self)
        layout.addWidget(self.subtitle_widget, 1)  # 1表示拉伸因子

        # 创建控制面板
        self.control_panel = ControlPanel(self)
        layout.addWidget(self.control_panel)

        # 设置布局
        central_widget.setLayout(layout)

        # 创建菜单
        self.menu_bar = MainMenu(self)
        self.setMenuBar(self.menu_bar)
        self.menu_bar.connect_signals(self)

        # 设置窗口样式
        self._apply_window_style()

    def _apply_window_style(self):
        """应用窗口样式"""
        try:
            # 获取窗口配置
            window_config = self.config_manager.get_window_config()
            self.logger.debug(f"窗口配置: {window_config}")

            # 获取透明度
            opacity = window_config.get('opacity', 0.9)
            self.logger.debug(f"窗口透明度: {opacity}")

            # 保存当前窗口标志位
            current_flags = self.windowFlags()

            # 设置窗口置顶
            if window_config.get('always_on_top', True):
                # 确保保留其他标志位
                new_flags = current_flags | Qt.WindowStaysOnTopHint
                if new_flags != current_flags:
                    self.setWindowFlags(new_flags)
                    self.show()  # 仅在标志位发生变化时重新显示窗口

            # 设置窗口透明度
            self.setWindowOpacity(opacity)
            self.logger.debug("窗口样式应用完成")
        except Exception as e:
            self.logger.error(f"应用窗口样式时出错: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _connect_signals(self):
        """连接信号"""
        try:
            self.logger.info("开始连接信号...")

            # 连接转录信号 - 添加信号存在性检查
            if hasattr(self.signals, 'new_text'):
                self.logger.debug("连接 new_text 信号")
                self.signals.new_text.connect(self.subtitle_widget.update_text)
            else:
                self.logger.warning("未找到 new_text 信号")

            if hasattr(self.signals, 'progress_updated'):
                self.logger.debug("连接 progress_updated 信号")
                self.signals.progress_updated.connect(self.control_panel.update_progress)
            else:
                self.logger.warning("未找到 progress_updated 信号")

            if hasattr(self.signals, 'status_updated'):
                self.logger.debug("连接 status_updated 信号")
                self.signals.status_updated.connect(self.control_panel.update_status)
            else:
                self.logger.warning("未找到 status_updated 信号")

            if hasattr(self.signals, 'error_occurred'):
                self.logger.debug("连接 error_occurred 信号")
                self.signals.error_occurred.connect(self._show_error)
            else:
                self.logger.warning("未找到 error_occurred 信号")

            if hasattr(self.signals, 'transcription_finished'):
                self.logger.debug("连接 transcription_finished 信号")
                self.signals.transcription_finished.connect(self._on_transcription_finished)
            else:
                self.logger.warning("未找到 transcription_finished 信号")

            # 连接新增的生命周期信号（如果存在）
            if hasattr(self.signals, 'transcription_started'):
                self.logger.debug("连接 transcription_started 信号")
                self.signals.transcription_started.connect(self._on_transcription_started)

            if hasattr(self.signals, 'transcription_paused'):
                self.logger.debug("连接 transcription_paused 信号")
                self.signals.transcription_paused.connect(self._on_transcription_paused)

            if hasattr(self.signals, 'transcription_resumed'):
                self.logger.debug("连接 transcription_resumed 信号")
                self.signals.transcription_resumed.connect(self._on_transcription_resumed)

            self.logger.info("信号连接完成")
        except Exception as e:
            self.logger.error(f"连接信号时出错: {str(e)}")
            self.logger.error(traceback.format_exc())

        # 连接控制面板信号
        self.control_panel.start_clicked.connect(self._on_start_clicked)
        self.control_panel.stop_clicked.connect(self._on_stop_clicked)

        # 模型管理菜单信号已在 MainMenu.connect_signals 中连接

    def _load_default_model(self):
        """加载默认模型"""
        # 获取默认模型
        default_model = self.config_manager.get_default_model()
        self.logger.info(f"获取默认模型: {default_model}")

        # 更新菜单选中状态 - 使用新的菜单结构
        try:
            # 检查是否使用新的菜单结构
            if hasattr(self.menu_bar, 'transcription_menu') and hasattr(self.menu_bar.transcription_menu, 'actions'):
                # 新的菜单结构
                for model_id, action in self.menu_bar.transcription_menu.actions.items():
                    if model_id in ['vosk_small', 'vosk_medium', 'vosk_large',
                                   'sherpa_0220_int8', 'sherpa_0220_std',
                                   'sherpa_0621_int8', 'sherpa_0621_std',
                                   'sherpa_0626_int8', 'sherpa_0626_std']:
                        if action.isCheckable():
                            action.setChecked(model_id == default_model)
            # 兼容旧的菜单结构
            elif hasattr(self.menu_bar, 'model_menu') and hasattr(self.menu_bar.model_menu, 'actions'):
                for key, action in self.menu_bar.model_menu.actions.items():
                    if key in ['vosk_small', 'sherpa_onnx_int8', 'sherpa_onnx_std', 'sherpa_0626_int8', 'sherpa_0626_std']:
                        action.setChecked(key == default_model)
        except Exception as e:
            self.logger.error(f"更新菜单选中状态时出错: {str(e)}")
            self.logger.error(traceback.format_exc())

        # 加载模型
        if self.model_manager.load_model(default_model):
            # 获取模型显示名称
            model_display_name = self._get_model_display_name(default_model)
            self.logger.debug(f"模型显示名称: {model_display_name}")

            # 获取引擎类型
            engine_type = self.model_manager.get_current_engine_type()
            self.logger.info(f"默认引擎类型: {engine_type}")

            # 更新字幕窗口的引擎类型
            self.subtitle_widget.current_engine_type = engine_type
            self.logger.info(f"更新字幕窗口的引擎类型: {engine_type}")

            # 更新状态栏
            self.signals.status_updated.emit(f"已加载ASR模型: {model_display_name}")

            # 在字幕窗口显示初始信息
            model_info = f"已加载ASR模型: {model_display_name}"
            info_text = f"{model_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"

            # 设置字幕窗口文本
            self.subtitle_widget.transcript_text = []
            self.subtitle_widget.subtitle_label.setText(info_text)

            self.logger.info(f"成功加载默认模型: {default_model}")
        else:
            error_msg = f"加载默认模型 {default_model} 失败"
            self.logger.error(error_msg)
            self.signals.error_occurred.emit(error_msg)

    def _load_audio_devices(self):
        """加载音频设备"""
        # 获取音频设备
        devices = self.audio_processor.get_audio_devices()

        # 设置设备列表
        self.control_panel.set_devices(devices)

        # 如果有设备，尝试查找并选择 "CABLE in 16 ch"，如果没有则选择第一个
        if devices:
            # 默认选择第一个设备
            default_device = devices[0]

            # 尝试查找 "CABLE in 16 ch" 设备
            for device in devices:
                if "CABLE" in device.name and "16 ch" in device.name:
                    default_device = device
                    break

            # 设置当前设备
            self.audio_processor.set_current_device(default_device)

            # 在设备下拉列表中选中该设备
            for i in range(self.control_panel.device_combo.count()):
                if self.control_panel.device_combo.itemText(i) == default_device.name:
                    self.control_panel.device_combo.setCurrentIndex(i)
                    break

    @pyqtSlot()
    def _on_start_clicked(self):
        """开始按钮点击处理"""
        # 导入 Sherpa-ONNX 日志工具
        try:
            from src.utils.sherpa_logger import sherpa_logger
        except ImportError:
            # 如果导入失败，创建一个简单的日志记录器
            class DummyLogger:
                def debug(self, msg): print(f"DEBUG: {msg}")
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
            sherpa_logger = DummyLogger()

        sherpa_logger.info("开始按钮被点击")

        # 强制使用vosk_small模型
        model_type = "vosk_small"
        self.model_manager.model_type = "vosk_small"

        # 获取当前引擎信息
        engine_type = self.model_manager.get_current_engine_type()
        engine_info = type(self.model_manager.current_engine).__name__ if self.model_manager.current_engine else "None"

        sherpa_logger.info(f"当前模型类型: {model_type}")
        sherpa_logger.info(f"当前引擎类型: {engine_type}")
        sherpa_logger.info(f"当前引擎: {engine_info}")

        # 清空字幕窗口的内容
        if hasattr(self.subtitle_widget, 'transcript_text'):
            self.subtitle_widget.transcript_text = []
        if hasattr(self.subtitle_widget, 'output_file'):
            self.subtitle_widget.output_file = None

        # 重置保存标志
        MainWindow._has_saved_transcript = False

        # 根据当前模式执行不同的转录功能
        if self.is_file_mode and HAS_FILE_TRANSCRIBER and self.file_transcriber:
            # 文件转录模式
            sherpa_logger.info("文件转录模式")

            if not self.file_path:
                error_msg = "请先选择要转录的文件"
                sherpa_logger.error(error_msg)
                self.signals.error_occurred.emit(error_msg)
                self.control_panel.reset()
                return

            sherpa_logger.info(f"文件路径: {self.file_path}")

            # 检查文件是否存在
            if not os.path.exists(self.file_path):
                error_msg = f"文件不存在: {self.file_path}"
                sherpa_logger.error(error_msg)
                self.signals.error_occurred.emit(error_msg)
                self.control_panel.reset()
                return

            # 获取模型显示名称
            model_display_name = self._get_model_display_name(model_type)
            sherpa_logger.info(f"模型显示名称: {model_display_name}")

            # 检查当前引擎是否支持文件转录
            current_engine_type = self.model_manager.get_current_engine_type()
            sherpa_logger.info(f"当前引擎类型: {current_engine_type}")
            sherpa_logger.info(f"当前引擎: {type(self.model_manager.current_engine).__name__ if self.model_manager.current_engine else 'None'}")

            if current_engine_type == "vosk_small":
                # 使用 Vosk 模型转录文件
                sherpa_logger.info("使用 Vosk 模型转录文件")

                # 检查模型类型与引擎类型是否一致
                model_type = self.model_manager.model_type
                if model_type != current_engine_type:
                    sherpa_logger.warning(f"模型类型 {model_type} 与引擎类型 {current_engine_type} 不一致")
                    sherpa_logger.warning("这可能导致功能异常，请确保选择正确的模型类型")

                # 创建识别器
                sherpa_logger.info("创建 Vosk 识别器")
                recognizer = self.model_manager.create_recognizer()
                if not recognizer:
                    error_msg = "创建识别器失败"
                    sherpa_logger.error(error_msg)
                    self.signals.error_occurred.emit(error_msg)
                    self.control_panel.reset()
                    return

                sherpa_logger.info(f"识别器创建成功: {type(recognizer)}")
                sherpa_logger.info(f"识别器引擎类型: {getattr(recognizer, 'engine_type', 'unknown')}")

                # 再次检查识别器的引擎类型是否与当前引擎类型一致
                recognizer_engine_type = getattr(recognizer, 'engine_type', None)
                if recognizer_engine_type != current_engine_type:
                    sherpa_logger.warning(f"识别器引擎类型 {recognizer_engine_type} 与当前引擎类型 {current_engine_type} 不一致")
                    sherpa_logger.warning("这可能导致功能异常，请确保选择正确的模型类型")

                # 开始文件转录
                sherpa_logger.info(f"开始文件转录: {self.file_path}")
                if not self.file_transcriber.start_transcription(self.file_path, recognizer):
                    error_msg = "开始文件转录失败"
                    sherpa_logger.error(error_msg)
                    self.signals.error_occurred.emit(error_msg)
                    self.control_panel.reset()
                    return

                # 更新状态
                status_msg = f"正在使用 {model_display_name} 转录文件: {os.path.basename(self.file_path)}..."
                sherpa_logger.info(status_msg)
                self.signals.status_updated.emit(status_msg)

                # 更新字幕窗口
                subtitle_msg = f"正在使用 {model_display_name} 转录文件...\n引擎类型: {current_engine_type}"
                self.subtitle_widget.subtitle_label.setText(subtitle_msg)

            elif current_engine_type and current_engine_type.startswith("sherpa"):
                # 使用 Sherpa-ONNX 模型直接转录文件
                sherpa_logger.info(f"使用 Sherpa-ONNX 模型转录文件 (类型: {current_engine_type})")

                # 检查模型类型与引擎类型是否一致
                model_type = self.model_manager.model_type
                if model_type != current_engine_type:
                    sherpa_logger.warning(f"模型类型 {model_type} 与引擎类型 {current_engine_type} 不一致")
                    sherpa_logger.warning("这可能导致功能异常，请确保选择正确的模型类型")

                    # 尝试重新加载正确的模型
                    sherpa_logger.info(f"尝试重新加载模型: {model_type}")
                    if not self.model_manager.load_model(model_type):
                        sherpa_logger.error(f"重新加载模型 {model_type} 失败")
                        error_msg = f"无法使用请求的模型类型: {model_type}"
                        self.signals.error_occurred.emit(error_msg)
                        self.control_panel.reset()
                        return

                    # 更新引擎类型
                    current_engine_type = self.model_manager.get_current_engine_type()
                    sherpa_logger.info(f"重新加载后的引擎类型: {current_engine_type}")

                # 使用 model_manager 实例作为 recognizer
                recognizer = self.model_manager
                sherpa_logger.info(f"使用 model_manager 实例作为 recognizer: {type(recognizer)}")
                sherpa_logger.info(f"recognizer.current_engine: {type(recognizer.current_engine).__name__ if recognizer.current_engine else 'None'}")

                # 再次检查引擎类型
                engine_type = recognizer.get_current_engine_type()
                if engine_type != current_engine_type:
                    sherpa_logger.warning(f"recognizer引擎类型 {engine_type} 与当前引擎类型 {current_engine_type} 不一致")
                    sherpa_logger.warning("这可能导致功能异常，请确保选择正确的模型类型")

                # 开始文件转录
                sherpa_logger.info(f"开始文件转录: {self.file_path}")
                if not self.file_transcriber.start_transcription(self.file_path, recognizer):
                    error_msg = "开始文件转录失败"
                    sherpa_logger.error(error_msg)
                    self.signals.error_occurred.emit(error_msg)
                    self.control_panel.reset()
                    return

                # 更新状态
                status_msg = f"正在使用 {model_display_name} 转录文件: {os.path.basename(self.file_path)}..."
                sherpa_logger.info(status_msg)
                self.signals.status_updated.emit(status_msg)

                # 更新字幕窗口
                subtitle_msg = f"正在使用 {model_display_name} 转录文件...\n引擎类型: {current_engine_type}"
                self.subtitle_widget.subtitle_label.setText(subtitle_msg)

            else:
                # 不支持的引擎类型
                error_msg = f"不支持的引擎类型: {current_engine_type}"
                sherpa_logger.error(error_msg)
                self.signals.error_occurred.emit(error_msg)
                self.control_panel.reset()
                return

            # 禁用相关菜单项
            self.menu_bar.update_menu_state(is_recording=True)
        else:
            # 系统音频模式（默认模式或文件转录不可用时）
            sherpa_logger.info("系统音频模式")

            if self.is_file_mode:
                # 如果文件转录不可用，回退到系统音频模式
                self.is_file_mode = False
                status_msg = "文件转录功能不可用，已切换到系统音频模式"
                sherpa_logger.warning(status_msg)
                self.signals.status_updated.emit(status_msg)

            # 获取模型显示名称
            model_display_name = self._get_model_display_name(model_type)
            sherpa_logger.info(f"模型显示名称: {model_display_name}")

            # 检查模型类型与引擎类型是否一致
            current_engine_type = self.model_manager.get_current_engine_type()
            if model_type != current_engine_type:
                sherpa_logger.warning(f"模型类型 {model_type} 与引擎类型 {current_engine_type} 不一致")
                sherpa_logger.warning("这可能导致功能异常，请确保选择正确的模型类型")

                # 尝试重新加载正确的模型
                sherpa_logger.info(f"尝试重新加载模型: {model_type}")
                if not self.model_manager.load_model(model_type):
                    sherpa_logger.error(f"重新加载模型 {model_type} 失败")
                    error_msg = f"无法使用请求的模型类型: {model_type}"
                    self.signals.error_occurred.emit(error_msg)
                    self.control_panel.reset()
                    return

                # 更新引擎类型
                current_engine_type = self.model_manager.get_current_engine_type()
                sherpa_logger.info(f"重新加载后的引擎类型: {current_engine_type}")

            # 创建识别器
            sherpa_logger.info("创建识别器")
            recognizer = self.model_manager.create_recognizer()
            if not recognizer:
                error_msg = "创建识别器失败"
                sherpa_logger.error(error_msg)
                self.signals.error_occurred.emit(error_msg)
                self.control_panel.reset()
                return

            sherpa_logger.info(f"识别器创建成功: {type(recognizer)}")

            # 检查识别器的引擎类型
            recognizer_engine_type = getattr(recognizer, 'engine_type', None)
            sherpa_logger.info(f"识别器引擎类型: {recognizer_engine_type}")

            # 确认引擎类型与模型类型一致
            if recognizer_engine_type != model_type:
                sherpa_logger.warning(f"识别器引擎类型 ({recognizer_engine_type}) 与模型类型 ({model_type}) 不一致")
                sherpa_logger.warning("这可能导致功能异常，请确保选择正确的模型类型")

            # 开始系统音频捕获
            sherpa_logger.info("开始系统音频捕获")
            if not self.audio_processor.start_capture(recognizer):
                error_msg = "开始音频捕获失败"
                sherpa_logger.error(error_msg)
                self.signals.error_occurred.emit(error_msg)
                self.control_panel.reset()
                return

            # 设置subtitle_widget的audio_worker属性，用于在停止转录时获取最后一个单词
            try:
                if hasattr(self.audio_processor, 'worker') and self.audio_processor.worker:
                    sherpa_logger.info("设置subtitle_widget的audio_worker属性")
                    self.subtitle_widget.audio_worker = self.audio_processor.worker
                    sherpa_logger.info(f"audio_worker设置成功: {type(self.audio_processor.worker)}")
            except Exception as e:
                sherpa_logger.error(f"设置subtitle_widget的audio_worker属性错误: {e}")
                import traceback
                sherpa_logger.error(traceback.format_exc())

            # 更新状态
            status_msg = f"正在使用 {model_display_name} 转录系统音频..."
            sherpa_logger.info(status_msg)
            self.signals.status_updated.emit(status_msg)

            # 更新字幕窗口
            subtitle_msg = f"正在使用 {model_display_name} 转录系统音频...\n引擎类型: {recognizer_engine_type}"
            self.subtitle_widget.subtitle_label.setText(subtitle_msg)

            # 禁用相关菜单项
            self.menu_bar.update_menu_state(is_recording=True)

    # 用于跟踪是否已保存文件
    _has_saved_transcript = False

    @pyqtSlot()
    def _on_stop_clicked(self):
        """停止按钮点击处理"""
        # 重置保存标志
        MainWindow._has_saved_transcript = False

        # 导入日志工具
        try:
            from src.utils.sherpa_logger import sherpa_logger
        except ImportError:
            # 如果导入失败，创建一个简单的日志记录器
            class DummyLogger:
                def debug(self, msg): print(f"DEBUG: {msg}")
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
            sherpa_logger = DummyLogger()

        # 根据当前模式执行不同的停止功能
        if self.is_file_mode and HAS_FILE_TRANSCRIBER and self.file_transcriber:
            # 文件转录模式
            sherpa_logger.info("停止文件转录")
            if not self.file_transcriber.stop_transcription():
                self.signals.error_occurred.emit("停止文件转录失败")
                return
        else:
            # 系统音频模式
            sherpa_logger.info("停止系统音频捕获")

            # 停止音频捕获 - 先停止捕获，防止在处理最终结果时继续接收新的部分结果
            if not self.audio_processor.stop_capture():
                self.signals.error_occurred.emit("停止音频捕获失败")
                return

            # 在停止捕获后，尝试获取最终结果
            try:
                # 检查当前引擎类型
                current_engine_type = self.model_manager.get_current_engine_type()
                sherpa_logger.info(f"当前引擎类型: {current_engine_type}")

                # 对于Vosk模型，特殊处理
                if current_engine_type == "vosk_small" or current_engine_type == "vosk":
                    # 如果有recognizer，获取最终结果
                    if hasattr(self, 'recognizer') and self.recognizer:
                        sherpa_logger.info("获取Vosk最终识别结果")
                        final_result = self.recognizer.get_final_result()
                        sherpa_logger.info(f"Vosk最终结果: {final_result}")

                        # 如果有文本，更新字幕窗口
                        if final_result:
                            sherpa_logger.info(f"更新字幕窗口: {final_result}")
                            self.subtitle_widget.update_text(final_result)
                        # 注意：我们不再尝试处理最后的部分结果，因为静音检测已经处理了句子结束
            except Exception as e:
                sherpa_logger.error(f"获取最终结果错误: {e}")
                import traceback
                sherpa_logger.error(traceback.format_exc())

            # 音频捕获已经在前面停止了

        # 重新启用相关菜单项
        self.menu_bar.update_menu_state(is_recording=False)

        # 保存转录文本
        try:
            # 直接获取转录文本
            if hasattr(self.subtitle_widget, 'transcript_text') and self.subtitle_widget.transcript_text:
                # 确保 transcripts 目录存在
                import os
                import time
                save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "transcripts")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 生成带时间戳的文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                # 获取当前模型类型
                model_type = self.model_manager.model_type if hasattr(self.model_manager, 'model_type') else "unknown"

                # 确定转录类型（在线或文件）
                transcription_type = "FILE" if self.is_file_mode else "ONLINE"

                # 生成文件名，包含明显的转录类型标识和自动保存标记
                filename = f"transcript_{transcription_type}_{model_type}_{timestamp}_AUTO.txt"

                # 完整的保存路径
                save_path = os.path.join(save_dir, filename)

                # 保存文件 - 使用带时间戳的转录历史记录
                with open(save_path, 'w', encoding='utf-8') as f:
                    # 获取带时间戳的转录历史记录
                    timestamped_transcript = self.subtitle_widget.get_timestamped_transcript()
                    f.write(timestamped_transcript)

                # 同时保存一个包含所有数据的调试文件
                debug_path = save_path.replace('.txt', '_debug.txt')
                with open(debug_path, 'w', encoding='utf-8') as f:
                    # 获取所有转录数据
                    all_data = self.subtitle_widget.get_all_transcript_data()

                    # 写入带时间戳的转录历史
                    f.write("=== 带时间戳的转录历史 ===\n")
                    timestamped_text = []
                    for text, timestamp in all_data['timestamped_transcript']:
                        timestamped_text.append(f"[{timestamp}] {text}")
                    f.write('\n'.join(timestamped_text))
                    f.write("\n\n")

                    # 写入完整转录历史
                    f.write("=== 完整转录历史 ===\n")
                    f.write('\n'.join(all_data['full_transcript']))
                    f.write("\n\n")

                    # 写入部分结果历史
                    f.write("=== 部分结果历史 ===\n")
                    f.write('\n'.join(all_data['partial_results']))
                    f.write("\n\n")

                    # 写入当前显示内容
                    f.write("=== 当前显示内容 ===\n")
                    f.write(all_data['current_display'])

                # 保存SRT格式的字幕文件
                srt_path = save_path.replace('.txt', '.srt')
                try:
                    with open(srt_path, 'w', encoding='utf-8') as f:
                        # 获取所有转录数据
                        all_data = self.subtitle_widget.get_all_transcript_data()

                        # 生成SRT格式的字幕
                        for i, (text, timestamp) in enumerate(all_data['timestamped_transcript'], 1):
                            # 解析时间戳
                            h, m, s = timestamp.split(':')
                            start_time = f"00:{h}:{m},{s}00"

                            # 计算结束时间（假设每段字幕持续5秒）
                            h_end, m_end, s_end = int(h), int(m), int(s) + 5
                            if s_end >= 60:
                                s_end -= 60
                                m_end += 1
                            if m_end >= 60:
                                m_end -= 60
                                h_end += 1
                            end_time = f"00:{h_end:02d}:{m_end:02d},{s_end:02d}00"

                            # 写入SRT格式
                            f.write(f"{i}\n")
                            f.write(f"{start_time} --> {end_time}\n")
                            f.write(f"{text}\n\n")
                except Exception as e:
                    print(f"保存SRT文件错误: {e}")
                    import traceback
                    traceback.print_exc()

                # 设置保存标志
                MainWindow._has_saved_transcript = True

                # 更新状态
                self.signals.status_updated.emit(f"转录已停止并保存到: {save_path}")

                # 使用延迟确保在字幕窗口显示保存信息
                def update_subtitle_with_save_info():
                    # 获取当前转录文本
                    current_text = self.subtitle_widget.subtitle_label.text()

                    # 确保保存信息显示在最下方
                    if "转录已保存到" not in current_text:
                        # 添加保存信息
                        save_info = f"\n\n转录已停止并保存到: {save_path}"
                        self.subtitle_widget.subtitle_label.setText(current_text + save_info)

                        # 滚动到底部
                        self.subtitle_widget._scroll_to_bottom()

                # 使用较长的延迟确保信息显示
                QTimer.singleShot(500, update_subtitle_with_save_info)
            else:
                # 更新状态
                self.signals.status_updated.emit("转录已停止，但没有内容可保存")
        except Exception as e:
            print(f"保存转录文本错误: {e}")
            import traceback
            traceback.print_exc()
            # 更新状态
            self.signals.status_updated.emit("转录已停止，但保存失败")

    def _on_device_selected(self, device):
        """设备选择处理"""
        # 设置当前设备
        self.audio_processor.set_current_device(device)

        # 在状态栏显示设备信息（即使空间有限也尝试显示）
        self.signals.status_updated.emit(f"已选择设备: {device.name}")

        # 在字幕窗口显示设备信息
        device_info = f"已选择设备: {device.name}"

        # 只有在没有进行转录时才更新字幕窗口
        if not self.control_panel.is_transcribing:
            self.subtitle_widget.transcript_text = []
            self.subtitle_widget.subtitle_label.setText(device_info)
            # 滚动到顶部
            QTimer.singleShot(100, lambda: self.subtitle_widget.verticalScrollBar().setValue(0))

    @pyqtSlot()
    def _on_transcription_started(self):
        """转录开始处理"""
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，使用标准日志
                sherpa_logger = logging.getLogger(__name__)

            sherpa_logger.info("转录开始处理")

            # 更新菜单状态
            self.menu_bar.update_menu_state(is_recording=True)

            # 更新状态
            self.signals.status_updated.emit("转录已开始")

            # 清空转录文本
            self.subtitle_widget.transcript_text = []

            # 更新字幕窗口
            self.subtitle_widget.subtitle_label.setText("正在转录...")

        except Exception as e:
            logging.error(f"转录开始处理错误: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.signals.error_occurred.emit(f"转录开始处理错误: {e}")

    @pyqtSlot()
    def _on_transcription_paused(self):
        """转录暂停处理"""
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，使用标准日志
                sherpa_logger = logging.getLogger(__name__)

            sherpa_logger.info("转录暂停处理")

            # 更新状态
            self.signals.status_updated.emit("转录已暂停")

            # 更新字幕窗口
            current_text = self.subtitle_widget.subtitle_label.text()
            if current_text:
                self.subtitle_widget.subtitle_label.setText(current_text + "\n\n[转录已暂停]")

            # 滚动到底部
            QTimer.singleShot(100, self.subtitle_widget._scroll_to_bottom)

        except Exception as e:
            logging.error(f"转录暂停处理错误: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.signals.error_occurred.emit(f"转录暂停处理错误: {e}")

    @pyqtSlot()
    def _on_transcription_resumed(self):
        """转录恢复处理"""
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，使用标准日志
                sherpa_logger = logging.getLogger(__name__)

            sherpa_logger.info("转录恢复处理")

            # 更新状态
            self.signals.status_updated.emit("转录已恢复")

            # 更新字幕窗口
            current_text = self.subtitle_widget.subtitle_label.text()
            if "[转录已暂停]" in current_text:
                # 移除暂停标记
                new_text = current_text.replace("\n\n[转录已暂停]", "")
                self.subtitle_widget.subtitle_label.setText(new_text + "\n\n[转录已恢复]")
            else:
                self.subtitle_widget.subtitle_label.setText(current_text + "\n\n[转录已恢复]")

            # 滚动到底部
            QTimer.singleShot(100, self.subtitle_widget._scroll_to_bottom)

        except Exception as e:
            logging.error(f"转录恢复处理错误: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.signals.error_occurred.emit(f"转录恢复处理错误: {e}")

    @pyqtSlot()
    def _on_transcription_finished(self):
        """转录完成处理"""
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，使用标准日志
                sherpa_logger = logging.getLogger(__name__)

            sherpa_logger.info("转录完成处理开始")

            # 重置控制面板
            self.control_panel.reset()
            sherpa_logger.debug("控制面板已重置")

            # 重新启用相关菜单项
            self.menu_bar.update_menu_state(is_recording=False)
            sherpa_logger.debug("菜单状态已更新")

            # 如果已经保存过文件，则不再保存
            if hasattr(MainWindow, '_has_saved_transcript') and MainWindow._has_saved_transcript:
                sherpa_logger.info("转录已保存，跳过保存步骤")
                self.signals.status_updated.emit("转录已完成")
                return

            sherpa_logger.info("准备保存转录结果")
        except Exception as e:
            logging.error(f"转录完成处理初始化错误: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.signals.error_occurred.emit(f"转录完成处理错误: {e}")
            return

        # 保存转录文本
        try:
            # 直接获取转录文本
            if hasattr(self.subtitle_widget, 'transcript_text') and self.subtitle_widget.transcript_text:
                # 确保 transcripts 目录存在
                import os
                import time
                save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "transcripts")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 生成带时间戳的文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                # 获取当前模型类型
                model_type = self.model_manager.model_type if hasattr(self.model_manager, 'model_type') else "unknown"

                # 确定转录类型（在线或文件）
                transcription_type = "FILE" if self.is_file_mode else "ONLINE"

                # 生成文件名，包含明显的转录类型标识和自动保存标记
                filename = f"transcript_{transcription_type}_{model_type}_{timestamp}_AUTO.txt"

                # 完整的保存路径
                save_path = os.path.join(save_dir, filename)

                # 保存文件 - 使用带时间戳的转录历史记录
                with open(save_path, 'w', encoding='utf-8') as f:
                    # 获取带时间戳的转录历史记录
                    timestamped_transcript = self.subtitle_widget.get_timestamped_transcript()
                    f.write(timestamped_transcript)

                # 同时保存一个包含所有数据的调试文件
                debug_path = save_path.replace('.txt', '_debug.txt')
                with open(debug_path, 'w', encoding='utf-8') as f:
                    # 获取所有转录数据
                    all_data = self.subtitle_widget.get_all_transcript_data()

                    # 写入带时间戳的转录历史
                    f.write("=== 带时间戳的转录历史 ===\n")
                    timestamped_text = []
                    for text, timestamp in all_data['timestamped_transcript']:
                        timestamped_text.append(f"[{timestamp}] {text}")
                    f.write('\n'.join(timestamped_text))
                    f.write("\n\n")

                    # 写入完整转录历史
                    f.write("=== 完整转录历史 ===\n")
                    f.write('\n'.join(all_data['full_transcript']))
                    f.write("\n\n")

                    # 写入部分结果历史
                    f.write("=== 部分结果历史 ===\n")
                    f.write('\n'.join(all_data['partial_results']))
                    f.write("\n\n")

                    # 写入当前显示内容
                    f.write("=== 当前显示内容 ===\n")
                    f.write(all_data['current_display'])

                # 保存SRT格式的字幕文件
                srt_path = save_path.replace('.txt', '.srt')
                try:
                    with open(srt_path, 'w', encoding='utf-8') as f:
                        # 获取所有转录数据
                        all_data = self.subtitle_widget.get_all_transcript_data()

                        # 生成SRT格式的字幕
                        for i, (text, timestamp) in enumerate(all_data['timestamped_transcript'], 1):
                            # 解析时间戳
                            h, m, s = timestamp.split(':')
                            start_time = f"00:{h}:{m},{s}00"

                            # 计算结束时间（假设每段字幕持续5秒）
                            h_end, m_end, s_end = int(h), int(m), int(s) + 5
                            if s_end >= 60:
                                s_end -= 60
                                m_end += 1
                            if m_end >= 60:
                                m_end -= 60
                                h_end += 1
                            end_time = f"00:{h_end:02d}:{m_end:02d},{s_end:02d}00"

                            # 写入SRT格式
                            f.write(f"{i}\n")
                            f.write(f"{start_time} --> {end_time}\n")
                            f.write(f"{text}\n\n")
                except Exception as e:
                    print(f"保存SRT文件错误: {e}")
                    import traceback
                    traceback.print_exc()

                # 设置保存标志
                MainWindow._has_saved_transcript = True

                # 更新状态
                self.signals.status_updated.emit(f"转录已完成并保存到: {save_path}")

                # 使用延迟确保在字幕窗口显示保存信息
                def update_subtitle_with_save_info():
                    # 获取当前转录文本
                    current_text = self.subtitle_widget.subtitle_label.text()

                    # 确保保存信息显示在最下方
                    if "转录已保存到" not in current_text:
                        # 添加保存信息
                        save_info = f"\n\n转录已完成并保存到: {save_path}"
                        self.subtitle_widget.subtitle_label.setText(current_text + save_info)

                        # 滚动到底部
                        self.subtitle_widget._scroll_to_bottom()

                # 使用较长的延迟确保信息显示
                QTimer.singleShot(500, update_subtitle_with_save_info)
            else:
                # 更新状态
                self.signals.status_updated.emit("转录已完成，但没有内容可保存")
        except Exception as e:
            print(f"保存转录文本错误: {e}")
            import traceback
            traceback.print_exc()
            # 更新状态
            self.signals.status_updated.emit("转录已完成，但保存失败")

    @pyqtSlot(str)
    def _show_error(self, error_message):
        """显示错误消息"""
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，使用标准日志
                sherpa_logger = logging.getLogger(__name__)

            sherpa_logger.error(f"显示错误消息: {error_message}")

            # 显示错误对话框
            QMessageBox.critical(self, "错误", error_message)

        except Exception as e:
            logging.error(f"显示错误消息时出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # 尝试使用更简单的方式显示错误
            try:
                print(f"错误: {error_message}")
                print(f"显示错误消息时出错: {e}")
            except:
                pass

    @pyqtSlot()
    def save_transcript(self):
        """保存转录文本"""
        try:
            # 导入 Sherpa-ONNX 日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，创建一个简单的日志记录器
                class DummyLogger:
                    def debug(self, msg): print(f"DEBUG: {msg}")
                    def info(self, msg): print(f"INFO: {msg}")
                    def warning(self, msg): print(f"WARNING: {msg}")
                    def error(self, msg): print(f"ERROR: {msg}")
                sherpa_logger = DummyLogger()

            sherpa_logger.info("保存转录文本")

            # 直接获取转录文本
            if hasattr(self.subtitle_widget, 'transcript_text') and self.subtitle_widget.transcript_text:
                # 确保 transcripts 目录存在
                import os
                import time
                save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "transcripts")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 生成带时间戳的文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                # 获取当前模型类型
                model_type = self.model_manager.model_type if hasattr(self.model_manager, 'model_type') else "unknown"

                # 确定转录类型（在线或文件）
                transcription_type = "FILE" if self.is_file_mode else "ONLINE"

                # 生成文件名，包含明显的转录类型标识
                filename = f"transcript_{transcription_type}_{model_type}_{timestamp}.txt"

                # 完整的保存路径
                save_path = os.path.join(save_dir, filename)

                # 保存文件 - 使用带时间戳的转录历史记录
                with open(save_path, 'w', encoding='utf-8') as f:
                    # 获取带时间戳的转录历史记录
                    timestamped_transcript = self.subtitle_widget.get_timestamped_transcript()
                    f.write(timestamped_transcript)

                # 同时保存一个包含所有数据的调试文件
                debug_path = save_path.replace('.txt', '_debug.txt')
                with open(debug_path, 'w', encoding='utf-8') as f:
                    # 获取所有转录数据
                    all_data = self.subtitle_widget.get_all_transcript_data()

                    # 写入带时间戳的转录历史
                    f.write("=== 带时间戳的转录历史 ===\n")
                    timestamped_text = []
                    for text, timestamp in all_data['timestamped_transcript']:
                        timestamped_text.append(f"[{timestamp}] {text}")
                    f.write('\n'.join(timestamped_text))
                    f.write("\n\n")

                    # 写入完整转录历史
                    f.write("=== 完整转录历史 ===\n")
                    f.write('\n'.join(all_data['full_transcript']))
                    f.write("\n\n")

                    # 写入部分结果历史
                    f.write("=== 部分结果历史 ===\n")
                    f.write('\n'.join(all_data['partial_results']))
                    f.write("\n\n")

                    # 写入当前显示内容
                    f.write("=== 当前显示内容 ===\n")
                    f.write(all_data['current_display'])

                    # 写入引擎信息
                    f.write("\n\n=== 引擎信息 ===\n")
                    f.write(f"模型类型: {model_type}\n")
                    f.write(f"引擎类型: {self.model_manager.get_current_engine_type()}\n")
                    f.write(f"保存时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

                # 保存SRT格式的字幕文件
                srt_path = save_path.replace('.txt', '.srt')
                try:
                    with open(srt_path, 'w', encoding='utf-8') as f:
                        # 获取所有转录数据
                        all_data = self.subtitle_widget.get_all_transcript_data()

                        # 生成SRT格式的字幕
                        for i, (text, timestamp) in enumerate(all_data['timestamped_transcript'], 1):
                            # 解析时间戳
                            h, m, s = timestamp.split(':')
                            start_time = f"00:{h}:{m},{s}00"

                            # 计算结束时间（假设每段字幕持续5秒）
                            h_end, m_end, s_end = int(h), int(m), int(s) + 5
                            if s_end >= 60:
                                s_end -= 60
                                m_end += 1
                            if m_end >= 60:
                                m_end -= 60
                                h_end += 1
                            end_time = f"00:{h_end:02d}:{m_end:02d},{s_end:02d}00"

                            # 写入SRT格式
                            f.write(f"{i}\n")
                            f.write(f"{start_time} --> {end_time}\n")
                            f.write(f"{text}\n\n")
                except Exception as e:
                    sherpa_logger.error(f"保存SRT文件错误: {e}")
                    import traceback
                    traceback.print_exc()

                # 设置保存标志
                MainWindow._has_saved_transcript = True

                # 更新状态
                self.signals.status_updated.emit(f"转录已保存到: {save_path}")

                # 使用延迟确保在字幕窗口显示保存信息
                def update_subtitle_with_save_info():
                    # 获取当前转录文本
                    current_text = self.subtitle_widget.subtitle_label.text()

                    # 确保保存信息显示在最下方
                    if "转录已保存到" not in current_text:
                        # 添加保存信息
                        save_info = f"\n\n转录已保存到: {save_path}"
                        self.subtitle_widget.subtitle_label.setText(current_text + save_info)

                        # 滚动到底部
                        self.subtitle_widget._scroll_to_bottom()

                # 使用较长的延迟确保信息显示
                QTimer.singleShot(500, update_subtitle_with_save_info)

                # 返回保存路径
                return save_path
            else:
                # 更新状态
                self.signals.status_updated.emit("没有内容可保存")
                return None
        except Exception as e:
            print(f"保存转录文本错误: {e}")
            import traceback
            traceback.print_exc()
            # 更新状态
            self.signals.status_updated.emit("保存转录文本失败")
            return None

    def set_recognition_language(self, language):
        """
        设置识别语言

        Args:
            language: 语言代码
        """
        # 如果当前是文件模式，切换回系统音频模式
        if self.is_file_mode:
            self.is_file_mode = False
            if hasattr(self.control_panel, 'set_transcription_mode'):
                self.control_panel.set_transcription_mode("system")

            # 更新菜单选中状态
            self.menu_bar.transcription_menu.system_audio_action.setChecked(True)
            self.menu_bar.transcription_menu.actions['select_file'].setChecked(False)

        # 设置语言
        print(f"设置识别语言: {language}")
        self.signals.status_updated.emit(f"已设置识别语言: {language}")

        # 在字幕窗口显示语言设置信息
        language_info = f"已设置识别语言: {language}"

        # 只有在没有进行转录时才更新字幕窗口
        if not self.control_panel.is_transcribing:
            self.subtitle_widget.transcript_text = []
            info_text = f"{language_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"
            self.subtitle_widget.subtitle_label.setText(info_text)
            # 滚动到顶部
            QTimer.singleShot(100, lambda: self.subtitle_widget.verticalScrollBar().setValue(0))

    def set_asr_model(self, model_name):
        """
        设置ASR模型

        Args:
            model_name: 模型名称
        """
        # 导入 Sherpa-ONNX 日志工具
        try:
            from src.utils.sherpa_logger import sherpa_logger
        except ImportError:
            # 如果导入失败，创建一个简单的日志记录器
            class DummyLogger:
                def debug(self, msg): print(f"DEBUG: {msg}")
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
            sherpa_logger = DummyLogger()

        # 获取模型显示名称
        model_display_name = self._get_model_display_name(model_name)
        sherpa_logger.info(f"设置ASR模型: {model_name} ({model_display_name})")

        # 记录当前引擎状态
        current_engine_type = self.model_manager.get_current_engine_type()
        current_engine = type(self.model_manager.current_engine).__name__ if self.model_manager.current_engine else "None"
        sherpa_logger.info(f"当前引擎类型: {current_engine_type}")
        sherpa_logger.info(f"当前引擎: {current_engine}")

        # 更新菜单选中状态
        for key, action in self.menu_bar.model_menu.actions.items():
            if key in ['vosk_small', 'sherpa_onnx_int8', 'sherpa_onnx_std', 'sherpa_0626_int8', 'sherpa_0626_std']:
                action.setChecked(key == model_name)

        # 加载模型
        if self.model_manager.load_model(model_name):
            # 获取更新后的引擎信息
            new_engine_type = self.model_manager.get_current_engine_type()
            new_engine = type(self.model_manager.current_engine).__name__ if self.model_manager.current_engine else "None"
            sherpa_logger.info(f"新引擎类型: {new_engine_type}")
            sherpa_logger.info(f"新引擎: {new_engine}")

            # 检查加载后的引擎类型是否与请求的模型类型一致
            if new_engine_type != model_name:
                sherpa_logger.warning(f"加载后的引擎类型 {new_engine_type} 与请求的模型类型 {model_name} 不一致")
                sherpa_logger.warning(f"这可能导致功能异常，请检查模型配置和文件是否正确")

                # 尝试再次加载，确保使用正确的模型类型
                sherpa_logger.info(f"尝试再次加载模型: {model_name}")
                if not self.model_manager.load_model(model_name):
                    sherpa_logger.error(f"再次加载模型 {model_name} 失败")
                    error_msg = f"无法加载请求的模型类型: {model_name}"
                    self.signals.error_occurred.emit(error_msg)
                    return

                # 再次获取引擎信息
                new_engine_type = self.model_manager.get_current_engine_type()
                new_engine = type(self.model_manager.current_engine).__name__ if self.model_manager.current_engine else "None"
                sherpa_logger.info(f"再次加载后的引擎类型: {new_engine_type}")
                sherpa_logger.info(f"再次加载后的引擎: {new_engine}")

            # 更新状态栏
            status_msg = f"已加载ASR模型: {model_display_name} (引擎: {new_engine_type})"
            sherpa_logger.info(status_msg)
            self.signals.status_updated.emit(status_msg)

            # 在字幕窗口显示模型设置信息
            model_info = f"已设置ASR模型: {model_display_name}\n引擎类型: {new_engine_type}\n引擎实例: {new_engine}"

            # 更新字幕窗口的引擎类型
            self.subtitle_widget.current_engine_type = new_engine_type
            sherpa_logger.info(f"更新字幕窗口的引擎类型: {new_engine_type}")

            # 只有在没有进行转录时才更新字幕窗口
            if not self.control_panel.is_transcribing:
                # 保留现有文本，如果有的话
                current_text = self.subtitle_widget.subtitle_label.text()

                # 如果当前文本为空或只包含准备就绪信息，则设置新文本
                if not current_text or "准备就绪" in current_text:
                    self.subtitle_widget.transcript_text = []
                    info_text = f"{model_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"
                    self.subtitle_widget.subtitle_label.setText(info_text)
                else:
                    # 否则，将模型信息添加到当前文本的最下方
                    self.subtitle_widget.subtitle_label.setText(current_text + "\n\n" + model_info)

                # 滚动到底部，确保最新信息可见
                QTimer.singleShot(100, self.subtitle_widget._scroll_to_bottom)
        else:
            error_msg = f"加载ASR模型 {model_display_name} 失败"
            sherpa_logger.error(error_msg)
            self.signals.error_occurred.emit(error_msg)

    def _get_model_display_name(self, model_name):
        """
        获取模型的显示名称

        Args:
            model_name: 模型名称

        Returns:
            str: 模型显示名称
        """
        model_display_names = {
            'vosk_small': 'VOSK Small 模型',
            'sherpa_onnx_int8': 'Sherpa-ONNX int8量化模型',
            'sherpa_onnx_std': 'Sherpa-ONNX 标准模型',
            'sherpa_0626_int8': 'Sherpa-ONNX 2023-06-26 int8 模型',
            'sherpa_0626_std': 'Sherpa-ONNX 2023-06-26 标准模型',
            'argos': 'Argostranslate 模型',
            'opus': 'Opus-Mt-ONNX 模型'
        }

        return model_display_names.get(model_name, model_name)

    def set_rtm_model(self, model_name):
        """
        设置RTM模型

        Args:
            model_name: 模型名称
        """
        # 获取模型显示名称
        model_display_name = self._get_model_display_name(model_name)

        # 更新菜单选中状态
        for key, action in self.menu_bar.model_menu.actions.items():
            if key in ['argos', 'opus']:
                action.setChecked(key == model_name)

        # 这里是设置RTM模型的占位代码
        print(f"设置RTM模型: {model_name}")
        self.signals.status_updated.emit(f"已设置RTM模型: {model_display_name}")

        # 在字幕窗口显示模型设置信息
        model_info = f"已设置RTM模型: {model_display_name}"

        # 只有在没有进行转录时才更新字幕窗口
        if not self.control_panel.is_transcribing:
            # 保留现有文本，如果有的话
            current_text = self.subtitle_widget.subtitle_label.text()

            # 如果当前文本为空或只包含准备就绪信息，则设置新文本
            if not current_text or "准备就绪" in current_text:
                self.subtitle_widget.transcript_text = []
                info_text = f"{model_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"
                self.subtitle_widget.subtitle_label.setText(info_text)
            else:
                # 否则，将模型信息添加到当前文本的最下方
                self.subtitle_widget.subtitle_label.setText(current_text + "\n\n" + model_info)

            # 滚动到底部，确保最新信息可见
            QTimer.singleShot(100, self.subtitle_widget._scroll_to_bottom)

    def set_background_mode(self, mode):
        """
        设置背景模式

        Args:
            mode: 背景模式
        """
        self.subtitle_widget.set_background_mode(mode)
        self.signals.status_updated.emit(f"已设置背景模式: {mode}")

    def set_font_size(self, size):
        """
        设置字体大小

        Args:
            size: 字体大小
        """
        self.subtitle_widget.set_font_size(size)
        self.signals.status_updated.emit(f"已设置字体大小: {size}")

    def select_file(self):
        """打开文件选择对话框并处理选择的文件"""
        try:
            # 使用 Windows 风格的文件对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择音频/视频文件",
                "",  # 默认目录
                "媒体文件 (*.mp3 *.wav *.mp4 *.avi *.mkv *.mov);;所有文件 (*)",
                options=QFileDialog.DontUseNativeDialog
            )

            if file_path:
                self._on_file_selected(file_path)
            else:
                # 如果用户取消选择，更新状态
                self.signals.status_updated.emit("已取消文件选择")

        except Exception as e:
            print(f"选择文件错误: {e}")
            self.signals.error_occurred.emit(f"选择文件错误: {e}")

    def show_system_info(self):
        """显示系统信息"""
        # 这里是显示系统信息的占位代码
        info = "系统信息:\n"
        info += f"Python版本: {sys.version}\n"
        info += f"操作系统: {os.name}\n"

        QMessageBox.information(self, "系统信息", info)

    def check_model_directory(self):
        """检查模型目录"""
        # 检查模型目录
        model_status = self.model_manager.check_model_directory()

        # 构建状态信息
        info = "模型目录状态:\n"
        for name, exists in model_status.items():
            status = "可用" if exists else "不可用"
            info += f"{name}: {status}\n"

        QMessageBox.information(self, "模型目录", info)

    def show_asr_config_dialog(self):
        """显示ASR配置对话框"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("ASR 配置")
        layout = QVBoxLayout()

        form_layout = QFormLayout()

        # 创建标签和输入框
        enable_endpoint_label = QLabel("启用端点检测:")
        self.enable_endpoint_edit = QLineEdit(str(self.model_manager.current_engine.config.get("enable_endpoint", 1)))
        form_layout.addRow(enable_endpoint_label, self.enable_endpoint_edit)

        rule1_label = QLabel("规则1尾随静音时间 (秒):")
        self.rule1_edit = QLineEdit(str(self.model_manager.current_engine.config.get("rule1_min_trailing_silence", 3.0)))
        form_layout.addRow(rule1_label, self.rule1_edit)

        rule2_label = QLabel("规则2尾随静音时间 (秒):")
        self.rule2_edit = QLineEdit(str(self.model_manager.current_engine.config.get("rule2_min_trailing_silence", 1.5)))
        form_layout.addRow(rule2_label, self.rule2_edit)

        rule3_label = QLabel("规则3最小语音长度 (帧):")
        self.rule3_edit = QLineEdit(str(self.model_manager.current_engine.config.get("rule3_min_utterance_length", 25)))
        form_layout.addRow(rule3_label, self.rule3_edit)

        layout.addLayout(form_layout)

        # 创建保存按钮
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.save_asr_config)
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def save_asr_config(self):
        """保存ASR配置"""
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，创建一个简单的日志记录器
                class DummyLogger:
                    def debug(self, msg): print(f"DEBUG: {msg}")
                    def info(self, msg): print(f"INFO: {msg}")
                    def warning(self, msg): print(f"WARNING: {msg}")
                    def error(self, msg): print(f"ERROR: {msg}")
                sherpa_logger = DummyLogger()

            # 获取输入值
            try:
                enable_endpoint = int(self.enable_endpoint_edit.text())
                rule1_min_trailing_silence = float(self.rule1_edit.text())
                rule2_min_trailing_silence = float(self.rule2_edit.text())
                rule3_min_utterance_length = int(self.rule3_edit.text())
            except ValueError as e:
                sherpa_logger.error(f"输入值格式错误: {e}")
                self.signals.error_occurred.emit(f"输入值格式错误: {e}")
                return

            # 创建ASR配置字典
            asr_config = {
                "enable_endpoint": enable_endpoint,
                "rule1_min_trailing_silence": rule1_min_trailing_silence,
                "rule2_min_trailing_silence": rule2_min_trailing_silence,
                "rule3_min_utterance_length": rule3_min_utterance_length
            }

            sherpa_logger.debug(f"保存ASR配置: {asr_config}")

            # 检查self.config的类型
            sherpa_logger.debug(f"save_asr_config: self.config类型: {type(self.config)}")

            # 根据self.config的类型更新配置
            if hasattr(self.config, 'update_and_save'):
                # 如果是ConfigManager实例
                sherpa_logger.debug("使用ConfigManager.update_and_save方法更新配置")
                self.config.update_and_save("asr", asr_config)
            elif isinstance(self.config, dict):
                # 如果是字典，直接更新字典
                sherpa_logger.debug("使用字典方式更新配置")
                if "asr" not in self.config:
                    self.config["asr"] = {}
                self.config["asr"].update(asr_config)
                sherpa_logger.debug("配置已更新，但无法保存到文件（字典模式）")
            else:
                # 如果是其他类型，记录警告
                sherpa_logger.warning(f"未知的config类型: {type(self.config)}，无法保存ASR配置")
                return

            # 更新当前引擎的配置
            if self.model_manager and self.model_manager.current_engine:
                sherpa_logger.debug("更新当前引擎的配置")
                for key, value in asr_config.items():
                    self.model_manager.current_engine.config[key] = value
                sherpa_logger.debug(f"当前引擎配置已更新: {self.model_manager.current_engine.config}")

            # 显示成功消息
            self.signals.status_updated.emit("ASR配置已保存")
            sherpa_logger.info("ASR配置已保存")

        except Exception as e:
            print(f"保存ASR配置错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.signals.error_occurred.emit(f"保存ASR配置错误: {e}")

    # 这些方法已在文件末尾重新实现，这里删除以避免重复

    def show_usage(self):
        """显示使用说明"""
        # 这里是显示使用说明的占位代码
        QMessageBox.information(self, "使用说明", "请参阅项目文档获取详细的使用说明。")

    def show_about(self):
        """显示关于信息"""
        # 这里是显示关于信息的占位代码
        QMessageBox.information(self, "关于", "实时字幕转录应用程序\n版本: 1.0.0")

    def save_window_state(self):
        """保存窗口状态到配置文件"""
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，创建一个简单的日志记录器
                class DummyLogger:
                    def debug(self, msg): print(f"DEBUG: {msg}")
                    def info(self, msg): print(f"INFO: {msg}")
                    def warning(self, msg): print(f"WARNING: {msg}")
                    def error(self, msg): print(f"ERROR: {msg}")
                sherpa_logger = DummyLogger()

            # 导入配置管理器
            from src.utils.config_manager import config_manager

            # 获取窗口几何信息
            geometry = self.geometry()

            # 创建窗口状态字典
            window_state = {
                "pos_x": geometry.x(),
                "pos_y": geometry.y(),
                "width": geometry.width(),
                "height": geometry.height(),
                "opacity": self.windowOpacity()
            }

            # 如果有背景模式属性，也保存它
            if hasattr(self, 'background_mode'):
                window_state["background_mode"] = self.background_mode

            sherpa_logger.debug(f"保存窗口状态: {window_state}")

            # 使用config_manager保存窗口状态
            try:
                from src.utils.config_manager import config_manager
                config_manager.update_and_save("window", window_state)
                sherpa_logger.debug("窗口状态保存完成")
            except Exception as e:
                sherpa_logger.error(f"保存窗口状态时出错: {str(e)}")
                sherpa_logger.error(traceback.format_exc())

        except Exception as e:
            print(f"保存窗口状态错误: {e}")
            import traceback
            print(traceback.format_exc())

    def closeEvent(self, event):
        """
        窗口关闭事件处理

        Args:
            event: 关闭事件
        """
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，创建一个简单的日志记录器
                class DummyLogger:
                    def debug(self, msg): print(f"DEBUG: {msg}")
                    def info(self, msg): print(f"INFO: {msg}")
                    def warning(self, msg): print(f"WARNING: {msg}")
                    def error(self, msg): print(f"ERROR: {msg}")
                sherpa_logger = DummyLogger()

            # 保存窗口状态
            self.save_window_state()

            # 停止所有转录活动
            if self.is_file_mode and HAS_FILE_TRANSCRIBER and self.file_transcriber:
                sherpa_logger.info("关闭窗口时停止文件转录")
                try:
                    self.file_transcriber.stop_transcription()
                except Exception as e:
                    sherpa_logger.error(f"停止文件转录时出错: {e}")
            else:
                sherpa_logger.info("关闭窗口时停止音频捕获")
                try:
                    # 检查audio_processor是否存在
                    if hasattr(self, 'audio_processor') and self.audio_processor:
                        self.audio_processor.stop_capture()
                    else:
                        sherpa_logger.warning("audio_processor不存在，跳过停止音频捕获")
                except Exception as e:
                    sherpa_logger.error(f"停止音频捕获时出错: {e}")

            # 检查COM状态并清理（如果需要）
            # 注意：在main.py中已经注册了应用程序退出时的COM清理，
            # 所以这里不需要重复清理，避免出现COM已释放的错误
            sherpa_logger.info("关闭窗口时跳过COM清理，由主程序负责")

        except Exception as e:
            logging.error(f"关闭窗口时出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            # 接受关闭事件
            event.accept()

    def load_window_state(self):
        """从配置文件加载窗口状态"""
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，创建一个简单的日志记录器
                class DummyLogger:
                    def debug(self, msg): print(f"DEBUG: {msg}")
                    def info(self, msg): print(f"INFO: {msg}")
                    def warning(self, msg): print(f"WARNING: {msg}")
                    def error(self, msg): print(f"ERROR: {msg}")
                sherpa_logger = DummyLogger()

            # 导入配置管理器
            from src.utils.config_manager import config_manager
            sherpa_logger.debug("使用config_manager获取窗口配置")

            try:
                # 获取窗口配置
                window_config = config_manager.get_config("window", {})
                ui_config = config_manager.get_config("ui", {})
            except Exception as e:
                sherpa_logger.error(f"获取配置时出错: {str(e)}")
                sherpa_logger.error(traceback.format_exc())
                window_config = {}
                ui_config = {}

            sherpa_logger.debug(f"窗口配置: {window_config}")

            # 设置窗口位置和大小
            if all(key in window_config for key in ["pos_x", "pos_y", "width", "height"]):
                self.setGeometry(
                    window_config["pos_x"],
                    window_config["pos_y"],
                    window_config["width"],
                    window_config["height"]
                )
                sherpa_logger.debug(f"设置窗口位置和大小: {window_config['pos_x']}, {window_config['pos_y']}, {window_config['width']}, {window_config['height']}")

            # 设置背景模式
            if "background_mode" in window_config:
                self.set_background_mode(window_config["background_mode"])
                sherpa_logger.debug(f"设置背景模式: {window_config['background_mode']}")

            # 设置透明度
            if "opacity" in window_config:
                self.setWindowOpacity(window_config["opacity"])
                sherpa_logger.debug(f"设置透明度: {window_config['opacity']}")

            # 设置字体大小
            if "font_size" in ui_config:
                self.set_font_size(ui_config["font_size"])
                sherpa_logger.debug(f"设置字体大小: {ui_config['font_size']}")

            sherpa_logger.debug("窗口状态加载完成")

        except Exception as e:
            print(f"加载窗口状态错误: {e}")
            import traceback
            print(traceback.format_exc())
            # 使用默认值
            self.setGeometry(100, 100, 800, 600)
            self.setWindowOpacity(0.7)

    def _on_file_selected(self, file_path):
        """
        文件选择回调

        Args:
            file_path: 选择的文件路径
        """
        if not file_path:
            return

        try:
            # 设置文件路径和转录模式
            self.file_path = file_path
            self.is_file_mode = True

            # 更新菜单选中状态
            self.menu_bar.transcription_menu.system_audio_action.setChecked(False)
            self.menu_bar.transcription_menu.actions['select_file'].setChecked(True)

            # 更新状态
            self.signals.status_updated.emit(f"已选择文件: {os.path.basename(file_path)}")

            # 不立即更新UI，等待获取文件信息后再更新

            # 更新控制面板状态（如果方法存在）
            if hasattr(self.control_panel, 'set_transcription_mode'):
                self.control_panel.set_transcription_mode("file", os.path.basename(file_path))

            # 检查文件是否存在
            if not os.path.exists(file_path):
                self.signals.error_occurred.emit(f"文件不存在: {file_path}")
                return

            # 获取文件信息
            try:
                # 获取文件大小
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

                # 获取文件时长
                probe = subprocess.run([
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    file_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                probe_data = json.loads(probe.stdout)
                duration = float(probe_data['format']['duration'])

                # 更新状态
                self.signals.status_updated.emit(
                    f"文件信息: {os.path.basename(file_path)} ({file_size_mb:.2f}MB, {duration:.2f}秒)"
                )

                # 显示文件信息
                minutes = int(duration // 60)
                seconds = int(duration % 60)

                # 不需要获取当前字幕文本，直接设置新内容

                # 准备文件信息文本
                file_info = (
                    f"已选择文件: {os.path.basename(file_path)}\n"
                    f"文件大小: {file_size_mb:.2f}MB\n"
                    f"时长: {minutes}分{seconds}秒\n"
                    f"点击'开始转录'按钮开始处理"
                )

                # 清空之前的转录内容，确保文件信息显示在最上方
                self.subtitle_widget.transcript_text = []

                # 直接设置文件信息为唯一内容
                self.subtitle_widget.subtitle_label.setText(file_info)

                # 滚动到顶部
                QTimer.singleShot(100, lambda: self.subtitle_widget.verticalScrollBar().setValue(0))

            except Exception as e:
                logging.error(f"获取文件信息错误: {e}")
                self.signals.error_occurred.emit(f"获取文件信息错误: {e}")

        except Exception as e:
            logging.error(f"处理文件错误: {e}")
            self.signals.error_occurred.emit(f"处理文件错误: {e}")

    # 此方法已在文件末尾重新实现，这里删除以避免重复

    def _show_model_manager(self):
        """显示模型管理对话框"""
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，使用标准日志
                sherpa_logger = self.logger

            sherpa_logger.info("显示模型管理对话框")

            # 创建模型管理对话框
            dialog = ModelManagerDialog(self)

            # 连接模型变更信号
            dialog.models_changed.connect(self._on_models_changed)

            # 显示对话框
            dialog.exec_()

        except Exception as e:
            self.logger.error(f"显示模型管理对话框时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.signals.error_occurred.emit(f"显示模型管理对话框时出错: {str(e)}")

    def _on_models_changed(self):
        """模型配置变更处理"""
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，使用标准日志
                sherpa_logger = self.logger

            sherpa_logger.info("模型配置已变更，更新模型列表")

            # 更新模型菜单
            self.menu_bar.model_menu.update_models()

            # 更新状态栏
            self.signals.status_updated.emit("模型配置已更新")

        except Exception as e:
            self.logger.error(f"处理模型配置变更时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.signals.error_occurred.emit(f"处理模型配置变更时出错: {str(e)}")

    def _on_plugin_status_changed(self, plugin_id, enabled):
        """插件状态变更处理

        Args:
            plugin_id: 插件ID
            enabled: 是否启用
        """
        try:
            # 导入日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，使用标准日志
                sherpa_logger = self.logger

            sherpa_logger.info(f"插件状态变更: {plugin_id} {'启用' if enabled else '禁用'}")

            # 如果是ASR插件，可能需要重新加载模型
            from src.core.plugins import PluginManager
            plugin_manager = PluginManager()

            # 获取插件元数据
            metadata = plugin_manager.get_plugin_metadata(plugin_id)
            if metadata and metadata.get('type') == 'asr':
                sherpa_logger.info(f"ASR插件状态变更，重新加载模型列表")

                # 更新模型菜单
                self.menu_bar.model_menu.update_models()

                # 更新状态栏
                self.signals.status_updated.emit(f"ASR插件 {plugin_id} 已{'启用' if enabled else '禁用'}")

        except Exception as e:
            self.logger.error(f"处理插件状态变更时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.signals.error_occurred.emit(f"处理插件状态变更时出错: {str(e)}")

    # 以下是新增的方法，用于支持新的菜单结构

    def set_language_mode(self, mode):
        """
        设置语言模式

        Args:
            mode: 语言模式，可选值为"en"、"zh"、"auto"
        """
        try:
            self.logger.info(f"设置语言模式: {mode}")

            # 更新配置
            self.config_manager.set_config(mode, "recognition", "language_mode")
            self.config_manager.save_config("main")

            # 更新状态栏
            self.signals.status_updated.emit(f"已设置语言模式: {self._get_language_mode_display(mode)}")

            # 在字幕窗口显示语言设置信息
            language_info = f"已设置识别语言: {self._get_language_mode_display(mode)}"

            # 只有在没有进行转录时才更新字幕窗口
            if hasattr(self.control_panel, 'is_transcribing') and not self.control_panel.is_transcribing:
                self.subtitle_widget.transcript_text = []
                info_text = f"{language_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"
                self.subtitle_widget.subtitle_label.setText(info_text)
                # 滚动到顶部
                QTimer.singleShot(100, lambda: self.subtitle_widget.verticalScrollBar().setValue(0))
        except Exception as e:
            self.logger.error(f"设置语言模式时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.signals.status_updated.emit(f"设置语言模式失败: {str(e)}")

    def _get_language_mode_display(self, mode):
        """获取语言模式显示名称"""
        if mode == "en":
            return "英文识别"
        elif mode == "zh":
            return "中文识别"
        elif mode == "auto":
            return "自动识别"
        return mode

    def set_audio_mode(self, mode):
        """
        设置音频模式

        Args:
            mode: 音频模式，可选值为"system"、"file"
        """
        try:
            self.logger.info(f"设置音频模式: {mode}")

            # 更新配置
            self.config_manager.set_config(mode, "recognition", "audio_mode")
            self.config_manager.save_config("main")

            # 更新状态栏
            self.signals.status_updated.emit(f"已设置音频模式: {self._get_audio_mode_display(mode)}")

            # 如果是文件模式，打开文件选择对话框
            if mode == "file":
                self.select_file()
            else:
                # 系统音频模式
                self.is_file_mode = False

                # 更新控制面板
                if hasattr(self.control_panel, 'set_transcription_mode'):
                    self.control_panel.set_transcription_mode("system")

                # 在字幕窗口显示模式设置信息
                mode_info = f"已设置音频模式: {self._get_audio_mode_display(mode)}"

                # 只有在没有进行转录时才更新字幕窗口
                if hasattr(self.control_panel, 'is_transcribing') and not self.control_panel.is_transcribing:
                    # 保留现有文本，如果有的话
                    current_text = self.subtitle_widget.subtitle_label.text()

                    # 如果当前文本为空或只包含准备就绪信息，则设置新文本
                    if not current_text or "准备就绪" in current_text:
                        self.subtitle_widget.transcript_text = []
                        info_text = f"{mode_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"
                        self.subtitle_widget.subtitle_label.setText(info_text)
                    else:
                        # 否则，将模式信息添加到当前文本的最下方
                        self.subtitle_widget.subtitle_label.setText(current_text + "\n\n" + mode_info)

                    # 滚动到底部，确保最新信息可见
                    QTimer.singleShot(100, self.subtitle_widget._scroll_to_bottom)
        except Exception as e:
            self.logger.error(f"设置音频模式时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.signals.status_updated.emit(f"设置音频模式失败: {str(e)}")

    def _get_audio_mode_display(self, mode):
        """获取音频模式显示名称"""
        if mode == "system":
            return "系统音频模式"
        elif mode == "file":
            return "文件音频模式"
        return mode

    def toggle_speaker_identification(self, enabled):
        """
        切换说话人识别功能

        Args:
            enabled: 是否启用
        """
        try:
            self.logger.info(f"切换说话人识别功能: {enabled}")

            # 更新配置
            self.config_manager.set_config(enabled, "recognition", "speaker_identification")
            self.config_manager.save_config("main")

            # 更新状态栏
            status = "启用" if enabled else "禁用"
            self.signals.status_updated.emit(f"已{status}说话人识别功能")

            # 在字幕窗口显示功能设置信息
            feature_info = f"已{status}说话人识别功能"

            # 只有在没有进行转录时才更新字幕窗口
            if hasattr(self.control_panel, 'is_transcribing') and not self.control_panel.is_transcribing:
                # 保留现有文本，如果有的话
                current_text = self.subtitle_widget.subtitle_label.text()

                # 如果当前文本为空或只包含准备就绪信息，则设置新文本
                if not current_text or "准备就绪" in current_text:
                    self.subtitle_widget.transcript_text = []
                    info_text = f"{feature_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"
                    self.subtitle_widget.subtitle_label.setText(info_text)
                else:
                    # 否则，将功能信息添加到当前文本的最下方
                    self.subtitle_widget.subtitle_label.setText(current_text + "\n\n" + feature_info)

                # 滚动到底部，确保最新信息可见
                QTimer.singleShot(100, self.subtitle_widget._scroll_to_bottom)
        except Exception as e:
            self.logger.error(f"切换说话人识别功能时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.signals.status_updated.emit(f"切换说话人识别功能失败: {str(e)}")

    def search_model_documentation(self):
        """搜索模型文档"""
        try:
            self.logger.info("搜索模型文档")

            # 打开浏览器搜索模型文档
            import webbrowser
            webbrowser.open("https://github.com/alphacep/vosk-api/wiki")

            # 更新状态栏
            self.signals.status_updated.emit("已打开模型文档搜索页面")
        except Exception as e:
            self.logger.error(f"搜索模型文档时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.signals.status_updated.emit(f"搜索模型文档失败: {str(e)}")

    def refresh_models(self):
        """刷新模型列表"""
        try:
            self.logger.info("刷新模型列表")

            # 重新加载模型列表
            # 由于ASRModelManager没有refresh_models方法，我们使用其他方式刷新
            # 例如，重新加载默认模型
            default_model = self.config_manager.get_default_model()
            if self.model_manager.load_model(default_model):
                self.logger.info(f"已重新加载默认模型: {default_model}")

            # 如果菜单有更新模型的方法，调用它
            if hasattr(self.menu_bar, 'model_menu') and hasattr(self.menu_bar.model_menu, 'update_models'):
                self.menu_bar.model_menu.update_models()
                self.logger.info("已更新模型菜单")

            # 更新状态栏
            self.signals.status_updated.emit("模型列表已刷新")
        except Exception as e:
            self.logger.error(f"刷新模型列表时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.signals.status_updated.emit(f"刷新模型列表失败: {str(e)}")

    def refresh_plugins(self):
        """刷新插件"""
        try:
            self.logger.info("刷新插件")

            # 刷新插件
            # 由于PluginManager可能没有refresh_plugins方法，我们使用其他方式刷新
            # 例如，重新加载插件
            from src.core.plugins import PluginManager
            plugin_manager = PluginManager()

            # 如果有reload_plugins方法，调用它
            if hasattr(plugin_manager, 'reload_plugins'):
                plugin_manager.reload_plugins()
                self.logger.info("已重新加载插件")

            # 更新状态栏
            self.signals.status_updated.emit("插件已刷新")
        except Exception as e:
            self.logger.error(f"刷新插件时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.signals.status_updated.emit(f"刷新插件失败: {str(e)}")

    def _show_plugin_manager(self):
        """显示插件管理器对话框"""
        try:
            self.logger.info("显示插件管理器对话框")

            # 创建并显示插件管理器对话框
            from src.ui.dialogs.plugin_manager_dialog import PluginManagerDialog
            dialog = PluginManagerDialog(self)
            dialog.exec_()

            # 对话框关闭后，刷新插件
            self.refresh_plugins()
        except Exception as e:
            self.logger.error(f"显示插件管理器对话框时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.signals.error_occurred.emit(f"显示插件管理器对话框失败: {str(e)}")