"""
主窗口模块
负责创建和管理应用程序的主窗口
"""
import os
import sys
import logging
import subprocess
import json
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot, QTimer

from src.ui.menu.main_menu import MainMenu
from src.ui.widgets.subtitle_widget import SubtitleWidget
from src.ui.widgets.control_panel import ControlPanel
from src.core.signals import TranscriptionSignals
from src.core.asr.model_manager import ASRModelManager
from src.core.audio.audio_processor import AudioProcessor
from src.utils.config_manager import config_manager
from src.utils.com_handler import com_handler

# 条件导入 FileTranscriber
try:
    from src.core.audio.file_transcriber import FileTranscriber
    HAS_FILE_TRANSCRIBER = True
except ImportError:
    HAS_FILE_TRANSCRIBER = False

class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        """初始化主窗口"""
        super().__init__()

        # 初始化COM
        com_handler.initialize_com()

        # 创建信号
        self.signals = TranscriptionSignals()

        # 创建模型管理器
        self.model_manager = ASRModelManager()

        # 创建音频处理器
        self.audio_processor = AudioProcessor(self.signals)

        # 创建文件转录器（如果可用）
        self.file_transcriber = None
        if HAS_FILE_TRANSCRIBER:
            self.file_transcriber = FileTranscriber(self.signals)

        # 加载配置
        self.config = config_manager

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
        # 获取窗口配置
        window_config = self.config.get_config('window', {})
        opacity = window_config.get('opacity', 0.9)

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

    def _connect_signals(self):
        """连接信号"""
        # 连接转录信号
        self.signals.new_text.connect(self.subtitle_widget.update_text)
        self.signals.progress_updated.connect(self.control_panel.update_progress)
        self.signals.status_updated.connect(self.control_panel.update_status)
        self.signals.error_occurred.connect(self._show_error)
        self.signals.transcription_finished.connect(self._on_transcription_finished)

        # 连接控制面板信号
        self.control_panel.start_clicked.connect(self._on_start_clicked)
        self.control_panel.stop_clicked.connect(self._on_stop_clicked)

    def _load_default_model(self):
        """加载默认模型"""
        # 获取默认模型
        transcription_config = self.config.get_config('transcription', {})
        default_model = transcription_config.get('default_model', 'vosk')

        # 更新菜单选中状态
        for key, action in self.menu_bar.model_menu.actions.items():
            if key in ['vosk', 'sherpa_int8', 'sherpa_std']:
                action.setChecked(key == default_model)

        # 加载模型
        if self.model_manager.load_model(default_model):
            # 获取模型显示名称
            model_display_name = self._get_model_display_name(default_model)

            # 更新状态栏
            self.signals.status_updated.emit(f"已加载ASR模型: {model_display_name}")

            # 在字幕窗口显示初始信息
            model_info = f"已加载ASR模型: {model_display_name}"
            info_text = f"{model_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"

            # 设置字幕窗口文本
            self.subtitle_widget.transcript_text = []
            self.subtitle_widget.subtitle_label.setText(info_text)
        else:
            self.signals.error_occurred.emit(f"加载默认模型 {default_model} 失败")

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

        # 获取当前模型类型
        model_type = self.model_manager.model_type
        sherpa_logger.info(f"当前模型类型: {model_type}")

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

            # 检查当前引擎是否支持文件转录
            current_engine_type = self.model_manager.get_current_engine_type()
            sherpa_logger.info(f"当前引擎类型: {current_engine_type}")

            if current_engine_type and current_engine_type.startswith("sherpa"):
                # 使用 Sherpa-ONNX 模型直接转录文件
                sherpa_logger.info("使用 Sherpa-ONNX 模型直接转录文件")

                # 使用 model_manager 实例作为 recognizer
                recognizer = self.model_manager
                sherpa_logger.info(f"使用 model_manager 实例作为 recognizer: {type(recognizer)}")

                # 开始文件转录
                sherpa_logger.info(f"开始文件转录: {self.file_path}")
                if not self.file_transcriber.start_transcription(self.file_path, recognizer):
                    error_msg = "开始文件转录失败"
                    sherpa_logger.error(error_msg)
                    self.signals.error_occurred.emit(error_msg)
                    self.control_panel.reset()
                    return

                # 更新状态
                status_msg = f"正在使用 Sherpa-ONNX 转录文件: {os.path.basename(self.file_path)}..."
                sherpa_logger.info(status_msg)
                self.signals.status_updated.emit(status_msg)
            else:
                # 使用 Vosk 模型转录文件
                sherpa_logger.info("使用 Vosk 模型转录文件")

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

                # 开始文件转录
                sherpa_logger.info(f"开始文件转录: {self.file_path}")
                if not self.file_transcriber.start_transcription(self.file_path, recognizer):
                    error_msg = "开始文件转录失败"
                    sherpa_logger.error(error_msg)
                    self.signals.error_occurred.emit(error_msg)
                    self.control_panel.reset()
                    return

                # 更新状态
                status_msg = f"正在使用 Vosk 转录文件: {os.path.basename(self.file_path)}..."
                sherpa_logger.info(status_msg)
                self.signals.status_updated.emit(status_msg)

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
            engine_type = getattr(recognizer, 'engine_type', None)
            sherpa_logger.info(f"识别器引擎类型: {engine_type}")

            # 开始系统音频捕获
            sherpa_logger.info("开始系统音频捕获")
            if not self.audio_processor.start_capture(recognizer):
                error_msg = "开始音频捕获失败"
                sherpa_logger.error(error_msg)
                self.signals.error_occurred.emit(error_msg)
                self.control_panel.reset()
                return

            # 更新状态
            status_msg = "正在转录系统音频..."
            sherpa_logger.info(status_msg)
            self.signals.status_updated.emit(status_msg)

            # 禁用相关菜单项
            self.menu_bar.update_menu_state(is_recording=True)

    # 用于跟踪是否已保存文件
    _has_saved_transcript = False

    @pyqtSlot()
    def _on_stop_clicked(self):
        """停止按钮点击处理"""
        # 重置保存标志
        MainWindow._has_saved_transcript = False

        # 根据当前模式执行不同的停止功能
        if self.is_file_mode and HAS_FILE_TRANSCRIBER and self.file_transcriber:
            # 文件转录模式
            if not self.file_transcriber.stop_transcription():
                self.signals.error_occurred.emit("停止文件转录失败")
                return
        else:
            # 系统音频模式
            if not self.audio_processor.stop_capture():
                self.signals.error_occurred.emit("停止音频捕获失败")
                return

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
                filename = f"transcript_{timestamp}.txt"

                # 完整的保存路径
                save_path = os.path.join(save_dir, filename)

                # 保存文件
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.subtitle_widget.transcript_text))

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
    def _on_transcription_finished(self):
        """转录完成处理"""
        # 重置控制面板
        self.control_panel.reset()

        # 重新启用相关菜单项
        self.menu_bar.update_menu_state(is_recording=False)

        # 如果已经保存过文件，则不再保存
        if MainWindow._has_saved_transcript:
            self.signals.status_updated.emit("转录已完成")
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
                filename = f"transcript_{timestamp}.txt"

                # 完整的保存路径
                save_path = os.path.join(save_dir, filename)

                # 保存文件
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.subtitle_widget.transcript_text))

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
        QMessageBox.critical(self, "错误", error_message)

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
        # 获取模型显示名称
        model_display_name = self._get_model_display_name(model_name)

        # 更新菜单选中状态
        for key, action in self.menu_bar.model_menu.actions.items():
            if key in ['vosk', 'sherpa_int8', 'sherpa_std']:
                action.setChecked(key == model_name)

        # 加载模型
        if self.model_manager.load_model(model_name):
            # 更新状态栏
            self.signals.status_updated.emit(f"已加载ASR模型: {model_display_name}")

            # 在字幕窗口显示模型设置信息
            model_info = f"已设置ASR模型: {model_display_name}"

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
            self.signals.error_occurred.emit(f"加载ASR模型 {model_display_name} 失败")

    def _get_model_display_name(self, model_name):
        """
        获取模型的显示名称

        Args:
            model_name: 模型名称

        Returns:
            str: 模型显示名称
        """
        model_display_names = {
            'vosk': 'VOSK Small 模型',
            'sherpa_int8': 'Sherpa-ONNX int8量化模型',
            'sherpa_std': 'Sherpa-ONNX 标准模型',
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

    def search_model_documentation(self):
        """搜索模型文档"""
        # 这里是搜索模型文档的占位代码
        QMessageBox.information(self, "模型文档", "请访问项目文档获取更多信息。")

    def toggle_speaker_identification(self, enabled):
        """
        切换说话人识别

        Args:
            enabled: 是否启用
        """
        # 这里是切换说话人识别的占位代码
        status = "启用" if enabled else "禁用"
        self.signals.status_updated.emit(f"已{status}说话人识别")

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

            # 更新配置
            self.config.update_and_save("window", window_state)

        except Exception as e:
            logging.error(f"保存窗口状态错误: {e}")

    def closeEvent(self, event):
        """
        窗口关闭事件处理

        Args:
            event: 关闭事件
        """
        try:
            # 保存窗口状态
            self.save_window_state()

            # 停止所有转录活动
            if self.is_file_mode and HAS_FILE_TRANSCRIBER and self.file_transcriber:
                self.file_transcriber.stop_transcription()
            else:
                self.audio_processor.stop_capture()

            # 释放COM
            com_handler.uninitialize_com()

        except Exception as e:
            logging.error(f"关闭窗口时出错: {e}")
        finally:
            # 接受关闭事件
            event.accept()

    def load_window_state(self):
        """从配置文件加载窗口状态"""
        try:
            # 获取窗口配置
            window_config = self.config.get_config("window", {})

            # 设置窗口位置和大小
            if all(key in window_config for key in ["pos_x", "pos_y", "width", "height"]):
                self.setGeometry(
                    window_config["pos_x"],
                    window_config["pos_y"],
                    window_config["width"],
                    window_config["height"]
                )

            # 设置背景模式
            if "background_mode" in window_config:
                self.set_background_mode(window_config["background_mode"])

            # 设置透明度
            if "opacity" in window_config:
                self.setWindowOpacity(window_config["opacity"])

            # 获取UI配置
            ui_config = self.config.get_config("ui", {})

            # 设置字体大小
            if "font_size" in ui_config:
                self.set_font_size(ui_config["font_size"])

        except Exception as e:
            logging.error(f"加载窗口状态错误: {e}")
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