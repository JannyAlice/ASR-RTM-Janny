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
from PyQt5.QtCore import Qt, pyqtSlot

from src.ui.menu.main_menu import MainMenu
from src.ui.widgets.subtitle_widget import SubtitleWidget
from src.ui.widgets.control_panel import ControlPanel
from src.core.signals import TranscriptionSignals
from src.core.asr.model_manager import ASRModelManager
from src.core.audio.audio_processor import AudioProcessor
from src.utils.config_manager import config_manager
from src.utils.com_handler import com_handler

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

        # 加载配置
        self.config = config_manager

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

        # 加载模型
        if not self.model_manager.load_model(default_model):
            self.signals.error_occurred.emit(f"加载默认模型 {default_model} 失败")

    def _load_audio_devices(self):
        """加载音频设备"""
        # 获取音频设备
        devices = self.audio_processor.get_audio_devices()

        # 设置设备列表
        self.control_panel.set_devices(devices)

        # 如果有设备，选择第一个
        if devices:
            self.audio_processor.set_current_device(devices[0])

    @pyqtSlot()
    def _on_start_clicked(self):
        """开始按钮点击处理"""
        # 创建识别器
        recognizer = self.model_manager.create_recognizer()
        if not recognizer:
            self.signals.error_occurred.emit("创建识别器失败")
            self.control_panel.reset()
            return

        # 开始捕获音频
        if not self.audio_processor.start_capture(recognizer):
            self.signals.error_occurred.emit("开始音频捕获失败")
            self.control_panel.reset()
            return

        # 更新状态
        self.signals.status_updated.emit("正在转录...")

    @pyqtSlot()
    def _on_stop_clicked(self):
        """停止按钮点击处理"""
        # 停止捕获音频
        if not self.audio_processor.stop_capture():
            self.signals.error_occurred.emit("停止音频捕获失败")
            return

        # 保存转录文本
        try:
            save_path = self.subtitle_widget.save_transcript()
            if save_path:
                # 更新状态
                self.signals.status_updated.emit(f"转录已停止并保存到: {save_path}")
            else:
                # 更新状态
                self.signals.status_updated.emit("转录已停止，但没有内容可保存")
        except Exception as e:
            print(f"保存转录文本错误: {e}")
            # 更新状态
            self.signals.status_updated.emit("转录已停止，但保存失败")

    def _on_device_selected(self, device):
        """设备选择处理"""
        # 设置当前设备
        self.audio_processor.set_current_device(device)
        # 在内容窗口显示设备信息
        self.subtitle_widget.update_text(f"已选择设备: {device.name}")
        self.signals.status_updated.emit(f"已选择设备: {device.name}")

    @pyqtSlot()
    def _on_transcription_finished(self):
        """转录完成处理"""
        # 重置控制面板
        self.control_panel.reset()

        # 保存转录文本
        try:
            save_path = self.subtitle_widget.save_transcript()
            if save_path:
                # 更新状态
                self.signals.status_updated.emit(f"转录已完成并保存到: {save_path}")
            else:
                # 更新状态
                self.signals.status_updated.emit("转录已完成，但没有内容可保存")
        except Exception as e:
            print(f"保存转录文本错误: {e}")
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
        # 这里是设置语言的占位代码
        print(f"设置识别语言: {language}")
        self.signals.status_updated.emit(f"已设置识别语言: {language}")

    def set_asr_model(self, model_name):
        """
        设置ASR模型

        Args:
            model_name: 模型名称
        """
        # 加载模型
        if self.model_manager.load_model(model_name):
            self.signals.status_updated.emit(f"已加载ASR模型: {model_name}")
        else:
            self.signals.error_occurred.emit(f"加载ASR模型 {model_name} 失败")

    def _prepare_file_transcription(self, file_path, duration, file_size_mb):
        """
        准备文件转录

        Args:
            file_path: 文件路径
            duration: 文件时长(秒)
            file_size_mb: 文件大小(MB)
        """
        # 根据文件大小选择转录方法
        large_file_threshold = 20  # MB

        if file_size_mb > large_file_threshold:
            self.signals.status_updated.emit(
                f"检测到大型文件: {file_size_mb:.2f}MB，准备使用增强模式处理"
            )
        else:
            self.signals.status_updated.emit(
                f"检测到普通文件: {file_size_mb:.2f}MB，准备使用标准模式处理"
            )

    def set_rtm_model(self, model_name):
        """
        设置RTM模型

        Args:
            model_name: 模型名称
        """
        # 这里是设置RTM模型的占位代码
        print(f"设置RTM模型: {model_name}")
        self.signals.status_updated.emit(f"已设置RTM模型: {model_name}")

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

            # 停止音频捕获
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
            self.file_path = file_path
            self.signals.status_updated.emit(f"已选择文件: {os.path.basename(file_path)}")
            self.subtitle_widget.update_text(f"已选择文件: {os.path.basename(file_path)}")

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

                self.signals.status_updated.emit(
                    f"文件信息: {os.path.basename(file_path)} ({file_size_mb:.2f}MB, {duration:.2f}秒)"
                )

                # 准备转录
                self._prepare_file_transcription(file_path, duration, file_size_mb)

            except Exception as e:
                logging.error(f"获取文件信息错误: {e}")
                self.signals.error_occurred.emit(f"获取文件信息错误: {e}")

        except Exception as e:
            logging.error(f"处理文件错误: {e}")
            self.signals.error_occurred.emit(f"处理文件错误: {e}")