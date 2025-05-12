"""
主窗口模块
负责创建和管理应用程序的主窗口
"""
import os
import sys
import logging
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot

from src.ui.menu.main_menu import MainMenu
from src.ui.widgets.subtitle_widget import SubtitleWidget
from src.ui.widgets.control_panel import ControlPanel
from src.core.signals import TranscriptionSignals
from src.core.asr.model_manager_new import ASRModelManager  # 使用新的 ASRModelManager
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

        # 设置窗口透明度
        self.setWindowOpacity(opacity)

        # 设置窗口置顶
        if window_config.get('always_on_top', True):
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

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
        self.control_panel.device_changed.connect(self._on_device_changed)

    def _load_default_model(self):
        """加载默认模型"""
        # 获取默认模型
        default_model = self.model_manager.get_default_model()
        if not default_model:
            self.signals.error_occurred.emit("未找到默认模型")
            return

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

        # 更新状态
        self.signals.status_updated.emit("转录已停止")

    @pyqtSlot(str)
    def _on_device_changed(self, device_name):
        """设备选择变更处理"""
        # 查找设备
        devices = self.audio_processor.get_audio_devices()
        for device in devices:
            if device.name == device_name:
                # 设置当前设备
                self.audio_processor.set_current_device(device)
                self.signals.status_updated.emit(f"已选择设备: {device.name}")
                break

    @pyqtSlot()
    def _on_transcription_finished(self):
        """转录完成处理"""
        # 重置控制面板
        self.control_panel.reset()

        # 更新状态
        self.signals.status_updated.emit("转录已完成")

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
        """选择文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频/视频文件",
            "",
            "音频文件 (*.wav *.mp3 *.ogg);;视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)")

        if file_path:
            self.signals.status_updated.emit(f"已选择文件: {file_path}")
            # 这里是处理文件的占位代码
            # 实际实现需要调用文件转录功能

    def show_system_info(self):
        """显示系统信息"""
        # 这里是显示系统信息的占位代码
        info = "系统信息:\n"
        info += f"Python版本: {sys.version}\n"
        info += f"操作系统: {os.name}\n"

        QMessageBox.information(self, "系统信息", info)

    def check_model_directory(self):
        """检查模型目录"""
        # 获取可用模型
        available_models = self.model_manager.get_available_models()

        # 构建状态信息
        info = "模型状态:\n"
        for name, enabled in available_models.items():
            status = "可用" if enabled else "不可用"
            info += f"{name}: {status}\n"

        QMessageBox.information(self, "模型状态", info)

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