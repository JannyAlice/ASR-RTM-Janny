"""
控制面板模块
负责提供用户控制界面元素
"""
import traceback
from PyQt5.QtWidgets import (QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
                            QProgressBar, QLabel, QComboBox)
from PyQt5.QtCore import pyqtSignal, pyqtSlot

from src.utils.config_manager import config_manager
from src.utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)

class ControlPanel(QWidget):
    """控制面板类"""

    # 定义信号
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    device_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        """
        初始化控制面板

        Args:
            parent: 父控件
        """
        super().__init__(parent)

        # 加载配置
        self.config_manager = config_manager
        self.colors_config = self.config_manager.get_ui_config('colors', default={})
        self.styles_config = self.config_manager.get_ui_config('styles', default={})

        # 转录模式标志
        self.transcription_mode = "system"  # 默认为系统音频模式
        self.current_file = None

        # 创建控件
        self._create_widgets()

        # 创建布局
        self._create_layout()

        # 连接信号
        self._connect_signals()

        # 应用样式
        self._apply_styles()

    def _create_widgets(self):
        """创建控件"""
        # 创建按钮
        self.start_button = QPushButton("开始转录", self)
        self.record_button = QPushButton("开始录音", self)
        self.is_transcribing = False

        # 创建设备选择下拉框
        self.device_combo = QComboBox(self)
        self.device_combo.setPlaceholderText("选择音频设备")

        # 创建进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v/%m")

    def _create_layout(self):
        """创建布局"""
        # 创建水平布局
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)  # 设置控件间距

        # 按新顺序添加控件
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.record_button)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.device_combo)

        # 设置进度条可伸缩
        controls_layout.setStretchFactor(self.progress_bar, 1)

        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(controls_layout)

        # 设置布局
        self.setLayout(main_layout)

    def _connect_signals(self):
        """连接信号"""
        self.start_button.clicked.connect(self._on_transcribe_clicked)
        self.record_button.clicked.connect(self._on_record_clicked)
        self.device_combo.currentTextChanged.connect(self._on_device_changed)

    def _apply_styles(self):
        """应用样式"""
        try:
            # 获取颜色配置
            button_start_color = self.colors_config.get('button_start', 'rgba(50, 150, 50, 200)')
            button_record_color = self.colors_config.get('button_record', 'rgba(50, 50, 150, 200)')

            logger.debug(f"按钮颜色配置: start={button_start_color}, record={button_record_color}")

            # 获取样式配置
            button_padding = self.styles_config.get('button_padding', '8px 20px')
            button_border_radius = self.styles_config.get('button_border_radius', 4)

            logger.debug(f"按钮样式配置: padding={button_padding}, border_radius={button_border_radius}")

            # 设置按钮样式
            self.start_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {button_start_color};
                    color: white;
                    border: none;
                    padding: {button_padding};
                    border-radius: {button_border_radius}px;
                }}
                QPushButton:hover {{
                    background-color: rgba(60, 170, 60, 220);
                }}
                QPushButton:pressed {{
                    background-color: rgba(40, 130, 40, 220);
                }}
                QPushButton:disabled {{
                    background-color: rgba(50, 150, 50, 100);
                    color: rgba(255, 255, 255, 120);
                }}
            """)

            self.record_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {button_record_color};
                    color: white;
                    border: none;
                    padding: {button_padding};
                    border-radius: {button_border_radius}px;
                }}
                QPushButton:hover {{
                    background-color: rgba(60, 60, 170, 220);
                }}
                QPushButton:pressed {{
                    background-color: rgba(40, 40, 130, 220);
                }}
                QPushButton:disabled {{
                    background-color: rgba(50, 50, 150, 100);
                    color: rgba(255, 255, 255, 120);
                }}
            """)

            # 设置进度条样式
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #555;
                    border-radius: 3px;
                    text-align: center;
                    background-color: rgba(40, 40, 40, 180);
                    font-weight: bold;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: rgba(74, 144, 226, 180);
                    width: 10px;
                    margin: 0.5px;
                }
            """)

            # 设置设备下拉列表样式
            self.device_combo.setStyleSheet("""
                QComboBox {
                    border: 1px solid #555;
                    border-radius: 3px;
                    padding: 1px 18px 1px 3px;
                    background-color: rgba(40, 40, 40, 180);
                    color: white;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 15px;
                    border-left-width: 1px;
                    border-left-color: #555;
                    border-left-style: solid;
                    border-top-right-radius: 3px;
                    border-bottom-right-radius: 3px;
                }
                QComboBox::down-arrow {
                    image: url(:/images/dropdown.png);
                }
                QComboBox QAbstractItemView {
                    border: 1px solid #555;
                    selection-background-color: rgba(74, 144, 226, 180);
                    background-color: rgba(40, 40, 40, 180);
                    color: white;
                }
                QComboBox QAbstractItemView::item {
                    min-height: 20px;
                }
                QComboBox QAbstractItemView::item:selected {
                    background-color: rgba(74, 144, 226, 180);
                }
            """)

            logger.debug("控制面板样式应用完成")
        except Exception as e:
            logger.error(f"应用样式时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def _on_transcribe_clicked(self):
        """转录按钮点击处理"""
        if not self.is_transcribing:
            # 开始转录模式
            if hasattr(self, 'transcription_mode') and self.transcription_mode == "file":
                self.start_button.setText("停止文件转录")
            else:
                self.start_button.setText("停止转录")

            self.device_combo.setEnabled(False)
            self.record_button.setEnabled(False)
            self.start_clicked.emit()
        else:
            # 停止转录模式
            if hasattr(self, 'transcription_mode') and self.transcription_mode == "file":
                self.start_button.setText("开始转录文件")
            else:
                self.start_button.setText("开始转录")

            # 只有在文件模式时才禁用设备选择和录音按钮
            if hasattr(self, 'transcription_mode'):
                self.device_combo.setEnabled(self.transcription_mode != "file")
                self.record_button.setEnabled(self.transcription_mode != "file")
            else:
                self.device_combo.setEnabled(True)
                self.record_button.setEnabled(True)

            self.stop_clicked.emit()

        self.is_transcribing = not self.is_transcribing

    def _on_record_clicked(self):
        """录音按钮点击处理(占位功能)"""
        self.record_button.setText("停止录音" if self.record_button.text() == "开始录音" else "开始录音")

    def _on_device_changed(self, device_name):
        """设备选择变更处理"""
        self.device_changed.emit(device_name)

    def set_devices(self, devices):
        """
        设置设备列表

        Args:
            devices: 设备列表
        """
        self.device_combo.clear()
        for device in devices:
            self.device_combo.addItem(device.name, device)

    @pyqtSlot(int, str)
    def update_progress(self, value, text=None):
        """
        更新进度条

        Args:
            value: 进度值
            text: 进度文本
        """
        self.progress_bar.setValue(value)
        if text:
            self.progress_bar.setFormat(text)

    @pyqtSlot(str)
    def update_status(self, status):
        """
        更新状态信息并转发到字幕区域
        Args:
            status: 状态文本
        """
        # 安全访问信号
        parent = self.parent()
        if hasattr(parent, 'signals') and hasattr(parent.signals, 'status_updated'):
            parent.signals.status_updated.emit(status)
        else:
            print(f"状态更新: {status}")  # 备用日志输出

    def reset(self):
        """重置控制面板状态"""
        self.start_button.setEnabled(True)
        self.start_button.setText("开始转录")
        self.record_button.setEnabled(True)
        self.record_button.setText("开始录音")
        self.device_combo.setEnabled(True)
        self.is_transcribing = False
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - %v/%m")

    def set_transcription_mode(self, mode, file_name=None):
        """
        设置转录模式

        Args:
            mode: 转录模式 ('system' 或 'file')
            file_name: 文件名（仅在文件模式下使用）
        """
        self.transcription_mode = mode
        self.current_file = file_name

        if mode == "file":
            # 文件转录模式
            self.start_button.setText("开始转录文件")
            self.device_combo.setEnabled(False)
            self.record_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("准备转录文件...")
        else:
            # 系统音频模式
            self.start_button.setText("开始转录")
            self.device_combo.setEnabled(True)
            self.record_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p% - %v/%m")
