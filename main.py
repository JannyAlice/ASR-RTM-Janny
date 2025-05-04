#!/usr/bin/env python3
"""
实时语音转录应用程序
主要功能：
1. 实时语音识别（ASR）
2. 字幕显示
3. 系统音频捕获
4. 配置文件管理
"""

# 在导入任何库之前设置环境变量，防止COM初始化冲突
import os
os.environ["PYTHONCOM_INITIALIZE"] = "0"  # 禁止 pythoncom 自动初始化
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"  # 禁用高DPI缩放

# 导入必要的库
import sys
from PyQt5.QtWidgets import QApplication

# 导入自定义模块
from src.utils.sherpa_logger import sherpa_logger
from src.core.asr.model_manager import ASRModelManager
from src.utils.config_manager import load_config
from src.utils.config_manager import config_manager

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import pythoncom

class COMHandler:
    """COM 处理器类"""
    
    def __init__(self):
        self._initialized = False
    
    def initialize_com(self):
        """初始化 COM"""
        try:
            if not self._initialized:
                pythoncom.CoInitialize()
                self._initialized = True
                logger.info("COM 初始化成功")
        except Exception as e:
            logger.error(f"COM 初始化失败: {e}")
            raise
    
    def uninitialize_com(self):
        """反初始化 COM"""
        try:
            if self._initialized:
                pythoncom.CoUninitialize()
                self._initialized = False
                logger.info("COM 反初始化成功")
        except Exception as e:
            logger.error(f"COM 反初始化失败: {e}")
            raise

# 创建单例实例
com_handler = COMHandler()

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt, QSettings

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时字幕")
        
        # 加载设置
        self.settings = QSettings('RealTimeTrans', 'RealtimeSubtitles')
        self.load_window_settings()
        
        # 初始化 ASR 管理器
        self.asr_manager = ASRModelManager()
        
        # 创建 UI
        self.init_ui()
        
        # 连接信号
        self.setup_signals()
    
    def init_ui(self):
        """初始化 UI"""
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        
        # 创建模型选择下拉框
        self.model_combo = QComboBox()
        self.refresh_model_list()
        layout.addWidget(QLabel("选择模型:"))
        layout.addWidget(self.model_combo)
        
        # 创建音频设备选择下拉框
        self.device_combo = QComboBox()
        self.refresh_device_list()
        layout.addWidget(QLabel("选择音频设备:"))
        layout.addWidget(self.device_combo)
        
        # 创建控制按钮
        self.start_button = QPushButton("开始识别")
        self.stop_button = QPushButton("停止识别")
        self.stop_button.setEnabled(False)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        # 创建字幕显示标签
        self.subtitle_label = QLabel()
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 150);
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 16pt;
            }
        """)
        layout.addWidget(self.subtitle_label)
        
        # 创建状态标签
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
    
    def setup_signals(self):
        """设置信号连接"""
        # 按钮信号
        self.start_button.clicked.connect(self.start_recognition)
        self.stop_button.clicked.connect(self.stop_recognition)
        
        # 下拉框信号
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        
        # ASR 管理器信号
        self.asr_manager.model_loaded.connect(self.on_model_loaded)
        self.asr_manager.recognition_started.connect(self.on_recognition_started)
        self.asr_manager.recognition_stopped.connect(self.on_recognition_stopped)
        self.asr_manager.error_occurred.connect(self.show_error_message)
        
        # 转录信号
        self.asr_manager.signals.new_text.connect(self.update_subtitle_text)
        self.asr_manager.signals.status_updated.connect(self.update_status)
        self.asr_manager.signals.error_occurred.connect(self.show_error_message)
    
    def refresh_model_list(self):
        """刷新模型列表"""
        self.model_combo.clear()
        available_engines = self.asr_manager.get_available_engines()
        for engine, enabled in available_engines.items():
            if enabled:
                self.model_combo.addItem(engine)
    
    def refresh_device_list(self):
        """刷新设备列表"""
        self.device_combo.clear()
        devices = self.asr_manager.get_audio_devices()
        for device in devices:
            self.device_combo.addItem(device.name, device)
    
    def start_recognition(self):
        """开始识别"""
        # 加载选中的模型
        model_type = self.model_combo.currentText()
        if not self.asr_manager.load_model(model_type):
            self.show_error_message("加载模型失败")
            return
            
        # 设置音频设备
        device = self.device_combo.currentData()
        if not self.asr_manager.set_audio_device(device):
            self.show_error_message("设置音频设备失败")
            return
            
        # 开始识别
        if self.asr_manager.start_recognition():
            self.update_ui_state(True)
        else:
            self.show_error_message("启动识别失败")
    
    def stop_recognition(self):
        """停止识别"""
        if self.asr_manager.stop_recognition():
            self.update_ui_state(False)
        else:
            self.show_error_message("停止识别失败")
    
    def on_model_changed(self, model_type):
        """模型改变处理"""
        self.asr_manager.load_model(model_type)
    
    def on_device_changed(self, device_name):
        """设备改变处理"""
        device = self.device_combo.currentData()
        self.asr_manager.set_audio_device(device)
    
    def on_model_loaded(self, model_type):
        """模型加载完成处理"""
        self.status_label.setText(f"已加载模型: {model_type}")
    
    def on_recognition_started(self):
        """识别开始处理"""
        self.status_label.setText("识别已开始")
    
    def on_recognition_stopped(self):
        """识别停止处理"""
        self.status_label.setText("识别已停止")
    
    def update_subtitle_text(self, text):
        """更新字幕文本"""
        self.subtitle_label.setText(text)
    
    def update_status(self, status):
        """更新状态"""
        self.status_label.setText(status)
    
    def update_ui_state(self, is_running):
        """更新 UI 状态"""
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)
        self.model_combo.setEnabled(not is_running)
        self.device_combo.setEnabled(not is_running)
    
    def show_error_message(self, message):
        """显示错误消息"""
        QMessageBox.warning(self, "错误", message)
    
    def load_window_settings(self):
        """加载窗口设置"""
        pos = self.settings.value("pos", None)
        size = self.settings.value("size", None)
        if pos:
            self.move(pos)
        if size:
            self.resize(size)
    
    def save_window_settings(self):
        """保存窗口设置"""
        self.settings.setValue("pos", self.pos())
        self.settings.setValue("size", self.size())
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止识别
        self.asr_manager.stop_recognition()
        # 保存设置
        self.save_window_settings()
        event.accept()

def main():
    """主程序入口"""
    try:
        # 加载配置
        config = load_config()
        logger.info("配置加载成功")
        
        # 获取 Sherpa-ONNX 日志文件路径
        sherpa_log_file = sherpa_logger.get_log_file()
        logger.info(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

        # 初始化COM
        com_handler.initialize_com()
        logger.info("COM 初始化成功")

        # 创建应用实例
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # 创建主窗口
        window = MainWindow()
        window.show()
        
        # 运行应用
        result = app.exec_()
        
        # 清理 COM
        com_handler.uninitialize_com()
        logger.info("COM 清理完成")
        
        return result

    except Exception as e:
        logger.error(f"程序启动错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 确保 COM 清理
        try:
            com_handler.uninitialize_com()
        except:
            pass
            
        return 1

if __name__ == "__main__":
    sys.exit(main())
