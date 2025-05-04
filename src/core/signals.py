"""
信号模块
定义应用程序中使用的所有信号
"""
from PyQt5.QtCore import QObject, pyqtSignal

class TranscriptionSignals(QObject):
    """转录信号类"""
    
    # 文本相关信号
    new_text = pyqtSignal(str)  # 新识别文本
    partial_result = pyqtSignal(str)  # 部分识别结果
    
    # 状态相关信号
    status_updated = pyqtSignal(str)  # 状态更新
    progress_updated = pyqtSignal(int, str)  # 进度更新
    
    # 错误相关信号
    error_occurred = pyqtSignal(str)  # 错误发生
    
    def __init__(self):
        """初始化信号类"""
        super().__init__()
