"""
信号模块
定义应用程序中使用的所有信号
"""
from PyQt5.QtCore import QObject, pyqtSignal

class TranscriptionSignals(QObject):
    """转录信号类"""
    
    # 文本更新信号
    new_text = pyqtSignal(str)
    
    # 进度更新信号
    progress_updated = pyqtSignal(int, str)
    
    # 转录完成信号
    transcription_finished = pyqtSignal()
    
    # 状态更新信号
    status_updated = pyqtSignal(str)
    
    # 错误信号
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        """初始化信号类"""
        super().__init__()
