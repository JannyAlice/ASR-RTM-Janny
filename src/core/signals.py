"""
信号模块
定义应用程序中使用的所有信号，确保信号命名和参数类型的一致性
"""
from PyQt5.QtCore import QObject, pyqtSignal
import logging

logger = logging.getLogger(__name__)

class TranscriptionSignals(QObject):
    """
    转录信号类

    定义所有与语音转录相关的信号，包括文本更新、状态更新、进度更新、错误处理和转录生命周期事件。
    所有信号都遵循一致的命名约定，并提供详细的文档说明其用途、参数和使用场景。
    """

    # 文本相关信号
    new_text = pyqtSignal(str)  # 保持原名称以兼容现有代码
    """
    新识别文本信号

    当有新的完整转录文本可用时发出此信号。

    Args:
        str: 完整的转录文本
    """

    partial_result = pyqtSignal(str)  # 保持原名称以兼容现有代码
    """
    部分识别结果信号

    当有部分转录结果可用时发出此信号，通常用于实时显示正在识别的内容。

    Args:
        str: 部分转录文本
    """

    # 状态相关信号
    status_updated = pyqtSignal(str)
    """
    状态更新信号

    当转录状态发生变化时发出此信号，用于更新UI状态栏或日志。

    Args:
        str: 状态描述文本
    """

    progress_updated = pyqtSignal(int, str)
    """
    进度更新信号

    当转录进度发生变化时发出此信号，用于更新进度条和进度文本。

    Args:
        int: 进度百分比（0-100）
        str: 进度描述文本
    """

    # 错误相关信号
    error_occurred = pyqtSignal(str)
    """
    错误发生信号

    当转录过程中发生错误时发出此信号，用于显示错误消息和记录日志。

    Args:
        str: 错误描述文本
    """

    # 转录生命周期信号
    transcription_started = pyqtSignal()
    """
    转录开始信号

    当转录过程开始时发出此信号，用于更新UI状态和记录日志。
    """

    transcription_finished = pyqtSignal()
    """
    转录完成信号

    当转录过程完成时发出此信号，用于更新UI状态、保存结果和记录日志。
    """

    transcription_paused = pyqtSignal()
    """
    转录暂停信号

    当转录过程暂停时发出此信号，用于更新UI状态和记录日志。
    """

    transcription_resumed = pyqtSignal()
    """
    转录恢复信号

    当转录过程从暂停状态恢复时发出此信号，用于更新UI状态和记录日志。
    """

    def __init__(self):
        """初始化信号类"""
        super().__init__()
        logger.debug("TranscriptionSignals 初始化完成")
