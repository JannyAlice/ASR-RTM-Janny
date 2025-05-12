"""
Qt兼容性模块
提供Qt版本信息和兼容性函数，确保在不同Qt绑定之间的一致性
"""
import sys
import logging
from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR

logger = logging.getLogger(__name__)

def get_qt_version_info():
    """
    获取Qt版本信息
    
    Returns:
        dict: 包含Qt版本信息的字典
    """
    return {
        "qt_version": QT_VERSION_STR,
        "binding": "PyQt5",
        "binding_version": PYQT_VERSION_STR,
        "python_version": sys.version
    }

def log_qt_info(logger):
    """
    记录Qt版本信息到日志
    
    Args:
        logger: 日志记录器
    """
    info = get_qt_version_info()
    logger.info(f"Qt版本: {info['qt_version']}")
    logger.info(f"Qt绑定: {info['binding']} {info['binding_version']}")
    logger.info(f"Python版本: {info['python_version']}")

def is_pyqt5():
    """
    检查当前是否使用PyQt5
    
    Returns:
        bool: 如果使用PyQt5则返回True，否则返回False
    """
    return get_qt_version_info()["binding"] == "PyQt5"

def get_exec_method(app):
    """
    获取Qt应用程序的exec方法
    
    PyQt5使用exec_()，而PySide6使用exec()
    
    Args:
        app: Qt应用程序实例
        
    Returns:
        callable: 应用程序的exec方法
    """
    if hasattr(app, 'exec_'):
        return app.exec_
    else:
        return app.exec

def connect_signal(signal, slot, connection_type=None):
    """
    连接信号到槽
    
    处理PyQt5和PySide6之间的API差异
    
    Args:
        signal: 信号
        slot: 槽函数
        connection_type: 连接类型，默认为None（使用默认连接类型）
        
    Returns:
        bool: 连接是否成功
    """
    try:
        if connection_type is not None:
            return signal.connect(slot, connection_type)
        else:
            return signal.connect(slot)
    except Exception as e:
        logger.error(f"连接信号失败: {str(e)}")
        return False
