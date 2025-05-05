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
import traceback
from PyQt5.QtWidgets import QApplication

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

# 导入自定义模块
from src.utils.sherpa_logger import sherpa_logger
from src.utils.config_manager import load_config
from src.utils.config_manager import config_manager
from src.utils.com_handler import com_handler

# 检查COM处理器状态
logger.info("检查COM处理器状态: " + ("已初始化" if hasattr(com_handler, "_initialized") and com_handler._initialized else "未初始化"))

def main():
    """主程序入口"""
    try:
        # 加载配置
        config = load_config()
        logger.info("配置加载成功")

        # 获取 Sherpa-ONNX 日志文件路径
        sherpa_log_file = sherpa_logger.get_log_file()
        logger.info(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

        # 确保COM已初始化（在主程序中只初始化一次）
        if not hasattr(com_handler, "_initialized") or not com_handler._initialized:
            logger.info("主程序中初始化COM...")
            com_handler.initialize_com()
            logger.info("COM 初始化成功")
        else:
            logger.info("COM 已经初始化，跳过")

        # 创建应用实例
        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        # 导入MainWindow类（在COM初始化后导入，避免循环导入问题）
        from src.ui.main_window import MainWindow

        # 创建主窗口 - 使用src/ui/main_window.py中的MainWindow类
        # 传递recognizer=None，确保MainWindow不会重复初始化COM
        window = MainWindow(recognizer=None)
        window.show()

        # 运行应用
        logger.info("开始应用程序主循环")
        result = app.exec_()
        logger.info("应用程序主循环结束")

        # 清理 COM（确保只清理一次）
        if hasattr(com_handler, "_initialized") and com_handler._initialized:
            logger.info("主程序中清理COM...")
            com_handler.uninitialize_com()
            logger.info("COM 清理完成")
        else:
            logger.info("COM 已经清理，跳过")

        return result

    except Exception as e:
        logger.error(f"程序启动错误: {str(e)}")
        logger.error(traceback.format_exc())

        # 确保 COM 清理
        try:
            if hasattr(com_handler, "_initialized") and com_handler._initialized:
                logger.info("异常处理中清理COM...")
                com_handler.uninitialize_com()
                logger.info("异常处理中COM清理完成")
        except Exception as com_error:
            logger.error(f"COM清理错误: {com_error}")

        return 1

if __name__ == "__main__":
    sys.exit(main())
