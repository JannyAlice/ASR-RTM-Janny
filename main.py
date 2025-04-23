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
from src.ui.main_window import MainWindow
from src.utils.com_handler import com_handler

def main():
    """
    主程序入口
    初始化COM、创建并运行应用程序
    """
    try:
        # 初始化COM
        com_handler.initialize_com()
        print("COM 初始化成功")

        # 创建应用
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # 使用Fusion风格

        # 创建主窗口
        window = MainWindow()
        window.show()

        # 运行应用
        sys.exit(app.exec_())

    except Exception as e:
        print(f"程序启动错误: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # 释放COM
        com_handler.uninitialize_com()
        print("COM 已释放")

if __name__ == "__main__":
    main()
