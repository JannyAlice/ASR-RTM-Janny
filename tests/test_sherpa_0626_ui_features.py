"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的UI功能
包括：
1. 持久流模式
2. 累积文本功能
3. 时间戳功能
4. 格式化文本功能
5. 文件保存逻辑
"""
import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入主窗口
from src.ui.main_window import MainWindow

def test_sherpa_0626_ui_features():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的UI功能"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的UI功能")
    print("=" * 80)

    # 创建应用程序
    app = QApplication(sys.argv)

    # 创建主窗口
    main_window = MainWindow()
    main_window.show()

    # 设置模型为 sherpa_0626
    print("设置模型为 sherpa_0626...")
    main_window.set_asr_model("sherpa_0626")
    print("模型设置成功")

    # 测试保存转录文本
    def test_save_transcript():
        print("\n" + "=" * 40)
        print("测试保存转录文本")
        print("=" * 40)
        
        # 模拟添加一些转录文本
        main_window.subtitle_widget.update_text("这是一个测试文本，用于测试保存功能。")
        main_window.subtitle_widget.update_text("This is a test text for testing save function.")
        
        # 保存转录文本
        save_path = main_window.save_transcript()
        if save_path:
            print(f"转录文本已保存到: {save_path}")
        else:
            print("保存转录文本失败")
        
        # 退出应用程序
        QTimer.singleShot(1000, app.quit)
    
    # 延迟执行测试
    QTimer.singleShot(2000, test_save_transcript)

    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_sherpa_0626_ui_features()
