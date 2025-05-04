"""
测试 Sherpa-ONNX 模型的在线转录和文件转录功能
"""
import os
import sys
import time
import json
import numpy as np
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt, QTimer

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

from src.ui.main_window import MainWindow
from src.core.asr.model_manager import ASRModelManager
from src.utils.config_manager import config_manager

def test_sherpa_onnx_models():
    """测试 Sherpa-ONNX 模型的加载和识别"""
    print("开始测试 Sherpa-ONNX 模型...")

    # 创建模型管理器
    model_manager = ASRModelManager()

    # 测试 int8 模型
    print("\n=== 测试 Sherpa-ONNX int8 模型 ===")
    if model_manager.load_model("sherpa_int8"):
        print("成功加载 Sherpa-ONNX int8 模型")

        # 创建识别器
        recognizer = model_manager.create_recognizer()
        if recognizer:
            print("成功创建 Sherpa-ONNX int8 识别器")

            # 生成测试音频数据（1秒的静音）
            sample_rate = 16000
            audio_data = np.zeros(sample_rate, dtype=np.float32)
            audio_data = (audio_data * 32768).astype(np.int16).tobytes()

            # 测试识别
            print("测试识别...")
            recognizer.AcceptWaveform(audio_data)
            result = recognizer.Result()
            print(f"识别结果: {result}")

            # 测试最终结果
            final_result = recognizer.FinalResult()
            print(f"最终结果: {final_result}")
        else:
            print("创建 Sherpa-ONNX int8 识别器失败")
    else:
        print("加载 Sherpa-ONNX int8 模型失败")

    # 测试标准模型
    print("\n=== 测试 Sherpa-ONNX 标准模型 ===")
    if model_manager.load_model("sherpa_std"):
        print("成功加载 Sherpa-ONNX 标准模型")

        # 创建识别器
        recognizer = model_manager.create_recognizer()
        if recognizer:
            print("成功创建 Sherpa-ONNX 标准识别器")

            # 生成测试音频数据（1秒的静音）
            sample_rate = 16000
            audio_data = np.zeros(sample_rate, dtype=np.float32)
            audio_data = (audio_data * 32768).astype(np.int16).tobytes()

            # 测试识别
            print("测试识别...")
            recognizer.AcceptWaveform(audio_data)
            result = recognizer.Result()
            print(f"识别结果: {result}")

            # 测试最终结果
            final_result = recognizer.FinalResult()
            print(f"最终结果: {final_result}")
        else:
            print("创建 Sherpa-ONNX 标准识别器失败")
    else:
        print("加载 Sherpa-ONNX 标准模型失败")

def test_main_window():
    """测试主窗口中的 Sherpa-ONNX 模型切换"""
    app = QApplication(sys.argv)

    # 创建主窗口
    main_window = MainWindow()
    main_window.show()

    # 设置定时器，在应用程序启动后执行测试
    def run_tests():
        # 测试切换到 Sherpa-ONNX int8 模型
        print("\n=== 测试切换到 Sherpa-ONNX int8 模型 ===")
        main_window.set_asr_model("sherpa_int8")

        # 等待一段时间
        QTimer.singleShot(2000, lambda: test_standard_model())

    def test_standard_model():
        # 测试切换到 Sherpa-ONNX 标准模型
        print("\n=== 测试切换到 Sherpa-ONNX 标准模型 ===")
        main_window.set_asr_model("sherpa_std")

        # 等待一段时间
        QTimer.singleShot(2000, lambda: test_vosk_model())

    def test_vosk_model():
        # 测试切换回 Vosk 模型
        print("\n=== 测试切换回 Vosk 模型 ===")
        main_window.set_asr_model("vosk")

        # 等待一段时间
        QTimer.singleShot(2000, lambda: app.quit())

    # 启动测试
    QTimer.singleShot(1000, run_tests)

    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    # 选择要运行的测试
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        test_main_window()
    else:
        test_sherpa_onnx_models()
