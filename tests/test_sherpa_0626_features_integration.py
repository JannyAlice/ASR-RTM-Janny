"""
集成测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的高级功能
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
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtCore import QTimer

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入 Sherpa-ONNX 日志工具
from src.utils.sherpa_logger import sherpa_logger

# 导入主窗口
from src.ui.main_window import MainWindow

def test_sherpa_0626_features_integration():
    """集成测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的高级功能"""
    print("=" * 80)
    print("集成测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的高级功能")
    print("=" * 80)
    
    # 设置日志
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")
    
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 创建主窗口
    main_window = MainWindow()
    main_window.show()
    
    # 设置模型为 sherpa_0626
    print("设置模型为 sherpa_0626...")
    main_window.set_asr_model("sherpa_0626")
    print("模型设置成功")
    
    # 测试在线转录
    def test_online_transcription():
        print("\n" + "=" * 40)
        print("测试在线转录")
        print("=" * 40)
        
        # 开始转录
        main_window._on_start_clicked()
        
        # 等待10秒
        QTimer.singleShot(10000, stop_transcription)
    
    # 停止转录
    def stop_transcription():
        print("停止转录...")
        main_window._on_stop_clicked()
        
        # 等待1秒
        QTimer.singleShot(1000, test_save_transcript)
    
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
            
            # 检查保存的文件
            print("\n检查保存的文件:")
            
            # 检查TXT文件
            if os.path.exists(save_path):
                print(f"TXT文件存在: {save_path}")
                with open(save_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    print(f"TXT文件内容长度: {len(content)} 字符")
                    print(f"TXT文件内容前100个字符: {content[:100]}...")
            else:
                print(f"TXT文件不存在: {save_path}")
            
            # 检查调试文件
            debug_path = save_path.replace('.txt', '_debug.txt')
            if os.path.exists(debug_path):
                print(f"调试文件存在: {debug_path}")
                with open(debug_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    print(f"调试文件内容长度: {len(content)} 字符")
                    print(f"调试文件内容前100个字符: {content[:100]}...")
            else:
                print(f"调试文件不存在: {debug_path}")
            
            # 检查SRT文件
            srt_path = save_path.replace('.txt', '.srt')
            if os.path.exists(srt_path):
                print(f"SRT文件存在: {srt_path}")
                with open(srt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    print(f"SRT文件内容长度: {len(content)} 字符")
                    print(f"SRT文件内容前100个字符: {content[:100]}...")
            else:
                print(f"SRT文件不存在: {srt_path}")
        else:
            print("保存转录文本失败")
        
        # 等待1秒
        QTimer.singleShot(1000, test_get_formatted_transcript)
    
    # 测试获取格式化的转录文本
    def test_get_formatted_transcript():
        print("\n" + "=" * 40)
        print("测试获取格式化的转录文本")
        print("=" * 40)
        
        # 获取当前引擎
        engine = main_window.model_manager.current_engine
        if engine:
            print(f"当前引擎: {type(engine).__name__}")
            
            # 创建识别器
            recognizer = main_window.model_manager.create_recognizer()
            if recognizer:
                print(f"识别器: {type(recognizer).__name__}")
                
                # 检查识别器是否有 get_formatted_transcript 方法
                if hasattr(recognizer, 'get_formatted_transcript'):
                    print("识别器有 get_formatted_transcript 方法")
                    
                    # 模拟添加一些文本
                    if hasattr(recognizer, 'AcceptWaveform'):
                        # 创建一个简单的正弦波
                        sample_rate = 16000
                        duration = 1  # 秒
                        frequency = 440  # Hz
                        t = np.linspace(0, duration, int(sample_rate * duration), False)
                        audio_data = np.sin(2 * np.pi * frequency * t)
                        audio_data = (audio_data * 32767).astype(np.int16)
                        audio_bytes = audio_data.tobytes()
                        
                        # 处理音频数据
                        print("处理音频数据...")
                        recognizer.AcceptWaveform(audio_bytes)
                        
                        # 获取结果
                        result = recognizer.Result()
                        print(f"结果: {result}")
                        
                        # 获取最终结果
                        final = recognizer.FinalResult()
                        print(f"最终结果: {final}")
                    
                    # 获取格式化的转录文本
                    formatted_transcript = recognizer.get_formatted_transcript()
                    print(f"格式化的转录文本: {formatted_transcript}")
                else:
                    print("识别器没有 get_formatted_transcript 方法")
            else:
                print("创建识别器失败")
        else:
            print("当前引擎为空")
        
        # 等待1秒
        QTimer.singleShot(1000, finish_test)
    
    # 完成测试
    def finish_test():
        print("\n" + "=" * 40)
        print("测试完成")
        print("=" * 40)
        print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")
        
        # 退出应用程序
        app.quit()
    
    # 延迟执行测试
    QTimer.singleShot(2000, test_online_transcription)
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_sherpa_0626_features_integration()
