"""
测试主程序中的 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型功能
"""
import os
import sys
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入主窗口
from src.ui.main_window import MainWindow

def test_main_program():
    """测试主程序"""
    print("=" * 80)
    print("测试主程序中的 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型功能")
    print("=" * 80)
    
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 创建主窗口
    main_window = MainWindow()
    main_window.show()
    
    # 设置模型为 sherpa_0626
    def set_model():
        print("设置模型为 sherpa_0626...")
        main_window.set_asr_model("sherpa_0626")
        print("模型设置成功")
        
        # 延迟启动转录
        QTimer.singleShot(2000, start_transcription)
    
    # 开始转录
    def start_transcription():
        print("开始转录...")
        main_window._on_start_clicked()
        
        # 延迟停止转录
        QTimer.singleShot(10000, stop_transcription)
    
    # 停止转录
    def stop_transcription():
        print("停止转录...")
        main_window._on_stop_clicked()
        
        # 延迟测试文件转录
        QTimer.singleShot(2000, test_file_transcription)
    
    # 测试文件转录
    def test_file_transcription():
        print("测试文件转录...")
        
        # 查找测试文件
        test_files_dir = os.path.join(project_root, "tests", "test_files")
        if not os.path.exists(test_files_dir):
            os.makedirs(test_files_dir, exist_ok=True)
        
        # 默认测试文件
        default_test_file = os.path.join(test_files_dir, "test_audio.mp3")
        
        # 如果默认测试文件不存在，提示用户
        if not os.path.exists(default_test_file):
            print(f"默认测试文件不存在: {default_test_file}")
            print("请手动选择一个音频/视频文件进行测试")
            main_window.select_file()
        else:
            # 如果默认测试文件存在，直接使用
            print(f"使用默认测试文件: {default_test_file}")
            main_window._on_file_selected(default_test_file)
            
            # 开始转录
            main_window._on_start_clicked()
            
            # 延迟停止转录
            QTimer.singleShot(30000, stop_file_transcription)
    
    # 停止文件转录
    def stop_file_transcription():
        print("停止文件转录...")
        main_window._on_stop_clicked()
        
        # 延迟退出
        QTimer.singleShot(2000, app.quit)
    
    # 延迟设置模型
    QTimer.singleShot(1000, set_model)
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_main_program()
