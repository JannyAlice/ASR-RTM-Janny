"""
测试修复后的Sherpa-ONNX在线转录功能（文本格式化和部分结果处理）
"""
import os
import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入 sherpa_onnx 模块
try:
    import sherpa_onnx
    HAS_SHERPA_ONNX = True
except ImportError:
    HAS_SHERPA_ONNX = False
    print("未安装 sherpa_onnx 模块，无法测试 Sherpa-ONNX 模型")
    sys.exit(1)

# 导入 Sherpa-ONNX 日志工具
from src.utils.sherpa_logger import sherpa_logger

# 导入 ASRModelManager
from src.core.asr.model_manager import ASRModelManager

# 导入 SubtitleWidget
from src.ui.widgets.subtitle_widget import SubtitleWidget

# 导入音频捕获工具
import soundcard as sc

class TestWindow(QMainWindow):
    """测试窗口"""
    def __init__(self):
        super().__init__()

        # 设置窗口属性
        self.setWindowTitle("测试 Sherpa-ONNX 在线转录")
        self.resize(800, 400)

        # 创建中央控件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建布局
        layout = QVBoxLayout(central_widget)

        # 创建字幕控件
        self.subtitle_widget = SubtitleWidget(self)
        layout.addWidget(self.subtitle_widget, 1)  # 1表示拉伸因子

        # 设置布局
        central_widget.setLayout(layout)

        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
                color: #FFFFFF;
            }
            QWidget {
                background-color: #2D2D30;
                color: #FFFFFF;
            }
        """)

        # 设置字幕控件的引擎类型
        self.subtitle_widget.current_engine_type = "sherpa_int8"

        # 创建 ASRModelManager 实例
        self.model_manager = ASRModelManager()

        # 加载 Sherpa-ONNX int8 模型
        model_name = "sherpa_int8"
        print(f"加载模型: {model_name}")
        if not self.model_manager.load_model(model_name):
            print(f"加载模型 {model_name} 失败")
            return

        # 创建识别器
        print("创建识别器...")
        self.recognizer = self.model_manager.create_recognizer()
        if not self.recognizer:
            print("创建识别器失败")
            return

        print(f"识别器类型: {type(self.recognizer).__name__}")
        print(f"识别器引擎类型: {self.recognizer.engine_type}")

        # 获取音频设备
        print("获取音频设备...")
        speakers = sc.all_speakers()
        for i, speaker in enumerate(speakers):
            print(f"{i}: {speaker.name}")

        # 选择默认设备
        self.default_device = None
        for speaker in speakers:
            if "CABLE" in speaker.name:
                self.default_device = speaker
                break

        if not self.default_device:
            self.default_device = speakers[0]

        print(f"使用设备: {self.default_device.name}")

        # 设置参数
        self.sample_rate = 16000
        self.buffer_size = 4000  # 250ms

        # 创建转录结果文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.transcript_file = os.path.join(project_root, "transcripts", f"transcript_test_{timestamp}.txt")
        os.makedirs(os.path.dirname(self.transcript_file), exist_ok=True)

        # 启动音频捕获线程
        self.is_running = True
        self.audio_thread = None

        # 启动音频捕获
        self.start_audio_capture()

    def start_audio_capture(self):
        """启动音频捕获"""
        import threading
        self.audio_thread = threading.Thread(target=self.capture_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def capture_audio(self):
        """捕获音频"""
        try:
            with sc.get_microphone(id=str(self.default_device.id), include_loopback=True).recorder(samplerate=self.sample_rate) as mic:
                print("开始捕获音频...")
                print("请播放音频，按 Ctrl+C 停止...")

                # 循环捕获音频
                while self.is_running:
                    try:
                        # 捕获音频数据
                        data = mic.record(numframes=self.buffer_size)

                        # 转换为单声道
                        if data.shape[1] > 1:
                            data = np.mean(data, axis=1)

                        # 检查数据是否有效
                        if np.max(np.abs(data)) < 0.01:
                            print(".", end="", flush=True)
                            continue

                        # 处理音频数据
                        try:
                            # 使用 recognizer.AcceptWaveform 处理音频数据
                            if self.recognizer.AcceptWaveform(data):
                                # 获取结果
                                result_json = self.recognizer.Result()
                                import json
                                result = json.loads(result_json)
                                if "text" in result and result["text"]:
                                    print(f"\n完整结果: {result['text']}")
                                    # 保存到文件
                                    with open(self.transcript_file, "a", encoding="utf-8") as f:
                                        f.write(f"[完整结果] {result['text']}\n")

                                    # 更新字幕控件
                                    self.subtitle_widget.update_text(result["text"])

                                    # 检查内部变量
                                    if hasattr(self.recognizer, 'current_text'):
                                        print(f"current_text: {self.recognizer.current_text}")
                                        # 保存到文件
                                        with open(self.transcript_file, "a", encoding="utf-8") as f:
                                            f.write(f"[current_text] {self.recognizer.current_text}\n")

                                    if hasattr(self.recognizer, 'previous_result'):
                                        print(f"previous_result: {self.recognizer.previous_result}")
                                        # 保存到文件
                                        with open(self.transcript_file, "a", encoding="utf-8") as f:
                                            f.write(f"[previous_result] {self.recognizer.previous_result}\n")
                            else:
                                # 获取部分结果
                                partial_json = self.recognizer.PartialResult()
                                import json
                                partial = json.loads(partial_json)
                                if "partial" in partial and partial["partial"]:
                                    print(f"\r部分结果: {partial['partial']}", end="", flush=True)
                                    # 保存到文件
                                    with open(self.transcript_file, "a", encoding="utf-8") as f:
                                        f.write(f"[部分结果] {partial['partial']}\n")

                                    # 更新字幕控件
                                    self.subtitle_widget.update_text("PARTIAL:" + partial["partial"])
                        except Exception as e:
                            print(f"\n处理音频数据错误: {e}")
                            import traceback
                            print(traceback.format_exc())

                        # 等待一段时间
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"\n捕获音频数据错误: {e}")
                        import traceback
                        print(traceback.format_exc())
                        time.sleep(0.5)  # 出错后等待一段时间再继续

                # 获取最终结果（在循环结束后）
                try:
                    final_result_json = self.recognizer.FinalResult()
                    import json
                    final_result = json.loads(final_result_json)
                    if "text" in final_result and final_result["text"]:
                        print(f"\n最终转录结果: {final_result['text']}")
                        # 保存到文件
                        with open(self.transcript_file, "a", encoding="utf-8") as f:
                            f.write(f"[最终结果] {final_result['text']}\n")

                        # 更新字幕控件
                        self.subtitle_widget.update_text(final_result["text"])
                except Exception as e:
                    print(f"\n获取最终结果错误: {e}")
                    import traceback
                    print(traceback.format_exc())

        except Exception as e:
            print(f"捕获音频错误: {e}")
            import traceback
            print(traceback.format_exc())

    def closeEvent(self, event):
        """关闭窗口事件"""
        self.is_running = False
        if self.audio_thread:
            self.audio_thread.join(1)
        event.accept()

def test_sherpa_online_fixed6():
    """测试修复后的Sherpa-ONNX在线转录功能（文本格式化和部分结果处理）"""
    print("测试修复后的Sherpa-ONNX在线转录功能（文本格式化和部分结果处理）...")

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 创建应用程序
    app = QApplication(sys.argv)

    # 创建测试窗口
    window = TestWindow()
    window.show()

    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_sherpa_online_fixed6()
