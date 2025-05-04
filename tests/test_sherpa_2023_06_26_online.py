#!/usr/bin/env python3
"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的系统音频捕获和在线转录功能

此测试程序用于验证新模型的系统音频捕获和在线转录功能，可以播放MP4文件，
测试程序会实时捕获系统音频并进行转录。
"""

import os
import sys
import time
import json
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

# 导入 SubtitleWidget
from src.ui.widgets.subtitle_widget import SubtitleWidget

# 导入音频捕获工具
import soundcard as sc

# 模型路径
MODEL_2023_06_26_PATH = r"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26"


def load_model_2023_06_26(use_int8: bool = True):
    """
    加载 2023-06-26 模型

    Args:
        use_int8: 是否使用int8量化模型

    Returns:
        sherpa_onnx.OnlineRecognizer: 模型实例
    """
    try:
        print(f"加载 2023-06-26 模型 (use_int8={use_int8})...")

        # 确定模型文件名
        if use_int8:
            encoder = os.path.join(MODEL_2023_06_26_PATH, "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx")
            decoder = os.path.join(MODEL_2023_06_26_PATH, "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx")
            joiner = os.path.join(MODEL_2023_06_26_PATH, "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx")
        else:
            encoder = os.path.join(MODEL_2023_06_26_PATH, "encoder-epoch-99-avg-1-chunk-16-left-128.onnx")
            decoder = os.path.join(MODEL_2023_06_26_PATH, "decoder-epoch-99-avg-1-chunk-16-left-128.onnx")
            joiner = os.path.join(MODEL_2023_06_26_PATH, "joiner-epoch-99-avg-1-chunk-16-left-128.onnx")

        tokens = os.path.join(MODEL_2023_06_26_PATH, "tokens.txt")

        # 检查文件是否存在
        for file_path in [encoder, decoder, joiner, tokens]:
            if not os.path.exists(file_path):
                print(f"错误: 文件不存在: {file_path}")
                return None

        # 创建模型实例
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search"
        )

        # 添加引擎类型属性，以便与现有代码兼容
        recognizer.engine_type = "sherpa_2023_06_26"

        print(f"成功加载 2023-06-26 模型 ({'int8量化' if use_int8 else '标准'})")
        return recognizer

    except Exception as e:
        print(f"加载 2023-06-26 模型失败: {e}")
        import traceback
        print(traceback.format_exc())
        return None


class TestWindow(QMainWindow):
    """测试窗口"""
    def __init__(self):
        super().__init__()

        # 设置窗口属性
        self.setWindowTitle("测试 Sherpa-ONNX 2023-06-26 在线转录")
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
        self.subtitle_widget.current_engine_type = "sherpa_2023_06_26"

        # 加载 2023-06-26 模型
        print("加载 2023-06-26 模型...")
        self.recognizer = load_model_2023_06_26(use_int8=False)  # 使用标准模型
        if not self.recognizer:
            print("加载 2023-06-26 模型失败")
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
        self.transcript_file = os.path.join(project_root, "transcripts", f"transcript_2023_06_26_{timestamp}.txt")
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
                            # 创建一个新的流
                            stream = self.recognizer.create_stream()

                            # 接受音频数据
                            stream.accept_waveform(self.sample_rate, data)

                            # 添加尾部填充
                            tail_paddings = np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
                            stream.accept_waveform(self.sample_rate, tail_paddings)

                            # 标记输入结束
                            stream.input_finished()

                            # 解码
                            while self.recognizer.is_ready(stream):
                                self.recognizer.decode_stream(stream)

                            # 获取结果
                            result = self.recognizer.get_result(stream)

                            if result:
                                # 格式化文本：首字母大写，末尾加句号
                                text = result
                                if text:
                                    text = text[0].upper() + text[1:]
                                    if not text.endswith(('.', '!', '?')):
                                        text += '.'

                                print(f"\n完整结果: {text}")
                                # 保存到文件
                                with open(self.transcript_file, "a", encoding="utf-8") as f:
                                    f.write(f"[完整结果] {text}\n")

                                # 更新字幕控件
                                self.subtitle_widget.update_text(text)
                            else:
                                # 如果没有结果，显示部分结果（空字符串）
                                print(".", end="", flush=True)
                                # 更新字幕控件
                                self.subtitle_widget.update_text("PARTIAL: 正在识别...")

                                # 每10次没有结果时，保存一条记录
                                if np.random.randint(0, 10) == 0:
                                    with open(self.transcript_file, "a", encoding="utf-8") as f:
                                        f.write("[部分结果] 正在识别...\n")
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
                    # 创建一个新的流
                    stream = self.recognizer.create_stream()

                    # 标记输入结束
                    stream.input_finished()

                    # 解码
                    while self.recognizer.is_ready(stream):
                        self.recognizer.decode_stream(stream)

                    # 获取结果
                    final_result = self.recognizer.get_result(stream)

                    if final_result:
                        # 格式化文本：首字母大写，末尾加句号
                        text = final_result
                        if text:
                            text = text[0].upper() + text[1:]
                            if not text.endswith(('.', '!', '?')):
                                text += '.'

                        print(f"\n最终转录结果: {text}")
                        # 保存到文件
                        with open(self.transcript_file, "a", encoding="utf-8") as f:
                            f.write(f"[最终结果] {text}\n")

                        # 更新字幕控件
                        self.subtitle_widget.update_text(text)
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


def test_sherpa_2023_06_26_online():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的系统音频捕获和在线转录功能"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的系统音频捕获和在线转录功能")
    print("=" * 80)

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
    test_sherpa_2023_06_26_online()
