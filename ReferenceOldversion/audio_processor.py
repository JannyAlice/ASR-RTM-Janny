"""
音频处理模块
负责音频捕获和处理
"""
import threading
import time
import json
import numpy as np
import soundcard as sc
from typing import List, Any

from src.core.signals import TranscriptionSignals

class AudioDevice:
    """音频设备类"""

    def __init__(self, id, name, is_input=True):
        """
        初始化音频设备

        Args:
            id: 设备ID
            name: 设备名称
            is_input: 是否为输入设备
        """
        self.id = id
        self.name = name
        self.is_input = is_input

    def __str__(self):
        return f"{self.name} ({self.id})"

class AudioProcessor:
    """音频处理器类"""

    def __init__(self, signals: TranscriptionSignals):
        """
        初始化音频处理器

        Args:
            signals: 信号实例
        """
        self.signals = signals
        self.current_device = None
        self.is_capturing = False
        self.capture_thread = None
        self.sample_rate = 16000
        self.buffer_size = 4000

    def get_audio_devices(self) -> List[AudioDevice]:
        """
        获取音频设备列表

        Returns:
            List[AudioDevice]: 音频设备列表
        """
        devices = []

        try:
            # 获取所有输入设备
            speakers = sc.all_speakers()
            for speaker in speakers:
                devices.append(AudioDevice(speaker.id, speaker.name, False))

            # 获取所有输出设备
            mics = sc.all_microphones(include_loopback=True)
            for mic in mics:
                devices.append(AudioDevice(mic.id, mic.name, True))

            return devices

        except Exception as e:
            print(f"获取音频设备失败: {e}")
            return []

    def set_current_device(self, device: AudioDevice) -> bool:
        """
        设置当前设备

        Args:
            device: 音频设备

        Returns:
            bool: 设置是否成功
        """
        if not device:
            return False

        self.current_device = device
        return True

    def start_capture(self, recognizer: Any) -> bool:
        """
        开始捕获音频

        Args:
            recognizer: 识别器实例

        Returns:
            bool: 开始捕获是否成功
        """
        if self.is_capturing:
            return False

        if not self.current_device:
            self.signals.error_occurred.emit("未选择音频设备")
            return False

        # 设置捕获标志
        self.is_capturing = True

        # 初始化进度条为在线转录模式
        self.signals.progress_updated.emit(50, "转录时长: 00:00")

        # 创建捕获线程
        self.capture_thread = threading.Thread(
            target=self._capture_audio_thread,
            args=(recognizer,),
            daemon=True
        )

        # 启动线程
        self.capture_thread.start()

        return True

    def stop_capture(self) -> bool:
        """
        停止捕获音频

        Returns:
            bool: 停止捕获是否成功
        """
        if not self.is_capturing:
            return False

        # 清除捕获标志
        self.is_capturing = False

        # 等待线程结束
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        self.capture_thread = None

        # 重置进度条，但不立即更新，给字幕窗口留出时间显示保存信息
        # 使用延迟重置进度条
        def reset_progress_bar():
            self.signals.progress_updated.emit(0, "%p% - %v/%m")

        # 使用线程安全的方式延迟重置进度条
        threading.Timer(1.0, reset_progress_bar).start()

        return True

    def _capture_audio_thread(self, recognizer: Any) -> None:
        """
        音频捕获线程

        Args:
            recognizer: 识别器实例
        """
        start_time = time.time()
        last_progress_update = time.time()

        try:
            # 获取音频设备
            with sc.get_microphone(id=str(self.current_device.id), include_loopback=True).recorder(samplerate=self.sample_rate) as mic:
                # 发送状态信号
                self.signals.status_updated.emit(f"正在从 {self.current_device.name} 捕获音频...")

                # 循环捕获音频
                while self.is_capturing:
                    # 更新进度条显示转录时长
                    current_time = time.time()
                    if current_time - last_progress_update >= 0.5:  # 每0.5秒更新一次
                        elapsed_seconds = current_time - start_time
                        minutes = int(elapsed_seconds // 60)
                        seconds = int(elapsed_seconds % 60)
                        time_str = f"转录时长: {minutes:02d}:{seconds:02d}"
                        self.signals.progress_updated.emit(50, time_str)  # 使用固定的50%进度
                        last_progress_update = current_time

                    # 捕获音频数据
                    data = mic.record(numframes=self.buffer_size)

                    # 转换为单声道
                    if data.shape[1] > 1:
                        data = np.mean(data, axis=1)

                    # 处理音频数据
                    try:
                        # 检查数据是否有效
                        if np.max(np.abs(data)) < 0.01:
                            print("DEBUG: 音频数据几乎是静音，跳过")
                            continue

                        # 导入 Sherpa-ONNX 日志工具
                        try:
                            from src.utils.sherpa_logger import sherpa_logger
                        except ImportError:
                            # 如果导入失败，创建一个简单的日志记录器
                            class DummyLogger:
                                def debug(self, msg): print(f"DEBUG: {msg}")
                                def info(self, msg): print(f"INFO: {msg}")
                                def warning(self, msg): print(f"WARNING: {msg}")
                                def error(self, msg): print(f"ERROR: {msg}")
                            sherpa_logger = DummyLogger()

                        # 检查当前引擎类型
                        engine_type = getattr(recognizer, 'engine_type', None)
                        if engine_type and engine_type.startswith('sherpa'):
                            # 对于 Sherpa-ONNX 模型，直接传递 numpy 数组
                            sherpa_logger.debug(f"使用 Sherpa-ONNX 模型，直接传递 numpy 数组")
                            sherpa_logger.debug(f"音频数据类型: {type(data)}, 形状: {data.shape}, 最大值: {np.max(np.abs(data))}")
                            accept_result = recognizer.AcceptWaveform(data)
                        else:
                            # 对于 Vosk 模型，转换为 16 位整数字节
                            data_bytes = (data * 32767).astype(np.int16).tobytes()
                            print(f"DEBUG: 处理音频数据，类型: {type(data_bytes)}, 长度: {len(data_bytes)}")
                            accept_result = recognizer.AcceptWaveform(data_bytes)
                        sherpa_logger.debug(f"AcceptWaveform 结果: {accept_result}")

                        if accept_result:
                            # 获取完整结果
                            result = recognizer.Result()
                            print(f"DEBUG: 完整结果: {result}, 类型: {type(result)}")

                            # 直接使用结果文本（针对 Sherpa-ONNX 模型）
                            if isinstance(result, str) and not result.startswith('{'):
                                # 如果是纯文本字符串（不是 JSON），直接使用
                                text = result.strip()
                                print(f"DEBUG: 直接使用文本结果: {text}")
                            else:
                                # 尝试解析 JSON 或其他格式
                                try:
                                    if isinstance(result, str):
                                        result_json = json.loads(result)
                                    elif hasattr(result, 'text'):
                                        result_json = {"text": result.text}
                                    elif hasattr(result, '__str__'):
                                        result_json = {"text": str(result)}
                                    else:
                                        result_json = {"text": "无法解析的结果"}

                                    print(f"DEBUG: 解析后的结果: {result_json}")

                                    # 提取文本
                                    if 'text' in result_json and result_json['text'].strip():
                                        text = result_json['text'].strip()
                                    else:
                                        print(f"DEBUG: 结果中没有文本或文本为空")
                                        continue
                                except Exception as e:
                                    print(f"DEBUG: 解析结果错误: {e}")
                                    # 如果解析失败，尝试直接使用结果
                                    text = str(result).strip()
                                    if not text:
                                        continue

                            # 格式化文本
                            if text:
                                if len(text) > 0:
                                    text = text[0].upper() + text[1:]
                                if text[-1] not in ['.', '?', '!']:
                                    text += '.'
                                print(f"DEBUG: 发送文本: {text}")
                                self.signals.new_text.emit(text)
                            else:
                                print(f"DEBUG: 文本为空，不发送")
                        else:
                            # 获取部分结果
                            partial_result = recognizer.PartialResult()
                            print(f"DEBUG: 部分结果: {partial_result}, 类型: {type(partial_result)}")

                            # 直接使用结果文本（针对 Sherpa-ONNX 模型）
                            if isinstance(partial_result, str) and not partial_result.startswith('{'):
                                # 如果是纯文本字符串（不是 JSON），直接使用
                                partial_text = partial_result.strip()
                                print(f"DEBUG: 直接使用部分文本结果: {partial_text}")
                                if partial_text:
                                    print(f"DEBUG: 发送部分文本: {partial_text}")
                                    self.signals.new_text.emit("PARTIAL:" + partial_text)
                                else:
                                    print(f"DEBUG: 部分文本为空，不发送")
                            else:
                                # 尝试解析 JSON 或其他格式
                                try:
                                    if isinstance(partial_result, str):
                                        try:
                                            partial = json.loads(partial_result)
                                        except json.JSONDecodeError:
                                            # 如果不是有效的 JSON，直接使用文本
                                            partial = {"partial": partial_result}
                                    elif isinstance(partial_result, dict):
                                        partial = partial_result
                                    elif hasattr(partial_result, 'partial'):
                                        partial = {'partial': str(partial_result.partial)}
                                    else:
                                        partial = {'partial': str(partial_result)}

                                    print(f"DEBUG: 解析后的部分结果: {partial}")

                                    # 确保partial字段存在且有效
                                    partial_text = partial.get('partial', '').strip()
                                    if partial_text:
                                        print(f"DEBUG: 发送部分文本: {partial_text}")
                                        self.signals.new_text.emit("PARTIAL:" + partial_text)
                                    else:
                                        print(f"DEBUG: 部分结果中没有文本或文本为空")
                                except Exception as e:
                                    print(f"DEBUG: 解析部分结果错误: {e}")
                                    # 如果解析失败，尝试直接使用结果
                                    partial_text = str(partial_result).strip()
                                    if partial_text:
                                        print(f"DEBUG: 发送部分文本: {partial_text}")
                                        self.signals.new_text.emit("PARTIAL:" + partial_text)
                                    else:
                                        print(f"DEBUG: 部分文本为空，不发送")
                    except Exception as e:
                        self.signals.error_occurred.emit(f"音频处理错误: {str(e)}")
                        print(f"DEBUG: 音频处理错误: {e}")
                        import traceback
                        traceback.print_exc()

        except Exception as e:
            self.signals.error_occurred.emit(f"音频捕获错误: {e}")

        finally:
            # 发送状态信号
            elapsed_time = time.time() - start_time
            self.signals.status_updated.emit(f"音频捕获已停止，持续时间: {elapsed_time:.2f}秒")

            # 确保捕获标志被清除
            self.is_capturing = False
