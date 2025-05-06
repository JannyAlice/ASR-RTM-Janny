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
from PyQt5.QtCore import QObject, pyqtSignal, QThread

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

class AudioWorker(QObject):
    """音频处理工作线程"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    new_text = pyqtSignal(str)
    status = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, device, sample_rate, buffer_size, recognizer):
        super().__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.recognizer = recognizer
        self.running = True

    def process(self):
        """处理音频数据"""
        start_time = time.time()
        last_progress_update = time.time()

        try:
            with sc.get_microphone(id=str(self.device.id), include_loopback=True).recorder(
                samplerate=self.sample_rate
            ) as mic:
                self.status.emit(f"正在从 {self.device.name} 捕获音频...")

                while self.running:
                    # 捕获音频数据
                    data = mic.record(numframes=self.buffer_size)

                    # 转换为单声道
                    if data.shape[1] > 1:
                        data = np.mean(data, axis=1)

                    # 检查音频数据是否有效
                    if np.max(np.abs(data)) < 0.01:
                        continue

                    try:
                        # 处理音频数据
                        engine_type = getattr(self.recognizer, 'engine_type', None)
                        if engine_type and engine_type.startswith('sherpa'):
                            # 对于 Sherpa-ONNX 模型，直接传递 numpy 数组
                            accept_result = self.recognizer.AcceptWaveform(data)
                        else:
                            # 对于 Vosk 模型，转换为 16 位整数字节
                            data_bytes = (data * 32767).astype(np.int16).tobytes()
                            accept_result = self.recognizer.AcceptWaveform(data_bytes)

                        if accept_result:
                            # 获取完整结果
                            result = self.recognizer.Result()
                            text = self._parse_result(result)
                            if text:
                                self.new_text.emit(text)
                        else:
                            # 获取部分结果
                            partial = self.recognizer.PartialResult()
                            text = self._parse_partial_result(partial)
                            if text:
                                self.new_text.emit("PARTIAL:" + text)

                        # 更新进度
                        current_time = time.time()
                        if current_time - last_progress_update >= 0.5:
                            elapsed_seconds = current_time - start_time
                            minutes = int(elapsed_seconds // 60)
                            seconds = int(elapsed_seconds % 60)
                            time_str = f"转录时长: {minutes:02d}:{seconds:02d}"
                            self.progress.emit(50, time_str)
                            last_progress_update = current_time

                    except Exception as e:
                        self.error.emit(f"音频处理错误: {str(e)}")

        except Exception as e:
            self.error.emit(f"音频捕获错误: {str(e)}")
        finally:
            self.finished.emit()

    def _parse_result(self, result):
        """解析完整识别结果"""
        if isinstance(result, str) and not result.startswith('{'):
            return result.strip()

        try:
            if isinstance(result, str):
                result_json = json.loads(result)
            elif hasattr(result, 'text'):
                result_json = {"text": result.text}
            elif hasattr(result, '__str__'):
                result_json = {"text": str(result)}
            else:
                return None

            text = result_json.get('text', '').strip()
            if text:
                text = text[0].upper() + text[1:]
                if text[-1] not in ['.', '?', '!']:
                    text += '.'
                return text
        except:
            return str(result).strip()

        return None

    def _parse_partial_result(self, partial):
        """解析部分识别结果"""
        if isinstance(partial, str) and not partial.startswith('{'):
            return partial.strip()

        try:
            if isinstance(partial, str):
                try:
                    partial_json = json.loads(partial)
                except json.JSONDecodeError:
                    return partial.strip()
            elif isinstance(partial, dict):
                partial_json = partial
            elif hasattr(partial, 'partial'):
                partial_json = {'partial': str(partial.partial)}
            else:
                partial_json = {'partial': str(partial)}

            return partial_json.get('partial', '').strip()
        except:
            return str(partial).strip()

class AudioProcessor(QObject):
    """音频处理器类"""
    # 定义 Qt 信号
    new_text_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)

    def __init__(self, signals: TranscriptionSignals):
        super().__init__()
        self.signals = signals
        self.current_device = None
        self.is_capturing = False
        self.capture_thread = None
        self.sample_rate = 16000
        self.buffer_size = 4000
        self.worker_thread = None

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
        """开始捕获音频"""
        if self.is_capturing:
            return False

        if not self.current_device:
            self.error_signal.emit("未选择音频设备")
            return False

        # 创建工作线程
        self.worker_thread = QThread()
        self.worker = AudioWorker(
            self.current_device,
            self.sample_rate,
            self.buffer_size,
            recognizer
        )
        self.worker.moveToThread(self.worker_thread)

        # 连接信号
        self.worker_thread.started.connect(self.worker.process)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # 转发信号
        self.worker.new_text.connect(lambda x: self.new_text_signal.emit(x))
        self.worker.error.connect(lambda x: self.error_signal.emit(x))
        self.worker.status.connect(lambda x: self.status_signal.emit(x))
        self.worker.progress.connect(lambda x, y: self.progress_signal.emit(x, y))

        # 启动线程
        self.is_capturing = True
        self.worker_thread.start()

        return True

    def stop_capture(self) -> bool:
        """停止捕获音频"""
        if not self.is_capturing:
            return False

        if hasattr(self, 'worker') and self.worker:
            self.worker.running = False

        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()

        self.is_capturing = False
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
                            print(f"完整结果: {result}, 类型: {type(result)}, 引擎类型: {engine_type}")
                            sherpa_logger.info(f"完整结果: {result}, 类型: {type(result)}, 引擎类型: {engine_type}")

                            # 直接使用结果文本（针对 Sherpa-ONNX 模型）
                            if isinstance(result, str) and not result.startswith('{'):
                                # 如果是纯文本字符串（不是 JSON），直接使用
                                text = result.strip()
                                print(f"直接使用文本结果: {text}")
                                sherpa_logger.info(f"直接使用文本结果: {text}")
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

                                    print(f"解析后的结果: {result_json}")
                                    sherpa_logger.info(f"解析后的结果: {result_json}")

                                    # 提取文本
                                    if 'text' in result_json and result_json['text'].strip():
                                        text = result_json['text'].strip()
                                    else:
                                        print(f"结果中没有文本或文本为空")
                                        sherpa_logger.warning(f"结果中没有文本或文本为空")
                                        continue
                                except Exception as e:
                                    print(f"解析结果错误: {e}")
                                    sherpa_logger.error(f"解析结果错误: {e}")
                                    import traceback
                                    error_trace = traceback.format_exc()
                                    sherpa_logger.error(error_trace)
                                    print(error_trace)
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
                            print(f"部分结果: {partial_result}, 类型: {type(partial_result)}, 引擎类型: {engine_type}")
                            sherpa_logger.info(f"部分结果: {partial_result}, 类型: {type(partial_result)}, 引擎类型: {engine_type}")

                            # 直接使用结果文本（针对 Sherpa-ONNX 模型）
                            if isinstance(partial_result, str) and not partial_result.startswith('{'):
                                # 如果是纯文本字符串（不是 JSON），直接使用
                                partial_text = partial_result.strip()
                                print(f"直接使用部分文本结果: {partial_text}")
                                sherpa_logger.info(f"直接使用部分文本结果: {partial_text}")
                                if partial_text:
                                    print(f"发送部分文本: {partial_text}")
                                    sherpa_logger.info(f"发送部分文本: {partial_text}")
                                    self.signals.new_text.emit("PARTIAL:" + partial_text)
                                else:
                                    print(f"部分文本为空，不发送")
                                    sherpa_logger.debug(f"部分文本为空，不发送")
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

                                    print(f"解析后的部分结果: {partial}")
                                    sherpa_logger.info(f"解析后的部分结果: {partial}")

                                    # 确保partial字段存在且有效
                                    partial_text = partial.get('partial', '').strip()
                                    if partial_text:
                                        print(f"发送部分文本: {partial_text}")
                                        sherpa_logger.info(f"发送部分文本: {partial_text}")
                                        self.signals.new_text.emit("PARTIAL:" + partial_text)
                                    else:
                                        print(f"部分结果中没有文本或文本为空")
                                        sherpa_logger.warning(f"部分结果中没有文本或文本为空")
                                except Exception as e:
                                    print(f"解析部分结果错误: {e}")
                                    sherpa_logger.error(f"解析部分结果错误: {e}")
                                    import traceback
                                    error_trace = traceback.format_exc()
                                    sherpa_logger.error(error_trace)
                                    print(error_trace)
                                    # 如果解析失败，尝试直接使用结果
                                    partial_text = str(partial_result).strip()
                                    if partial_text:
                                        print(f"发送部分文本: {partial_text}")
                                        sherpa_logger.info(f"发送部分文本: {partial_text}")
                                        self.signals.new_text.emit("PARTIAL:" + partial_text)
                                    else:
                                        print(f"部分文本为空，不发送")
                                        sherpa_logger.debug(f"部分文本为空，不发送")
                    except Exception as e:
                        error_msg = f"音频处理错误: {str(e)}"
                        self.signals.error_occurred.emit(error_msg)
                        print(f"错误: {error_msg}")
                        sherpa_logger.error(error_msg)
                        import traceback
                        error_trace = traceback.format_exc()
                        sherpa_logger.error(error_trace)
                        print(error_trace)

        except Exception as e:
            error_msg = f"音频捕获错误: {e}"
            self.signals.error_occurred.emit(error_msg)
            print(f"错误: {error_msg}")
            try:
                from src.utils.sherpa_logger import sherpa_logger
                sherpa_logger.error(error_msg)
                import traceback
                error_trace = traceback.format_exc()
                sherpa_logger.error(error_trace)
                print(error_trace)
            except ImportError:
                import traceback
                traceback.print_exc()

        finally:
            # 发送状态信号
            elapsed_time = time.time() - start_time
            self.signals.status_updated.emit(f"音频捕获已停止，持续时间: {elapsed_time:.2f}秒")

            # 确保捕获标志被清除
            self.is_capturing = False
