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

                    # 转换为16位整数
                    data = (data * 32768).astype(np.int16).tobytes()

                    # 处理音频数据
                    try:
                        if recognizer.AcceptWaveform(data):
                            result = recognizer.Result()
                            # 兼容字符串和字典两种格式
                            if isinstance(result, str):
                                result_json = json.loads(result)
                            else:
                                result_json = result

                            if 'text' in result_json and result_json['text'].strip():
                                # 添加标点符号和首字母大写
                                text = result_json['text'].strip()
                                text = text[0].upper() + text[1:]
                                if text[-1] not in ['.', '?', '!']:
                                    text += '.'
                                self.signals.new_text.emit(text)
                        else:
                            partial_result = recognizer.PartialResult()

                            # 统一处理所有可能的返回格式
                            if isinstance(partial_result, str):
                                partial = json.loads(partial_result)
                            elif isinstance(partial_result, dict):
                                partial = partial_result
                            elif hasattr(partial_result, 'partial'):
                                partial = {'partial': str(partial_result.partial)}
                            else:
                                partial = {'partial': str(partial_result)}

                            # 确保partial字段存在且有效
                            partial_text = partial.get('partial', '').strip()
                            if partial_text:
                                self.signals.new_text.emit("PARTIAL:" + partial_text)
                    except Exception as e:
                        self.signals.error_occurred.emit(f"音频处理错误: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        partial_json = json.loads(partial)

                        if 'partial' in partial_json and partial_json['partial'].strip():
                            self.signals.new_text.emit(f"[部分] {partial_json['partial']}")

        except Exception as e:
            self.signals.error_occurred.emit(f"音频捕获错误: {e}")

        finally:
            # 发送状态信号
            elapsed_time = time.time() - start_time
            self.signals.status_updated.emit(f"音频捕获已停止，持续时间: {elapsed_time:.2f}秒")

            # 确保捕获标志被清除
            self.is_capturing = False
