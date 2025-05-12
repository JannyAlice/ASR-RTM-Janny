import os
import json
import numpy as np
from typing import Optional, Union
from vosk import Model, KaldiRecognizer


class VoskASR:
    """VOSK ASR 引擎封装类"""

    def __init__(self, model_path: str):
        """初始化 VOSK ASR 引擎

        Args:
            model_path: VOSK 模型路径
        """
        # 直接使用传入的模型路径，不再从config_manager获取
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.sample_rate = 16000

        # 设置引擎类型为vosk_small
        self.engine_type = "vosk_small"

        # 自动调用setup方法初始化引擎
        self.setup()

    def setup(self) -> bool:
        """设置 VOSK ASR 引擎

        Returns:
            bool: 是否设置成功
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"VOSK model path not found: {self.model_path}")
                return False

            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            return True

        except Exception as e:
            print(f"Error setting up VOSK ASR: {str(e)}")
            return False

    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """转录音频数据

        Args:
            audio_data: 音频数据，可以是字节或 numpy 数组

        Returns:
            str: 转录文本，如果失败则返回 None
        """
        if not self.recognizer:
            return None

        try:
            # 确保音频数据是字节类型
            if isinstance(audio_data, np.ndarray):
                audio_data = (audio_data * 32767).astype(np.int16).tobytes()

            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                return result.get("text", "")
            return None

        except Exception as e:
            print(f"Error in VOSK transcription: {str(e)}")
            return None

    def reset(self) -> None:
        """重置识别器状态"""
        if self.recognizer:
            self.recognizer.Reset()

    def create_recognizer(self) -> Optional[KaldiRecognizer]:
        """创建新的识别器实例

        Returns:
            KaldiRecognizer: 新的识别器实例，如果失败则返回 None
        """
        try:
            if not self.model:
                print("模型未初始化")
                return None

            recognizer = KaldiRecognizer(self.model, self.sample_rate)
            recognizer.SetWords(True)

            # 设置引擎类型
            recognizer.engine_type = "vosk_small"

            return recognizer
        except Exception as e:
            print(f"创建识别器失败: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def get_final_result(self) -> Optional[str]:
        """获取最终识别结果

        Returns:
            str: 最终识别文本，如果失败则返回 None
        """
        try:
            if self.recognizer:
                # 获取最终结果
                final_result = self.recognizer.FinalResult()
                print(f"Vosk原始最终结果: {final_result}")

                # 解析JSON
                result = json.loads(final_result)
                text = result.get("text", "").strip()
                print(f"Vosk解析后的最终结果: {text}")

                # 格式化文本
                if text:
                    # 首字母大写
                    if len(text) > 0:
                        text = text[0].upper() + text[1:]

                    # 如果文本末尾没有标点符号，添加句号
                    if text[-1] not in ['.', '?', '!', ',', ';', ':', '-']:
                        text += '.'

                    print(f"Vosk格式化后的最终结果: {text}")
                    return text

                return None
            return None
        except Exception as e:
            print(f"Error getting VOSK final result: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    def transcribe_file(self, file_path: str) -> Optional[str]:
        """转录音频文件

        Args:
            file_path: 音频文件路径

        Returns:
            str: 转录文本，如果失败则返回 None
        """
        import wave

        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None

            # 检查文件是否为WAV格式
            if not file_path.lower().endswith('.wav'):
                print(f"File is not a WAV file: {file_path}")
                # 可以在这里添加转换为WAV格式的代码
                return None

            # 打开WAV文件
            with wave.open(file_path, 'rb') as wf:
                # 检查采样率
                if wf.getframerate() != self.sample_rate:
                    print(f"Sample rate mismatch: {wf.getframerate()} != {self.sample_rate}")
                    # 可以在这里添加重采样的代码
                    return None

                # 创建新的识别器
                recognizer = KaldiRecognizer(self.model, self.sample_rate)
                recognizer.SetWords(True)

                # 设置引擎类型
                recognizer.engine_type = "vosk_small"

                # 读取音频数据并进行识别
                results = []
                chunk_size = 4000  # 每次读取的帧数

                # 读取整个文件的所有帧
                all_frames = []
                while True:
                    frames = wf.readframes(chunk_size)
                    if not frames:
                        break
                    all_frames.append(frames)

                # 重置文件指针
                wf.rewind()

                # 先处理前几个块，确保开头部分被正确识别
                for i in range(min(5, len(all_frames))):
                    recognizer.AcceptWaveform(all_frames[i])

                # 处理所有音频块
                for frames in all_frames:
                    if recognizer.AcceptWaveform(frames):
                        result = json.loads(recognizer.Result())
                        if result.get("text", "").strip():
                            results.append(result.get("text", ""))

                # 获取最终结果
                final_result_str = recognizer.FinalResult()
                print(f"文件转录最终结果原始字符串: {final_result_str}")

                final_result = json.loads(final_result_str)
                final_text = final_result.get("text", "").strip()
                print(f"文件转录最终结果解析后: {final_text}")

                if final_text:
                    # 将最终结果添加到结果列表
                    results.append(final_text)

                # 合并所有结果
                if results:
                    combined_result = " ".join(results)

                    # 格式化合并后的文本
                    if len(combined_result) > 0:
                        combined_result = combined_result[0].upper() + combined_result[1:]
                    if combined_result[-1] not in ['.', '?', '!', ',', ';', ':', '-']:
                        combined_result += '.'

                    print(f"文件转录合并结果: {combined_result}")
                    return combined_result
                else:
                    print("没有获取到任何转录结果")
                    return None

        except Exception as e:
            print(f"Error in VOSK file transcription: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
