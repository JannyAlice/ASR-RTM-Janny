"""
Sherpa-ONNX ASR 引擎实现
基于 sherpa-onnx 官方测试文件的方法
"""
import os
import json
import numpy as np
from typing import Optional, Union, Dict, Any

import sherpa_onnx


class SherpaOnnxASR:
    """Sherpa-ONNX ASR 引擎实现（支持双模式加载）"""

    def __init__(self, model_id: Optional[str] = None, model_path: Optional[str] = None, model_config: Optional[Dict[str, Any]] = None):
        """
        初始化 Sherpa-ONNX ASR 引擎

        Args:
            model_path: 模型路径
            model_config: 模型配置
        """
        self.model_id = model_id
        self.model_path = model_path
        self.model_config = model_config or {}
        self.recognizer = None
        self.sample_rate = 16000  # 默认采样率
        self.is_int8 = self.model_config.get("type", "int8").lower() == "int8"
        
        # 默认配置
        self.config = {
            "encoder": os.path.join(model_path, "encoder-epoch-99-avg-1.int8.onnx" if self.is_int8 else "encoder-epoch-99-avg-1.onnx"),
            "decoder": os.path.join(model_path, "decoder-epoch-99-avg-1.onnx"),
            "joiner": os.path.join(model_path, "joiner-epoch-99-avg-1.int8.onnx" if self.is_int8 else "joiner-epoch-99-avg-1.onnx"),
            "tokens": os.path.join(model_path, "tokens.txt"),
            "num_threads": 4,
            "sample_rate": 16000,
            "feature_dim": 80,
            "decoding_method": "greedy_search",
            "debug": False,
            "enable_endpoint": 1,
            "rule1_min_trailing_silence": 2.4,
            "rule2_min_trailing_silence": 1.2,
            "rule3_utterance_length": 20
        }

    def setup(self) -> bool:
        """
        设置引擎

        Returns:
            bool: 是否设置成功
        """
        try:
            # 更新配置
            if "config" in self.model_config:
                user_config = self.model_config["config"]
                for key, value in user_config.items():
                    if key in self.config and key not in ["encoder", "decoder", "joiner", "tokens"]:
                        self.config[key] = value

            # 使用 OnlineRecognizer 类的 from_transducer 静态方法创建实例
            # 这是 sherpa-onnx 1.11.2 版本的正确 API
            try:
                self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=self.config["encoder"],
                    decoder=self.config["decoder"],
                    joiner=self.config["joiner"],
                    tokens=self.config["tokens"],
                    num_threads=self.config["num_threads"],
                    sample_rate=self.config["sample_rate"],
                    feature_dim=self.config["feature_dim"],
                    decoding_method=self.config["decoding_method"],
                    enable_endpoint_detection=self.config["enable_endpoint"],
                    rule1_min_trailing_silence=self.config["rule1_min_trailing_silence"],
                    rule2_min_trailing_silence=self.config["rule2_min_trailing_silence"],
                    rule3_min_utterance_length=self.config["rule3_utterance_length"]
                )
            except Exception as e:
                print(f"使用 from_transducer 创建实例失败: {e}")
                return False

            model_type = "int8量化" if self.is_int8 else "标准"
            print(f"Sherpa-ONNX ASR ({model_type}模型) 初始化成功")
            return True

        except Exception as e:
            print(f"设置 Sherpa-ONNX ASR 引擎失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """
        转录音频数据

        Args:
            audio_data: 音频数据，可以是字节或numpy数组

        Returns:
            str: 转录文本，如果失败则返回None
        """
        try:
            if not self.recognizer:
                return None

            # 创建一个新的流
            stream = self.recognizer.create_stream()

            # 确保音频数据是numpy数组
            if isinstance(audio_data, bytes):
                # 将字节转换为16位整数数组
                import array
                audio_array = array.array('h', audio_data)
                audio_data = np.array(audio_array, dtype=np.float32) / 32768.0

            # 确保音频数据是单声道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # 处理音频数据
            try:
                # 接受音频数据
                stream.accept_waveform(self.sample_rate, audio_data)
                
                # 添加尾部填充（这是关键步骤，来自官方测试文件）
                tail_paddings = np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
                stream.accept_waveform(self.sample_rate, tail_paddings)
                
                # 标记输入结束
                stream.input_finished()
                
                # 解码
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)
                
                # 获取结果
                result = self.recognizer.get_result(stream)
                return result if result else None
                
            except Exception as e:
                print(f"处理音频数据错误: {e}")
                import traceback
                print(traceback.format_exc())
                return None

        except Exception as e:
            print(f"Sherpa-ONNX 转录错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def get_final_result(self) -> Optional[str]:
        """
        获取最终结果

        Returns:
            str: 最终识别结果文本，如果失败则返回None
        """
        try:
            if not self.recognizer:
                return None

            # 创建一个新的流
            stream = self.recognizer.create_stream()
            
            # 标记输入结束
            stream.input_finished()
            
            # 解码
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)
            
            # 获取结果
            result = self.recognizer.get_result(stream)
            return result if result else None

        except Exception as e:
            print(f"获取 Sherpa-ONNX 最终结果错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def reset(self) -> None:
        """重置识别器状态"""
        # 不需要做任何事情，因为我们在每次转录时都会创建新的流
        pass

    def __del__(self):
        """清理资源"""
        self.recognizer = None
