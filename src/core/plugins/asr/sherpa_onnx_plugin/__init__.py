"""
Sherpa-ONNX 插件包
提供基于sherpa-onnx系列模型的语音识别功能
支持多种模型变种：
- sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
- sherpa-onnx-streaming-zipformer-en-2023-06-26
"""
from .sherpa_onnx_plugin import SherpaOnnxPlugin

__all__ = ['SherpaOnnxPlugin']
