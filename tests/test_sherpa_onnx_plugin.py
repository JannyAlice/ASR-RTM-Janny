"""
测试Sherpa-ONNX插件
"""
import os
import sys
import time
import logging
import unittest
import numpy as np
from typing import Dict, Any

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入配置管理器
from src.utils.config_manager import ConfigManager
from src.core.plugins.asr.sherpa_onnx_plugin import SherpaOnnxPlugin

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestSherpaOnnxPlugin(unittest.TestCase):
    """测试Sherpa-ONNX插件"""

    def setUp(self):
        """测试前的准备工作"""
        # 加载配置
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        # 获取sherpa_onnx_std模型配置
        self.model_config = self.config_manager.get_model_config('sherpa_onnx_std')
        self.assertIsNotNone(self.model_config, "未找到sherpa_onnx_std模型配置")
        
        # 创建插件实例
        self.plugin = SherpaOnnxPlugin(self.model_config)
        
    def tearDown(self):
        """测试后的清理工作"""
        if hasattr(self, 'plugin') and self.plugin:
            self.plugin.cleanup()
            
    def test_plugin_initialization(self):
        """测试插件初始化"""
        # 检查插件ID
        self.assertEqual(self.plugin.get_id(), "sherpa_onnx_std")
        
        # 检查插件名称
        self.assertTrue("Sherpa-ONNX" in self.plugin.get_name())
        
        # 检查插件版本
        self.assertEqual(self.plugin.get_version(), "1.0.0")
        
        # 检查插件描述
        self.assertTrue("Sherpa-ONNX" in self.plugin.get_description())
        
        # 检查插件作者
        self.assertEqual(self.plugin.get_author(), "RealtimeTrans Team")
        
    def test_plugin_setup(self):
        """测试插件设置"""
        # 初始化插件
        result = self.plugin.initialize()
        self.assertTrue(result, "插件初始化失败")
        
        # 检查模型是否已加载
        self.assertIsNotNone(self.plugin.recognizer, "识别器未初始化")
        
        # 检查模型信息
        model_info = self.plugin.get_model_info()
        self.assertIsNotNone(model_info, "未获取到模型信息")
        self.assertEqual(model_info['engine'], "sherpa_onnx_std")
        
    def test_process_audio(self):
        """测试处理音频数据"""
        # 初始化插件
        result = self.plugin.initialize()
        self.assertTrue(result, "插件初始化失败")
        
        # 创建一个空白音频数据（1秒，16kHz采样率）
        audio_data = np.zeros(16000, dtype=np.int16)
        
        # 处理音频数据
        result = self.plugin.process_audio(audio_data)
        self.assertIsNotNone(result, "处理音频数据失败")
        
        # 检查结果
        self.assertIn("text", result, "结果中缺少text字段")
        self.assertIn("is_final", result, "结果中缺少is_final字段")
        
    def test_transcribe(self):
        """测试转录方法"""
        # 初始化插件
        result = self.plugin.initialize()
        self.assertTrue(result, "插件初始化失败")
        
        # 创建一个空白音频数据（1秒，16kHz采样率）
        audio_data = np.zeros(16000, dtype=np.int16)
        
        # 转录音频数据
        text = self.plugin.transcribe(audio_data)
        self.assertIsInstance(text, str, "转录结果不是字符串")
        
    def test_reset_stream(self):
        """测试重置流"""
        # 初始化插件
        result = self.plugin.initialize()
        self.assertTrue(result, "插件初始化失败")
        
        # 创建一个空白音频数据（1秒，16kHz采样率）
        audio_data = np.zeros(16000, dtype=np.int16)
        
        # 处理音频数据
        self.plugin.process_audio(audio_data)
        
        # 重置流
        self.plugin.reset_stream()
        self.assertIsNone(self.plugin.stream, "流未重置")
        
    def test_validate_files(self):
        """测试验证文件"""
        # 初始化插件
        result = self.plugin.initialize()
        self.assertTrue(result, "插件初始化失败")
        
        # 验证文件
        result = self.plugin.validate_files()
        self.assertTrue(result, "文件验证失败")
        
    def test_get_formatted_transcript(self):
        """测试格式化转录文本"""
        # 初始化插件
        result = self.plugin.initialize()
        self.assertTrue(result, "插件初始化失败")
        
        # 测试空文本
        text = self.plugin.get_formatted_transcript("")
        self.assertEqual(text, "", "空文本格式化错误")
        
        # 测试小写文本
        text = self.plugin.get_formatted_transcript("hello world")
        self.assertEqual(text, "Hello world.", "小写文本格式化错误")
        
        # 测试已格式化文本
        text = self.plugin.get_formatted_transcript("Hello world.")
        self.assertEqual(text, "Hello world.", "已格式化文本处理错误")
        
if __name__ == '__main__':
    unittest.main()
