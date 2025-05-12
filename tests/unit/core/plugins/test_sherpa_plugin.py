import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
import numpy as np
from src.core.plugins.asr.sherpa_plugin import SherpaPlugin

class TestSherpaPlugin(unittest.TestCase):
    """Sherpa 插件测试用例"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.plugin = SherpaPlugin()
        self.test_config = {
            "model_path": "models/asr/sherpa/sherpa_0626_std",
            "type": "standard",
            "config": {
                "sample_rate": 16000,
                "feature_dim": 80,
                "num_threads": 1,
                "debug": False,
                "enable_endpoint_detection": True
            }
        }
        
    @patch('sherpa_onnx.OnlineRecognizer')
    def test_setup(self, mock_recognizer):
        """测试插件初始化"""
        # 设置模拟对象
        mock_recognizer_instance = MagicMock()
        mock_recognizer.from_transducer.return_value = mock_recognizer_instance
        
        # 初始化插件
        self.plugin._config = self.test_config
        
        with patch('os.path.exists', return_value=True):
            result = self.plugin.setup()
            
        # 验证结果
        self.assertTrue(result)
        self.assertEqual(self.plugin.recognizer, mock_recognizer_instance)
        
        # 验证调用
        mock_recognizer.from_transducer.assert_called_once()
        call_args = mock_recognizer.from_transducer.call_args[1]
        self.assertEqual(call_args['sample_rate'], 16000)
        self.assertEqual(call_args['feature_dim'], 80)
        
    def test_process_audio(self):
        """测试音频处理"""
        # 创建模拟识别器
        mock_recognizer = MagicMock()
        mock_stream = MagicMock()
        mock_stream.result.text = "test transcription"
        mock_stream.result.is_final = True
        mock_recognizer.create_stream.return_value = mock_stream
        
        self.plugin.recognizer = mock_recognizer
        
        # 创建测试音频数据
        audio_data = np.zeros(1600, dtype=np.float32)
        
        # 处理音频
        result = self.plugin.process_audio(audio_data)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "test transcription")
        self.assertTrue(result["is_final"])
        
        # 验证调用
        mock_recognizer.create_stream.assert_called_once()
        mock_stream.accept_waveform.assert_called_once_with(audio_data)
        
    def test_get_info(self):
        """测试获取插件信息"""
        info = self.plugin.get_info()
        
        self.assertEqual(info["id"], "sherpa_0626_std")
        self.assertEqual(info["type"], "ASR")
        self.assertTrue(info["version"])
        
    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.plugin.cleanup()

if __name__ == '__main__':
    unittest.main()