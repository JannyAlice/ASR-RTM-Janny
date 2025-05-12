"""
ASR 模型管理器单元测试
测试 ASRModelManager 类的功能
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
import numpy as np
from src.core.asr.model_manager import ASRModelManager
from src.core.asr.vosk_engine import VoskASR
from src.core.asr.sherpa_engine import SherpaOnnxASR

class TestASRModelManager(unittest.TestCase):
    """ASRModelManager 类的测试用例"""

    def setUp(self):
        """每个测试方法执行前的设置"""
        # 创建模拟的配置管理器
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_config.return_value = {
            'models': {
                'vosk': {
                    'path': '/path/to/vosk/model',
                    'enabled': True
                },
                'sherpa': {
                    'path': '/path/to/sherpa/model',
                    'enabled': True
                }
            }
        }
        
        # 创建补丁
        self.config_patcher = patch('src.core.asr.model_manager.config_manager', self.mock_config_manager)
        
        # 启动补丁
        self.config_patcher.start()
        
        # 创建 ASRModelManager 实例
        self.manager = ASRModelManager()

    def tearDown(self):
        """每个测试方法执行后的清理"""
        # 停止补丁
        self.config_patcher.stop()

    @patch('os.path.exists')
    @patch('src.core.asr.model_manager.vosk.Model')
    def test_load_vosk_model(self, mock_vosk_model, mock_exists):
        """测试加载 VOSK 模型"""
        # 设置模拟对象
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_vosk_model.return_value = mock_model
        
        # 调用方法
        result = self.manager.load_model('vosk')
        
        # 验证结果
        self.assertTrue(result)
        mock_exists.assert_called_once_with('/path/to/vosk/model')
        mock_vosk_model.assert_called_once_with('/path/to/vosk/model')
        self.assertEqual(self.manager.current_model, mock_model)
        self.assertEqual(self.manager.model_path, '/path/to/vosk/model')
        self.assertEqual(self.manager.model_type, 'vosk')

    @patch('os.path.exists')
    def test_load_model_not_found(self, mock_exists):
        """测试加载不存在的模型"""
        # 设置模拟对象
        mock_exists.return_value = False
        
        # 调用方法
        result = self.manager.load_model('vosk')
        
        # 验证结果
        self.assertFalse(result)
        mock_exists.assert_called_once_with('/path/to/vosk/model')
        self.assertIsNone(self.manager.current_model)
        self.assertIsNone(self.manager.model_path)
        self.assertIsNone(self.manager.model_type)

    def test_load_model_invalid_name(self):
        """测试加载无效名称的模型"""
        # 调用方法
        result = self.manager.load_model('invalid_model')
        
        # 验证结果
        self.assertFalse(result)
        self.assertIsNone(self.manager.current_model)
        self.assertIsNone(self.manager.model_path)
        self.assertIsNone(self.manager.model_type)

    @patch('os.path.exists')
    @patch('src.core.asr.model_manager.vosk.Model')
    @patch('src.core.asr.model_manager.vosk.KaldiRecognizer')
    def test_create_recognizer_vosk(self, mock_recognizer, mock_model, mock_exists):
        """测试创建 VOSK 识别器"""
        # 设置模拟对象
        mock_exists.return_value = True
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_recognizer_instance = MagicMock()
        mock_recognizer.return_value = mock_recognizer_instance
        
        # 加载模型
        self.manager.load_model('vosk')
        
        # 调用方法
        result = self.manager.create_recognizer()
        
        # 验证结果
        self.assertEqual(result, mock_recognizer_instance)
        mock_recognizer.assert_called_once_with(mock_model_instance, 16000)

    def test_create_recognizer_no_model(self):
        """测试在没有加载模型的情况下创建识别器"""
        # 确保没有加载模型
        self.manager.current_model = None
        
        # 调用方法
        result = self.manager.create_recognizer()
        
        # 验证结果
        self.assertIsNone(result)

    @patch('os.path.exists')
    def test_check_model_directory(self, mock_exists):
        """测试检查模型目录"""
        # 设置模拟对象
        mock_exists.side_effect = [True, False]
        
        # 调用方法
        result = self.manager.check_model_directory()
        
        # 验证结果
        self.assertEqual(result, {'vosk': True, 'sherpa': False})
        self.assertEqual(mock_exists.call_count, 2)

    @patch('src.core.asr.model_manager.VoskASR')
    def test_initialize_engine_vosk(self, mock_vosk_asr):
        """测试初始化 VOSK 引擎"""
        # 设置模拟对象
        mock_engine = MagicMock()
        mock_engine.setup.return_value = True
        mock_vosk_asr.return_value = mock_engine
        
        # 设置配置
        self.mock_config_manager.get_config.return_value = {
            'path': '/path/to/vosk/model',
            'enabled': True
        }
        
        # 调用方法
        result = self.manager.initialize_engine('vosk')
        
        # 验证结果
        self.assertTrue(result)
        mock_vosk_asr.assert_called_once_with('/path/to/vosk/model')
        mock_engine.setup.assert_called_once()
        self.assertEqual(self.manager.current_engine, mock_engine)

    @patch('src.core.asr.model_manager.SherpaOnnxASR')
    def test_initialize_engine_sherpa(self, mock_sherpa_asr):
        """测试初始化 Sherpa 引擎"""
        # 设置模拟对象
        mock_engine = MagicMock()
        mock_engine.setup.return_value = True
        mock_sherpa_asr.return_value = mock_engine
        
        # 设置配置
        self.mock_config_manager.get_config.return_value = {
            'path': '/path/to/sherpa/model',
            'enabled': True
        }
        
        # 调用方法
        result = self.manager.initialize_engine('sherpa')
        
        # 验证结果
        self.assertTrue(result)
        mock_sherpa_asr.assert_called_once_with('/path/to/sherpa/model')
        mock_engine.setup.assert_called_once()
        self.assertEqual(self.manager.current_engine, mock_engine)

    def test_initialize_engine_invalid(self):
        """测试初始化无效的引擎"""
        # 调用方法
        result = self.manager.initialize_engine('invalid_engine')
        
        # 验证结果
        self.assertFalse(result)
        self.assertIsNone(self.manager.current_engine)

    def test_initialize_engine_disabled(self):
        """测试初始化被禁用的引擎"""
        # 设置配置
        self.mock_config_manager.get_config.return_value = {
            'path': '/path/to/vosk/model',
            'enabled': False
        }
        
        # 调用方法
        result = self.manager.initialize_engine('vosk')
        
        # 验证结果
        self.assertFalse(result)
        self.assertIsNone(self.manager.current_engine)

    def test_transcribe_no_engine(self):
        """测试在没有初始化引擎的情况下转录"""
        # 确保没有初始化引擎
        self.manager.current_engine = None
        
        # 调用方法
        result = self.manager.transcribe(b'test_audio_data')
        
        # 验证结果
        self.assertIsNone(result)

    def test_transcribe(self):
        """测试转录音频数据"""
        # 设置模拟引擎
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "test transcription"
        self.manager.current_engine = mock_engine
        
        # 调用方法
        result = self.manager.transcribe(b'test_audio_data')
        
        # 验证结果
        self.assertEqual(result, "test transcription")
        mock_engine.transcribe.assert_called_once_with(b'test_audio_data')

    def test_reset(self):
        """测试重置引擎状态"""
        # 设置模拟引擎
        mock_engine = MagicMock()
        self.manager.current_engine = mock_engine
        
        # 调用方法
        self.manager.reset()
        
        # 验证结果
        mock_engine.reset.assert_called_once()

    def test_reset_no_engine(self):
        """测试在没有初始化引擎的情况下重置"""
        # 确保没有初始化引擎
        self.manager.current_engine = None
        
        # 调用方法 - 不应该抛出异常
        self.manager.reset()

    def test_get_final_result(self):
        """测试获取最终识别结果"""
        # 设置模拟引擎
        mock_engine = MagicMock()
        mock_engine.get_final_result.return_value = "final transcription"
        self.manager.current_engine = mock_engine
        
        # 调用方法
        result = self.manager.get_final_result()
        
        # 验证结果
        self.assertEqual(result, "final transcription")
        mock_engine.get_final_result.assert_called_once()

    def test_get_final_result_no_engine(self):
        """测试在没有初始化引擎的情况下获取最终结果"""
        # 确保没有初始化引擎
        self.manager.current_engine = None
        
        # 调用方法
        result = self.manager.get_final_result()
        
        # 验证结果
        self.assertIsNone(result)

    def test_get_current_engine_type_vosk(self):
        """测试获取当前引擎类型为 VOSK"""
        # 设置模拟引擎
        self.manager.current_engine = MagicMock(spec=VoskASR)
        
        # 调用方法
        result = self.manager.get_current_engine_type()
        
        # 验证结果
        self.assertEqual(result, "vosk")

    def test_get_current_engine_type_sherpa(self):
        """测试获取当前引擎类型为 Sherpa"""
        # 设置模拟引擎
        self.manager.current_engine = MagicMock(spec=SherpaOnnxASR)
        
        # 调用方法
        result = self.manager.get_current_engine_type()
        
        # 验证结果
        self.assertEqual(result, "sherpa")

    def test_get_current_engine_type_none(self):
        """测试在没有初始化引擎的情况下获取引擎类型"""
        # 确保没有初始化引擎
        self.manager.current_engine = None
        
        # 调用方法
        result = self.manager.get_current_engine_type()
        
        # 验证结果
        self.assertIsNone(result)

    def test_get_available_engines(self):
        """测试获取可用的引擎列表"""
        # 设置配置
        self.mock_config_manager.get_config.side_effect = [
            {'path': '/path/to/vosk/model', 'enabled': True},
            {'path': '/path/to/sherpa/model', 'enabled': False}
        ]
        
        # 调用方法
        result = self.manager.get_available_engines()
        
        # 验证结果
        self.assertEqual(result, {'vosk': True, 'sherpa': False})
        self.assertEqual(self.mock_config_manager.get_config.call_count, 2)

if __name__ == '__main__':
    unittest.main()
