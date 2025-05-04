"""
VOSK ASR 引擎单元测试
测试 VoskASR 类的功能
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
import numpy as np
from src.core.asr.vosk_engine import VoskASR

class TestVoskASR(unittest.TestCase):
    """VoskASR 类的测试用例"""

    def setUp(self):
        """每个测试方法执行前的设置"""
        self.model_path = "test_model_path"
        self.asr = VoskASR(self.model_path)

    @patch('os.path.exists')
    @patch('src.core.asr.vosk_engine.Model')
    @patch('src.core.asr.vosk_engine.KaldiRecognizer')
    def test_setup_success(self, mock_recognizer, mock_model, mock_exists):
        """测试成功设置 VOSK ASR 引擎"""
        # 设置模拟对象
        mock_exists.return_value = True
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_recognizer_instance = MagicMock()
        mock_recognizer.return_value = mock_recognizer_instance

        # 调用方法
        result = self.asr.setup()

        # 验证结果
        self.assertTrue(result)
        mock_exists.assert_called_once_with(self.model_path)
        mock_model.assert_called_once_with(self.model_path)
        mock_recognizer.assert_called_once_with(mock_model_instance, self.asr.sample_rate)
        mock_recognizer_instance.SetWords.assert_called_once_with(True)
        self.assertEqual(self.asr.model, mock_model_instance)
        self.assertEqual(self.asr.recognizer, mock_recognizer_instance)

    @patch('os.path.exists')
    def test_setup_model_not_found(self, mock_exists):
        """测试模型路径不存在的情况"""
        # 设置模拟对象
        mock_exists.return_value = False

        # 调用方法
        result = self.asr.setup()

        # 验证结果
        self.assertFalse(result)
        mock_exists.assert_called_once_with(self.model_path)
        self.assertIsNone(self.asr.model)
        self.assertIsNone(self.asr.recognizer)

    @patch('os.path.exists')
    @patch('src.core.asr.vosk_engine.Model')
    def test_setup_exception(self, mock_model, mock_exists):
        """测试设置过程中发生异常的情况"""
        # 设置模拟对象
        mock_exists.return_value = True
        mock_model.side_effect = Exception("Test exception")

        # 调用方法
        result = self.asr.setup()

        # 验证结果
        self.assertFalse(result)
        mock_exists.assert_called_once_with(self.model_path)
        mock_model.assert_called_once_with(self.model_path)
        self.assertIsNone(self.asr.model)
        self.assertIsNone(self.asr.recognizer)

    def test_transcribe_no_recognizer(self):
        """测试在没有识别器的情况下转录"""
        # 确保没有设置识别器
        self.asr.recognizer = None

        # 调用方法
        result = self.asr.transcribe(b'test_audio_data')

        # 验证结果
        self.assertIsNone(result)

    def test_transcribe_with_numpy_array(self):
        """测试使用 numpy 数组转录"""
        # 设置模拟识别器
        self.asr.recognizer = MagicMock()
        self.asr.recognizer.AcceptWaveform.return_value = True
        self.asr.recognizer.Result.return_value = json.dumps({"text": "test transcription"})

        # 创建测试音频数据
        audio_data = np.array([0.1, 0.2, 0.3])

        # 调用方法
        result = self.asr.transcribe(audio_data)

        # 验证结果
        self.assertEqual(result, "test transcription")
        self.asr.recognizer.AcceptWaveform.assert_called_once()
        self.asr.recognizer.Result.assert_called_once()

    def test_transcribe_with_bytes(self):
        """测试使用字节数据转录"""
        # 设置模拟识别器
        self.asr.recognizer = MagicMock()
        self.asr.recognizer.AcceptWaveform.return_value = True
        self.asr.recognizer.Result.return_value = json.dumps({"text": "test transcription"})

        # 创建测试音频数据
        audio_data = b'test_audio_data'

        # 调用方法
        result = self.asr.transcribe(audio_data)

        # 验证结果
        self.assertEqual(result, "test transcription")
        self.asr.recognizer.AcceptWaveform.assert_called_once_with(audio_data)
        self.asr.recognizer.Result.assert_called_once()

    def test_transcribe_no_result(self):
        """测试转录没有结果的情况"""
        # 设置模拟识别器
        self.asr.recognizer = MagicMock()
        self.asr.recognizer.AcceptWaveform.return_value = False

        # 调用方法
        result = self.asr.transcribe(b'test_audio_data')

        # 验证结果
        self.assertIsNone(result)
        self.asr.recognizer.AcceptWaveform.assert_called_once_with(b'test_audio_data')
        self.asr.recognizer.Result.assert_not_called()

    def test_transcribe_exception(self):
        """测试转录过程中发生异常的情况"""
        # 设置模拟识别器
        self.asr.recognizer = MagicMock()
        self.asr.recognizer.AcceptWaveform.side_effect = Exception("Test exception")

        # 调用方法
        result = self.asr.transcribe(b'test_audio_data')

        # 验证结果
        self.assertIsNone(result)
        self.asr.recognizer.AcceptWaveform.assert_called_once_with(b'test_audio_data')

    def test_reset(self):
        """测试重置识别器状态"""
        # 设置模拟识别器
        self.asr.recognizer = MagicMock()

        # 调用方法
        self.asr.reset()

        # 验证结果
        self.asr.recognizer.Reset.assert_called_once()

    def test_reset_no_recognizer(self):
        """测试在没有识别器的情况下重置"""
        # 确保没有设置识别器
        self.asr.recognizer = None

        # 调用方法 - 不应该抛出异常
        self.asr.reset()

    def test_get_final_result(self):
        """测试获取最终识别结果"""
        # 设置模拟识别器
        self.asr.recognizer = MagicMock()
        self.asr.recognizer.FinalResult.return_value = json.dumps({"text": "final transcription"})

        # 调用方法
        result = self.asr.get_final_result()

        # 验证结果
        self.assertEqual(result, "final transcription")
        self.asr.recognizer.FinalResult.assert_called_once()

    def test_get_final_result_no_recognizer(self):
        """测试在没有识别器的情况下获取最终结果"""
        # 确保没有设置识别器
        self.asr.recognizer = None

        # 调用方法
        result = self.asr.get_final_result()

        # 验证结果
        self.assertIsNone(result)

    def test_get_final_result_exception(self):
        """测试获取最终结果过程中发生异常的情况"""
        # 设置模拟识别器
        self.asr.recognizer = MagicMock()
        self.asr.recognizer.FinalResult.side_effect = Exception("Test exception")

        # 调用方法
        result = self.asr.get_final_result()

        # 验证结果
        self.assertIsNone(result)
        self.asr.recognizer.FinalResult.assert_called_once()

if __name__ == '__main__':
    unittest.main()
