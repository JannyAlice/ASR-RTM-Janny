"""
音频处理器单元测试
测试AudioProcessor类的功能
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.core.audio.audio_processor import AudioProcessor, AudioDevice
from src.core.signals import TranscriptionSignals

class TestAudioDevice(unittest.TestCase):
    """AudioDevice类的测试用例"""

    def test_init(self):
        """测试初始化"""
        device = AudioDevice("test_id", "Test Device", True)
        self.assertEqual(device.id, "test_id")
        self.assertEqual(device.name, "Test Device")
        self.assertTrue(device.is_input)

    def test_str(self):
        """测试字符串表示"""
        device = AudioDevice("test_id", "Test Device")
        self.assertEqual(str(device), "Test Device (test_id)")

class TestAudioProcessor(unittest.TestCase):
    """AudioProcessor类的测试用例"""

    def setUp(self):
        """每个测试方法执行前的设置"""
        # 创建模拟的信号对象
        self.signals = MagicMock(spec=TranscriptionSignals)
        
        # 创建AudioProcessor实例
        self.processor = AudioProcessor(self.signals)

    @patch('src.core.audio.audio_processor.sc')
    def test_get_audio_devices(self, mock_sc):
        """测试获取音频设备列表"""
        # 创建模拟的扬声器和麦克风
        mock_speaker = MagicMock()
        mock_speaker.id = "speaker_id"
        mock_speaker.name = "Test Speaker"
        
        mock_mic = MagicMock()
        mock_mic.id = "mic_id"
        mock_mic.name = "Test Microphone"
        
        # 设置模拟的soundcard模块返回值
        mock_sc.all_speakers.return_value = [mock_speaker]
        mock_sc.all_microphones.return_value = [mock_mic]
        
        # 调用方法
        devices = self.processor.get_audio_devices()
        
        # 验证结果
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0].id, "speaker_id")
        self.assertEqual(devices[0].name, "Test Speaker")
        self.assertFalse(devices[0].is_input)
        self.assertEqual(devices[1].id, "mic_id")
        self.assertEqual(devices[1].name, "Test Microphone")
        self.assertTrue(devices[1].is_input)

    def test_set_current_device(self):
        """测试设置当前设备"""
        # 创建测试设备
        device = AudioDevice("test_id", "Test Device")
        
        # 设置当前设备
        result = self.processor.set_current_device(device)
        
        # 验证结果
        self.assertTrue(result)
        self.assertEqual(self.processor.current_device, device)
        
        # 测试设置None
        result = self.processor.set_current_device(None)
        self.assertFalse(result)

    def test_start_capture_no_device(self):
        """测试在没有设置设备的情况下开始捕获"""
        # 确保没有设置当前设备
        self.processor.current_device = None
        
        # 尝试开始捕获
        result = self.processor.start_capture(MagicMock())
        
        # 验证结果
        self.assertFalse(result)
        self.signals.error_occurred.emit.assert_called_once()

    def test_start_capture_already_capturing(self):
        """测试在已经捕获的情况下开始捕获"""
        # 设置捕获标志
        self.processor.is_capturing = True
        
        # 尝试开始捕获
        result = self.processor.start_capture(MagicMock())
        
        # 验证结果
        self.assertFalse(result)

    @patch('src.core.audio.audio_processor.threading.Thread')
    def test_start_capture(self, mock_thread):
        """测试开始捕获"""
        # 设置当前设备
        self.processor.current_device = AudioDevice("test_id", "Test Device")
        
        # 设置模拟线程
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        # 调用方法
        result = self.processor.start_capture(MagicMock())
        
        # 验证结果
        self.assertTrue(result)
        self.assertTrue(self.processor.is_capturing)
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()

    def test_stop_capture_not_capturing(self):
        """测试在没有捕获的情况下停止捕获"""
        # 确保捕获标志为False
        self.processor.is_capturing = False
        
        # 调用方法
        result = self.processor.stop_capture()
        
        # 验证结果
        self.assertFalse(result)

    def test_stop_capture(self):
        """测试停止捕获"""
        # 设置捕获标志和模拟线程
        self.processor.is_capturing = True
        self.processor.capture_thread = MagicMock()
        self.processor.capture_thread.is_alive.return_value = True
        
        # 调用方法
        result = self.processor.stop_capture()
        
        # 验证结果
        self.assertTrue(result)
        self.assertFalse(self.processor.is_capturing)
        self.processor.capture_thread.join.assert_called_once()
        self.assertIsNone(self.processor.capture_thread)

if __name__ == '__main__':
    unittest.main()
