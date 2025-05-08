import pytest
from pathlib import Path
import logging
import wave

from src.core.plugins import PluginManager
from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

@pytest.fixture
def setup_plugin_system():
    """设置插件系统"""
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # 初始化插件系统
    plugin_manager = PluginManager()
    plugin_manager.configure(config)
    
    # 获取插件注册表
    plugin_registry = plugin_manager.get_registry()
    
    return config, plugin_manager, plugin_registry

def test_vosk_transcription(setup_plugin_system, tmp_path):
    """测试 Vosk 转录功能"""
    config, plugin_manager, plugin_registry = setup_plugin_system
    
    # 创建测试音频文件
    audio_file = tmp_path / "test.wav"
    create_test_audio_file(audio_file)
    
    # 注册 Vosk 插件
    from src.core.plugins.asr.vosk_plugin import VoskPlugin
    plugin_registry.register("vosk_small", VoskPlugin)
    
    # 加载插件
    plugins = plugin_manager.load_plugins("asr")
    assert "vosk_small" in plugins
    
    # 获取插件实例并进行转录
    plugin = plugins["vosk_small"]
    result = plugin.transcribe_file(str(audio_file))
    
    # 验证结果
    assert result is not None
    assert len(result) > 0

def create_test_audio_file(file_path: Path):
    """创建测试音频文件"""
    with wave.open(str(file_path), 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16位
        wf.setframerate(16000)  # 16kHz
        wf.writeframes(b'\x00' * 32000)  # 1秒静音