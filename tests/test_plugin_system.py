"""插件系统测试模块"""
import pytest
from pathlib import Path
import sys
import logging
from typing import Dict, Any

from src.core.plugins import PluginManager, PluginRegistry
from src.core.asr import ASRModelManager

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def plugin_manager():
    """创建插件管理器实例"""
    return PluginManager()

@pytest.fixture
def plugin_registry(plugin_manager):
    """创建插件注册表实例"""
    return plugin_manager.get_registry()

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """测试配置"""
    return {
        "asr": {
            "models": {
                "vosk_small": {
                    "path": "models/asr/vosk/vosk-model-small-en-us-0.15",
                    "type": "standard",  # 使用与config.json一致的类型
                    "enabled": True,
                    "config": {
                        "sample_rate": 16000,
                        "use_words": True,
                        "channels": 1,
                        "buffer_size": 4000
                    }
                }
            }
        }
    }

def test_plugin_loading(plugin_manager, plugin_registry, test_config):
    """测试插件加载"""
    # 注册测试插件
    from src.core.plugins.asr.vosk_plugin import VoskPlugin
    plugin_registry.register("vosk_small", VoskPlugin)
    
    # 设置插件管理器的配置
    plugin_manager.configure(test_config)
    
    # 加载插件
    plugins = plugin_manager.load_plugins("asr")
    
    # 验证是否成功加载了Vosk插件
    assert "vosk_small" in plugins
    
    # 验证插件是否正确配置
    plugin = plugins["vosk_small"]
    assert plugin._config["path"] == "models/asr/vosk/vosk-model-small-en-us-0.15"
    assert plugin._config["type"] == "vosk"

def test_model_switching(test_config):
    """测试模型切换"""
    # 创建模型管理器
    model_manager = ASRModelManager(test_config)
    
    # 测试加载模型
    result = model_manager.load_model("vosk_small")
    assert result is True

if __name__ == "__main__":
    pytest.main(["-v", __file__])