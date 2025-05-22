"""插件系统测试模块"""
import pytest
from pathlib import Path
import sys
import logging
from typing import Dict, Any

from src.core.plugins import PluginManager, PluginRegistry
from src.core.asr import ASRModelManager
from src.utils.config_manager import config_manager

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
    # 初始化 vosk_small 配置，避免 NoneType 错误
    from src.utils.config_manager import config_manager
    config_manager.set_config({
        "name": "VOSK Small Model",
        "path": "/mock/absolute/path/to/vosk-model-small-en-us-0.15",  # 可根据实际测试环境调整
        "type": "vosk",
        "config": {
            "sample_rate": 16000,
            "use_words": True
        }
    }, 'asr', 'models', 'vosk_small')
    return {
        "asr": {
            "models": {
                "vosk_small": {
                    "path": config_manager.get_model_config('vosk_small')["path"],  # 禁止硬编码，统一由配置管理
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
    
    # 验证插件是否正确配置（直接与 mock 值比较，避免 config_manager 多实例问题）
    plugin = plugins["vosk_small"]
    assert plugin._config["path"] == "/mock/absolute/path/to/vosk-model-small-en-us-0.15"
    assert plugin._config["type"] == "standard"


def test_model_switching(test_config):
    """测试模型切换"""
    # 创建模型管理器（无参实例化）
    model_manager = ASRModelManager()
    # 直接 mock models_config 属性
    model_manager.models_config = test_config['asr']['models']
    # 测试加载模型
    result = model_manager.load_model("vosk_small")
    assert result is True

if __name__ == "__main__":
    pytest.main(["-v", __file__])