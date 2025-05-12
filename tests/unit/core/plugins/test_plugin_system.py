import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
from src.core.plugins.base.plugin_initializer import PluginInitializer
from src.core.plugins.base.plugin_registry import PluginRegistry
from src.core.plugins.asr.vosk_plugin import VoskPlugin
from src.core.plugins.asr.asr_plugin_adapter import ASRPluginAdapter

class TestPluginSystem(unittest.TestCase):
    """插件系统集成测试"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建临时配置文件
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.plugins_config_path = os.path.join(self.temp_dir, "plugins.json")
        
        # 创建测试用的核心配置
        self.core_config = {
            "asr": {
                "models": {
                    "vosk_small": {
                        "path": "models/asr/vosk/vosk-model-small-en-us-0.15",
                        "type": "standard",
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
        
        # 创建测试用的插件配置
        self.plugins_config = {
            "plugins": {
                "asr": {
                    "vosk_small": {
                        "enabled": True,
                        "type": "asr",
                        "model_config": "vosk_small",
                        "plugin_config": {
                            "use_words": True,
                            "show_confidence": False,
                            "buffer_size": 8000
                        }
                    }
                }
            },
            "plugin_system": {
                "version": "1.0.0",
                "logging": {
                    "enabled": True,
                    "level": "DEBUG",
                    "file": os.path.join(self.temp_dir, "test_plugins.log")
                }
            }
        }
        
        # 写入配置文件
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.core_config, f, indent=4)
            
        with open(self.plugins_config_path, 'w', encoding='utf-8') as f:
            json.dump(self.plugins_config, f, indent=4)
            
        # 初始化插件系统
        self.initializer = PluginInitializer(self.plugins_config_path)
        # 清理已存在的插件注册
        PluginRegistry()._plugin_classes.clear()
        PluginRegistry()._loaded_plugins.clear()
        
    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        # 清理临时文件和目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
    def test_config_loading(self):
        """测试配置加载功能"""
        # 测试加载插件配置
        plugins_config = self.initializer._load_config(self.plugins_config_path)
        self.assertIn("plugins", plugins_config)
        self.assertIn("asr", plugins_config["plugins"])
        
        # 测试加载模型配置
        model_config = self.initializer._get_model_config("vosk_small")
        self.assertIsNotNone(model_config)
        self.assertEqual(model_config["type"], "standard")
        
        # 测试配置合并
        plugin_config = plugins_config["plugins"]["asr"]["vosk_small"]
        merged_config = self.initializer._merge_configs(plugin_config, model_config)
        
        # 验证合并后的配置
        self.assertEqual(merged_config["model_path"], 
                        "models/asr/vosk/vosk-model-small-en-us-0.15")
        self.assertEqual(merged_config["config"]["buffer_size"], 8000)  # 插件配置覆盖了模型配置
        
    @patch('src.core.plugins.asr.vosk_plugin.Model')
    @patch('src.core.plugins.asr.vosk_plugin.KaldiRecognizer')
    def test_plugin_initialization(self, mock_recognizer, mock_model):
        """测试插件初始化"""
        # 设置模拟对象
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_recognizer_instance = MagicMock()
        mock_recognizer.return_value = mock_recognizer_instance
        
        # 初始化插件系统
        with patch('os.path.exists', return_value=True):
            loaded_plugins = self.initializer.initialize_plugins()
        
        # 验证是否成功加载Vosk插件
        self.assertIn("vosk_small", loaded_plugins)
        
        # 获取插件实例
        registry = PluginRegistry()
        plugin = registry.get_plugin("vosk_small")
        
        # 验证插件实例
        self.assertIsNotNone(plugin)
        self.assertIsInstance(plugin, VoskPlugin)
        
    @patch('src.core.plugins.asr.vosk_plugin.Model')
    @patch('src.core.plugins.asr.vosk_plugin.KaldiRecognizer')
    def test_asr_adapter(self, mock_recognizer, mock_model):
        """测试ASR插件适配器"""
        # 设置模拟对象
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_recognizer_instance = MagicMock()
        mock_recognizer.return_value = mock_recognizer_instance
        
        # 初始化插件系统
        with patch('os.path.exists', return_value=True):
            self.initializer.initialize_plugins()
        
        # 创建适配器
        adapter = ASRPluginAdapter()
        
        # 测试初始化引擎
        with patch('os.path.exists', return_value=True):
            result = adapter.initialize_engine("vosk_small")
            self.assertTrue(result)
        
        # 验证当前引擎类型
        engine_type = adapter.get_current_engine_type()
        self.assertEqual(engine_type, "vosk_small")
        
        # 清理
        adapter.cleanup()
        
    def test_plugin_hot_reload(self):
        """测试插件热重载功能"""
        # 修改插件配置
        self.plugins_config["plugins"]["asr"]["vosk_small"]["plugin_config"]["buffer_size"] = 16000
        
        # 重写配置文件
        with open(self.plugins_config_path, 'w', encoding='utf-8') as f:
            json.dump(self.plugins_config, f, indent=4)
            
        # 重新初始化插件
        with patch('os.path.exists', return_value=True):
            loaded_plugins = self.initializer.initialize_plugins()
            
        # 验证插件是否重新加载
        self.assertIn("vosk_small", loaded_plugins)
        
        # 获取插件实例
        registry = PluginRegistry()
        plugin = registry.get_plugin("vosk_small")
        
        # 验证新配置是否生效
        self.assertEqual(
            plugin._config["config"]["buffer_size"], 
            16000
        )
        
if __name__ == '__main__':
    unittest.main()