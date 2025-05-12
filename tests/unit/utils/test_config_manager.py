"""
配置管理器单元测试
测试ConfigManager类的功能
"""
import unittest
import os
import json
import tempfile
import shutil
from src.utils.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    """ConfigManager类的测试用例"""

    def setUp(self):
        """每个测试方法执行前的设置"""
        # 创建临时目录用于测试
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.test_dir, "config")
        self.config_file = "test_settings.json"
        self.config_path = os.path.join(self.config_dir, self.config_file)
        
        # 创建测试用的ConfigManager实例
        self.config_manager = ConfigManager(self.config_dir, self.config_file)

    def tearDown(self):
        """每个测试方法执行后的清理"""
        # 删除临时目录
        shutil.rmtree(self.test_dir)

    def test_init_creates_config_dir(self):
        """测试初始化时是否创建配置目录"""
        self.assertTrue(os.path.exists(self.config_dir), "配置目录应该被创建")

    def test_default_config(self):
        """测试默认配置是否正确加载"""
        # 获取默认配置
        default_config = self.config_manager._get_default_config()
        
        # 检查默认配置的关键部分
        self.assertIn("window", default_config, "默认配置应包含window部分")
        self.assertIn("ui", default_config, "默认配置应包含ui部分")
        self.assertIn("transcription", default_config, "默认配置应包含transcription部分")
        
        # 检查window部分的具体配置项
        window_config = default_config["window"]
        self.assertEqual(window_config["width"], 800, "窗口宽度应为800")
        self.assertEqual(window_config["height"], 600, "窗口高度应为600")

    def test_save_and_load_config(self):
        """测试保存和加载配置"""
        # 更新配置
        test_values = {"test_key": "test_value"}
        self.config_manager.update_config("test_section", test_values)
        
        # 保存配置
        self.config_manager.save_config()
        
        # 检查配置文件是否存在
        self.assertTrue(os.path.exists(self.config_path), "配置文件应该被创建")
        
        # 创建新的ConfigManager实例加载配置
        new_config_manager = ConfigManager(self.config_dir, self.config_file)
        
        # 检查配置是否正确加载
        loaded_value = new_config_manager.get_config("test_section", {}).get("test_key")
        self.assertEqual(loaded_value, "test_value", "加载的配置值应与保存的一致")

    def test_get_config(self):
        """测试获取配置"""
        # 设置测试配置
        self.config_manager.config = {
            "test_section": {
                "test_key": "test_value"
            }
        }
        
        # 测试获取存在的配置
        value = self.config_manager.get_config("test_section", {}).get("test_key")
        self.assertEqual(value, "test_value", "应返回正确的配置值")
        
        # 测试获取不存在的配置
        value = self.config_manager.get_config("non_existent_section", "default_value")
        self.assertEqual(value, "default_value", "对于不存在的部分应返回默认值")

    def test_get_ui_config(self):
        """测试获取UI配置"""
        # 设置测试UI配置
        self.config_manager.config = {
            "ui": {
                "test_key": "test_value"
            }
        }
        
        # 测试获取存在的UI配置
        value = self.config_manager.get_ui_config("test_key")
        self.assertEqual(value, "test_value", "应返回正确的UI配置值")
        
        # 测试获取不存在的UI配置
        value = self.config_manager.get_ui_config("non_existent_key", "default_value")
        self.assertEqual(value, "default_value", "对于不存在的键应返回默认值")

    def test_update_config(self):
        """测试更新配置"""
        # 初始配置为空
        self.config_manager.config = {}
        
        # 更新配置
        self.config_manager.update_config("test_section", {"test_key": "test_value"})
        
        # 检查配置是否更新
        self.assertIn("test_section", self.config_manager.config, "配置应包含更新的部分")
        self.assertEqual(
            self.config_manager.config["test_section"]["test_key"], 
            "test_value", 
            "配置值应被正确更新"
        )
        
        # 再次更新同一部分
        self.config_manager.update_config("test_section", {"another_key": "another_value"})
        
        # 检查是否保留原有配置并添加新配置
        self.assertEqual(
            self.config_manager.config["test_section"]["test_key"], 
            "test_value", 
            "原有配置值应被保留"
        )
        self.assertEqual(
            self.config_manager.config["test_section"]["another_key"], 
            "another_value", 
            "新配置值应被添加"
        )

    def test_update_and_save(self):
        """测试更新并保存配置"""
        # 更新并保存配置
        self.config_manager.update_and_save("test_section", {"test_key": "test_value"})
        
        # 检查配置文件是否存在
        self.assertTrue(os.path.exists(self.config_path), "配置文件应该被创建")
        
        # 读取配置文件内容
        with open(self.config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        
        # 检查保存的配置是否正确
        self.assertIn("test_section", saved_config, "保存的配置应包含更新的部分")
        self.assertEqual(
            saved_config["test_section"]["test_key"], 
            "test_value", 
            "保存的配置值应正确"
        )

if __name__ == '__main__':
    unittest.main()
