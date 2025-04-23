#!/usr/bin/env python3
"""
配置管理模块
负责加载、保存和管理应用程序配置
"""
import os
import json
import logging
from typing import Dict, Any

class ConfigManager:
    """配置管理器类，处理应用程序配置的加载和保存"""

    def __init__(self, config_dir: str = "config", config_file: str = "settings.json"):
        """
        初始化配置管理器

        参数:
            config_dir: 配置文件目录
            config_file: 配置文件名
        """
        self.config_dir = config_dir
        self.config_file = config_file
        self.config_path = os.path.join(config_dir, config_file)
        self.config = {}

        # 确保配置目录存在
        os.makedirs(config_dir, exist_ok=True)

        # 加载配置
        self.load_config()

    def load_config(self) -> None:
        """从文件加载配置"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print(f"[DEBUG] ConfigManager.load_config: 配置已从 {self.config_path} 加载")
                print(f"[DEBUG] ConfigManager.load_config: config = {self.config}")
                logging.info(f"配置已从 {self.config_path} 加载")
            else:
                print(f"[DEBUG] ConfigManager.load_config: 配置文件 {self.config_path} 不存在，将使用默认配置")
                logging.info(f"配置文件 {self.config_path} 不存在，将使用默认配置")
                self.config = self._get_default_config()
                self.save_config()  # 保存默认配置
        except Exception as e:
            print(f"[DEBUG] ConfigManager.load_config: 加载配置时出错: {e}")
            logging.error(f"加载配置时出错: {e}")
            self.config = self._get_default_config()

    def save_config(self) -> None:
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            logging.info(f"配置已保存到 {self.config_path}")
        except Exception as e:
            logging.error(f"保存配置时出错: {e}")

    def get_config(self, section: str, default: Any = None) -> Any:
        """
        获取指定部分的配置

        参数:
            section: 配置部分名称
            default: 如果部分不存在，返回的默认值

        返回:
            配置值或默认值
        """
        return self.config.get(section, default)

    def get_nested_config(self, path: str, default: Any = None) -> Any:
        """
        获取嵌套配置

        参数:
            path: 配置路径，使用点表示法，如 "asr.models.vosk"
            default: 如果路径不存在，返回的默认值

        返回:
            配置值或默认值
        """
        parts = path.split('.')
        config = self.config

        for part in parts:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default

        return config

    def get_ui_config(self, key: str, default: Any = None) -> Any:
        """
        获取UI配置

        参数:
            key: UI配置键
            default: 如果键不存在，返回的默认值

        返回:
            UI配置值或默认值
        """
        ui_config = self.get_config("ui", {})
        return ui_config.get(key, default)

    def update_config(self, section: str, values: Dict[str, Any]) -> None:
        """
        更新配置的指定部分

        参数:
            section: 配置部分名称
            values: 要更新的值字典
        """
        if section not in self.config:
            self.config[section] = {}

        self.config[section].update(values)

    def update_and_save(self, section: str, values: Dict[str, Any]) -> None:
        """
        更新配置并保存

        参数:
            section: 配置部分名称
            values: 要更新的值字典
        """
        self.update_config(section, values)
        self.save_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "window": {
                "pos_x": 100,
                "pos_y": 100,
                "width": 800,
                "height": 600,
                "background_mode": "translucent",
                "opacity": 0.7
            },
            "ui": {
                "font_size": 12,
                "font_family": "Microsoft YaHei",
                "theme": "dark"
            },
            "transcription": {
                "default_model": "vosk",
                "language": "zh",
                "sample_rate": 16000
            }
        }

# 创建全局配置管理器实例
# 使用 config.json 作为唯一的配置文件
CONFIG_DIR = "config"
CONFIG_FILE = "config.json"
config_manager = ConfigManager(config_dir=CONFIG_DIR, config_file=CONFIG_FILE)
print(f"[DEBUG] 创建全局配置管理器实例: config_dir={CONFIG_DIR}, config_file={CONFIG_FILE}")


