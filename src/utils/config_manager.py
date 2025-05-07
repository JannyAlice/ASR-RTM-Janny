#!/usr/bin/env python3
"""
配置管理模块
负责加载和管理配置信息
"""
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理类"""
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._config = self.load_config()
            # 验证配置
            if not self.validate_config(self._config):
                raise ValueError("配置验证失败")

    @property
    def config(self) -> Dict[str, Any]:
        """获取配置"""
        return self._config

    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_path = os.path.join('config', 'config.json')
            logger.debug(f"尝试加载配置文件: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            logger.info("配置文件加载成功")

            # 验证配置
            if not self.validate_config(config):
                raise ValueError("配置验证失败")

            return config

        except FileNotFoundError:
            logger.error(f"配置文件不存在: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"加载配置文件时发生错误: {str(e)}")
            raise

    def save_config(self) -> bool:
        """保存配置到文件"""
        try:
            config_path = os.path.join('config', 'config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            logger.info("配置保存成功")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            return False

    def get_config(self, key: str, default=None) -> Any:
        """
        获取配置

        Args:
            key: 配置键，可以是点分隔的路径，如 'app.window.geometry'
            default: 默认值，如果配置不存在则返回此值

        Returns:
            Any: 配置值或默认值
        """
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_ui_config(self, key: str, default=None) -> Any:
        """
        获取UI配置

        Args:
            key: 配置键，可以是点分隔的路径，如 'fonts.subtitle'
            default: 默认值，如果配置不存在则返回此值

        Returns:
            Any: 配置值或默认值
        """
        # 首先尝试从ui_config.json加载
        try:
            ui_config_path = os.path.join('config', 'ui_config.json')
            if os.path.exists(ui_config_path):
                with open(ui_config_path, 'r', encoding='utf-8') as f:
                    ui_config = json.load(f)

                # 解析键路径
                keys = key.split('.')
                value = ui_config
                for k in keys:
                    value = value[k]
                return value
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            # 如果从ui_config.json加载失败，尝试从主配置中的ui部分加载
            pass

        # 尝试从主配置的ui部分加载
        try:
            # 构建完整的键路径
            full_key = f"ui.{key}"
            return self.get_config(full_key, default)
        except (KeyError, TypeError):
            # 如果从主配置加载失败，返回默认值
            return default

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取指定模型的配置"""
        try:
            return self._config['asr']['models'].get(model_name)
        except KeyError:
            logger.error(f"未找到模型配置: {model_name}")
            return None

    def get_window_config(self) -> Dict[str, Any]:
        """获取窗口配置"""
        return self._config.get('window', {})

    def update_window_config(self, config: Dict[str, Any]) -> None:
        """更新窗口配置"""
        self._config['window'] = config
        self.save_config()

    def update_and_save(self, section: str, config: Dict[str, Any]) -> bool:
        """
        更新指定部分的配置并保存

        Args:
            section: 配置部分名称
            config: 新的配置值

        Returns:
            bool: 是否更新成功
        """
        try:
            # 更新配置
            self._config[section] = config

            # 保存配置
            return self.save_config()
        except Exception as e:
            logger.error(f"更新并保存配置失败: {str(e)}")
            return False

    def get_default_model(self) -> str:
        """获取默认模型名称"""
        return self._config.get('transcription', {}).get('default_model', 'vosk_small')

    def validate_model_files(self, model_path: str) -> bool:
        """验证模型文件完整性"""
        required_files = [
            "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
            "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
            "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
            "tokens.txt"
        ]

        try:
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    logger.error(f"缺少模型文件: {file_path}")
                    return False

            logger.info("模型文件验证通过")
            return True

        except Exception as e:
            logger.error(f"验证模型文件时发生错误: {str(e)}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置文件的完整性和正确性
        Returns:
            bool: 验证是否通过
        """
        required_keys = [
            'app',
            'transcription',
            'asr.models',
            'window'
        ]

        try:
            for key in required_keys:
                parts = key.split('.')
                temp = config
                for part in parts:
                    temp = temp[part]
            return True
        except Exception as e:
            logger.error(f"配置验证失败: {str(e)}")
            return False

# 创建全局配置管理器实例
config_manager = ConfigManager()

def load_config() -> Dict[str, Any]:
    """
    加载配置文件的便捷函数
    Returns:
        Dict[str, Any]: 配置信息字典
    """
    try:
        config_path = 'config/config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 验证配置
        if not config_manager.validate_config(config):
            raise ValueError("配置验证失败")

        return config

    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        raise

# 导出
__all__ = ['ConfigManager', 'config_manager', 'load_config']


