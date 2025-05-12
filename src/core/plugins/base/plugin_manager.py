"""
插件管理器模块
负责插件的加载、卸载和管理
"""
import os
import sys
import json
import importlib
import traceback
from typing import Dict, Any, Optional, List, Type

from .plugin_base import PluginBase
from .plugin_registry import PluginRegistry
from src.utils.config_manager import config_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PluginManager:
    """插件管理器类，单例模式"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.registry = PluginRegistry()
        self.config_manager = config_manager
        self.plugin_dirs = ["src/core/plugins"]
        self.plugins_config = {}
        self.plugin_metadata = {}

        self._initialized = True

    def configure(self, config: Dict[str, Any] = None) -> None:
        """配置插件管理器

        Args:
            config: 配置字典
        """
        if config is None:
            config = self.config_manager.get_all_config()

        # 获取插件目录
        plugin_dirs = config.get('plugin_dirs', self.plugin_dirs)
        if plugin_dirs:
            self.plugin_dirs = plugin_dirs

        # 获取插件配置
        self.plugins_config = self.config_manager.get_all_plugins()

        # 加载插件元数据
        self._load_plugin_metadata()

        logger.info(f"插件管理器配置完成，插件目录: {self.plugin_dirs}")

    def _load_plugin_metadata(self) -> None:
        """加载插件元数据"""
        try:
            # 遍历插件目录
            for plugin_dir in self.plugin_dirs:
                if not os.path.exists(plugin_dir):
                    logger.warning(f"插件目录不存在: {plugin_dir}")
                    continue

                # 遍历子目录
                for root, dirs, files in os.walk(plugin_dir):
                    # 查找metadata.json文件
                    metadata_file = os.path.join(root, "metadata.json")
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)

                            # 获取插件ID
                            plugin_id = metadata.get('id')
                            if not plugin_id:
                                logger.warning(f"插件元数据缺少ID: {metadata_file}")
                                continue

                            # 添加插件路径
                            metadata['path'] = os.path.dirname(metadata_file)

                            # 保存元数据
                            self.plugin_metadata[plugin_id] = metadata
                            logger.debug(f"已加载插件元数据: {plugin_id}")

                        except Exception as e:
                            logger.error(f"加载插件元数据失败: {metadata_file}, 错误: {str(e)}")
                            logger.error(traceback.format_exc())

            logger.info(f"已加载 {len(self.plugin_metadata)} 个插件元数据")

        except Exception as e:
            logger.error(f"加载插件元数据时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def get_registry(self) -> PluginRegistry:
        """获取插件注册表

        Returns:
            PluginRegistry: 插件注册表实例
        """
        return self.registry

    def register_plugin(self, plugin_id: str, plugin_class: Type[PluginBase]) -> bool:
        """注册插件

        Args:
            plugin_id: 插件ID
            plugin_class: 插件类

        Returns:
            bool: 注册是否成功
        """
        try:
            # 检查插件是否已注册
            if self.registry.is_registered(plugin_id):
                logger.warning(f"插件已注册: {plugin_id}")
                return True

            # 注册插件
            self.registry.register(plugin_id, plugin_class)
            logger.info(f"已注册插件: {plugin_id}")
            return True

        except Exception as e:
            logger.error(f"注册插件失败: {plugin_id}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def load_plugin(self, plugin_id: str) -> bool:
        """加载插件

        Args:
            plugin_id: 插件ID

        Returns:
            bool: 加载是否成功
        """
        try:
            # 检查插件是否已加载
            if self.registry.is_loaded(plugin_id):
                logger.warning(f"插件已加载: {plugin_id}")
                return True

            # 获取插件元数据
            metadata = self.plugin_metadata.get(plugin_id)
            if not metadata:
                logger.error(f"未找到插件元数据: {plugin_id}")
                return False

            # 获取插件路径
            plugin_path = metadata.get('path')
            if not plugin_path:
                logger.error(f"插件元数据缺少路径: {plugin_id}")
                return False

            # 获取插件模块
            module_path = metadata.get('module')
            if not module_path:
                logger.error(f"插件元数据缺少模块路径: {plugin_id}")
                return False

            # 获取插件类名
            class_name = metadata.get('class')
            if not class_name:
                logger.error(f"插件元数据缺少类名: {plugin_id}")
                return False

            # 导入插件模块
            sys.path.insert(0, os.path.dirname(plugin_path))
            module = importlib.import_module(module_path)
            sys.path.pop(0)

            # 获取插件类
            plugin_class = getattr(module, class_name)

            # 注册插件
            if not self.register_plugin(plugin_id, plugin_class):
                logger.error(f"注册插件失败: {plugin_id}")
                return False

            # 加载插件
            if not self.registry.load_plugin(plugin_id):
                logger.error(f"加载插件失败: {plugin_id}")
                return False

            logger.info(f"已加载插件: {plugin_id}")
            return True

        except Exception as e:
            logger.error(f"加载插件失败: {plugin_id}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def unload_plugin(self, plugin_id: str) -> bool:
        """卸载插件

        Args:
            plugin_id: 插件ID

        Returns:
            bool: 卸载是否成功
        """
        try:
            # 卸载插件
            if self.registry.unload_plugin(plugin_id):
                logger.info(f"已卸载插件: {plugin_id}")
                return True
            else:
                logger.warning(f"插件未加载: {plugin_id}")
                return False

        except Exception as e:
            logger.error(f"卸载插件失败: {plugin_id}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def reload_plugin(self, plugin_id: str) -> bool:
        """重新加载插件

        Args:
            plugin_id: 插件ID

        Returns:
            bool: 重新加载是否成功
        """
        try:
            # 卸载插件
            self.unload_plugin(plugin_id)

            # 加载插件
            return self.load_plugin(plugin_id)

        except Exception as e:
            logger.error(f"重新加载插件失败: {plugin_id}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def reload_plugins(self) -> None:
        """重新加载所有插件"""
        try:
            # 重新加载插件元数据
            self._load_plugin_metadata()

            # 获取已加载的插件
            loaded_plugins = self.registry.get_loaded_plugins()

            # 卸载所有插件
            for plugin_id in loaded_plugins:
                self.unload_plugin(plugin_id)

            # 加载启用的插件
            for plugin_id, metadata in self.plugin_metadata.items():
                # 获取插件配置
                plugin_config = self.config_manager.get_plugin_config(plugin_id) or {}

                # 检查插件是否启用
                if plugin_config.get('enabled', False):
                    self.load_plugin(plugin_id)

            logger.info("已重新加载所有插件")

        except Exception as e:
            logger.error(f"重新加载所有插件时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def get_plugin_metadata(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """获取插件元数据

        Args:
            plugin_id: 插件ID

        Returns:
            Optional[Dict[str, Any]]: 插件元数据
        """
        return self.plugin_metadata.get(plugin_id)

    def get_all_plugins(self) -> Dict[str, Dict[str, Any]]:
        """获取所有插件信息

        Returns:
            Dict[str, Dict[str, Any]]: 插件信息字典
        """
        result = {}

        # 遍历插件元数据
        for plugin_id, metadata in self.plugin_metadata.items():
            # 获取插件配置
            plugin_config = self.config_manager.get_plugin_config(plugin_id) or {}

            # 合并元数据和配置
            plugin_info = metadata.copy()
            plugin_info['enabled'] = plugin_config.get('enabled', False)
            plugin_info['loaded'] = self.registry.is_loaded(plugin_id)

            result[plugin_id] = plugin_info

        return result

    def get_available_models(self) -> List[str]:
        """获取可用的ASR模型插件

        Returns:
            List[str]: 可用的ASR模型插件ID列表
        """
        models = []

        # 遍历插件元数据
        for plugin_id, metadata in self.plugin_metadata.items():
            # 检查插件类型
            if metadata.get('type') == 'asr':
                # 获取插件配置
                plugin_config = self.config_manager.get_plugin_config(plugin_id) or {}

                # 检查插件是否启用
                if plugin_config.get('enabled', False):
                    models.append(plugin_id)

        return models

    def cleanup(self) -> None:
        """清理所有插件资源"""
        try:
            # 获取已加载的插件
            loaded_plugins = self.registry.get_loaded_plugins()

            # 卸载所有插件
            for plugin_id in loaded_plugins:
                self.unload_plugin(plugin_id)

            logger.info("已清理所有插件资源")

        except Exception as e:
            logger.error(f"清理插件资源时出错: {str(e)}")
            logger.error(traceback.format_exc())