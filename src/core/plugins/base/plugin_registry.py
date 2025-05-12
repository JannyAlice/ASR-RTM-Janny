"""
插件注册表模块
负责插件的注册和管理
"""
import traceback
from typing import Dict, Type, Optional, List

from .plugin_base import PluginBase
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PluginRegistry:
    """插件注册表类"""

    def __init__(self):
        """初始化插件注册表"""
        self.plugins = {}  # 插件类字典
        self.instances = {}  # 插件实例字典

    def register(self, plugin_id: str, plugin_class: Type[PluginBase]) -> bool:
        """注册插件

        Args:
            plugin_id: 插件ID
            plugin_class: 插件类

        Returns:
            bool: 注册是否成功
        """
        try:
            # 检查插件类是否继承自PluginBase
            if not issubclass(plugin_class, PluginBase):
                logger.error(f"插件类必须继承自PluginBase: {plugin_id}")
                return False

            # 注册插件
            self.plugins[plugin_id] = plugin_class
            logger.info(f"已注册插件: {plugin_id}")
            return True

        except Exception as e:
            logger.error(f"注册插件失败: {plugin_id}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def unregister(self, plugin_id: str) -> bool:
        """注销插件

        Args:
            plugin_id: 插件ID

        Returns:
            bool: 注销是否成功
        """
        try:
            # 检查插件是否已加载
            if plugin_id in self.instances:
                logger.warning(f"插件已加载，无法注销: {plugin_id}")
                return False

            # 检查插件是否已注册
            if plugin_id not in self.plugins:
                logger.warning(f"插件未注册: {plugin_id}")
                return False

            # 注销插件
            del self.plugins[plugin_id]
            logger.info(f"已注销插件: {plugin_id}")
            return True

        except Exception as e:
            logger.error(f"注销插件失败: {plugin_id}, 错误: {str(e)}")
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
            if plugin_id in self.instances:
                logger.warning(f"插件已加载: {plugin_id}")
                return True

            # 检查插件是否已注册
            if plugin_id not in self.plugins:
                logger.error(f"插件未注册: {plugin_id}")
                return False

            # 创建插件实例
            plugin_class = self.plugins[plugin_id]
            # 检查插件类的__init__方法是否接受参数
            if hasattr(plugin_class, '__init__') and plugin_class.__init__.__code__.co_argcount > 1:
                # 如果插件类的__init__方法接受参数，则传入空字典
                plugin = plugin_class({})
            else:
                # 否则不传入参数
                plugin = plugin_class()

            # 初始化插件
            if not plugin.initialize():
                logger.error(f"初始化插件失败: {plugin_id}")
                return False

            # 保存插件实例
            self.instances[plugin_id] = plugin
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
            # 检查插件是否已加载
            if plugin_id not in self.instances:
                logger.warning(f"插件未加载: {plugin_id}")
                return False

            # 获取插件实例
            plugin = self.instances[plugin_id]

            # 清理插件
            if not plugin.cleanup():
                logger.warning(f"清理插件失败: {plugin_id}")

            # 删除插件实例
            del self.instances[plugin_id]
            logger.info(f"已卸载插件: {plugin_id}")
            return True

        except Exception as e:
            logger.error(f"卸载插件失败: {plugin_id}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """获取插件实例

        Args:
            plugin_id: 插件ID

        Returns:
            Optional[PluginBase]: 插件实例
        """
        # 检查插件是否已加载
        if plugin_id not in self.instances:
            # 尝试加载插件
            if not self.load_plugin(plugin_id):
                logger.error(f"获取插件失败: {plugin_id}")
                return None

        return self.instances.get(plugin_id)

    def is_registered(self, plugin_id: str) -> bool:
        """检查插件是否已注册

        Args:
            plugin_id: 插件ID

        Returns:
            bool: 是否已注册
        """
        return plugin_id in self.plugins

    def is_loaded(self, plugin_id: str) -> bool:
        """检查插件是否已加载

        Args:
            plugin_id: 插件ID

        Returns:
            bool: 是否已加载
        """
        return plugin_id in self.instances

    def get_registered_plugins(self) -> List[str]:
        """获取已注册的插件ID列表

        Returns:
            List[str]: 已注册的插件ID列表
        """
        return list(self.plugins.keys())

    def get_loaded_plugins(self) -> List[str]:
        """获取已加载的插件ID列表

        Returns:
            List[str]: 已加载的插件ID列表
        """
        return list(self.instances.keys())

    def get_plugins_by_type(self, plugin_type: str) -> Dict[str, Type[PluginBase]]:
        """获取指定类型的所有插件

        Args:
            plugin_type: 插件类型

        Returns:
            Dict[str, Type[PluginBase]]: 插件字典
        """
        return {
            plugin_id: plugin_class
            for plugin_id, plugin_class in self.plugins.items()
            if getattr(plugin_class, 'plugin_type', None) == plugin_type
        }