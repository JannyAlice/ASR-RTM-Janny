"""
插件系统基础类
定义插件的基本接口和功能
"""
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

class PluginBase(ABC):
    """插件基础类"""

    def __init__(self):
        """初始化插件"""
        self._config: Dict[str, Any] = {}
        self._enabled: bool = False
        self._initialized: bool = False

    def initialize(self) -> bool:
        """初始化插件

        Returns:
            bool: 初始化是否成功
        """
        try:
            if self._initialized:
                logger.warning(f"插件已初始化: {self.get_id()}")
                return True

            # 调用子类的setup方法
            if not self.setup():
                logger.error(f"插件初始化失败: {self.get_id()}")
                return False

            self._initialized = True
            logger.info(f"插件初始化成功: {self.get_id()}")
            return True

        except Exception as e:
            logger.error(f"插件初始化时出错: {self.get_id()}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def cleanup(self) -> bool:
        """清理插件资源

        Returns:
            bool: 清理是否成功
        """
        try:
            if not self._initialized:
                logger.warning(f"插件未初始化: {self.get_id()}")
                return True

            # 调用子类的teardown方法
            if not self.teardown():
                logger.error(f"插件清理失败: {self.get_id()}")
                return False

            self._initialized = False
            logger.info(f"插件清理成功: {self.get_id()}")
            return True

        except Exception as e:
            logger.error(f"插件清理时出错: {self.get_id()}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def configure(self, config: Dict[str, Any]) -> bool:
        """配置插件

        Args:
            config: 配置字典

        Returns:
            bool: 配置是否成功
        """
        try:
            # 保存配置
            self._config = config

            # 如果已初始化，重新配置
            if self._initialized:
                if not self.reconfigure():
                    logger.error(f"插件重新配置失败: {self.get_id()}")
                    return False

            logger.info(f"插件配置成功: {self.get_id()}")
            return True

        except Exception as e:
            logger.error(f"插件配置时出错: {self.get_id()}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def enable(self) -> bool:
        """启用插件

        Returns:
            bool: 启用是否成功
        """
        try:
            if self._enabled:
                logger.warning(f"插件已启用: {self.get_id()}")
                return True

            # 如果未初始化，先初始化
            if not self._initialized and not self.initialize():
                logger.error(f"插件启用失败，初始化失败: {self.get_id()}")
                return False

            self._enabled = True
            logger.info(f"插件启用成功: {self.get_id()}")
            return True

        except Exception as e:
            logger.error(f"插件启用时出错: {self.get_id()}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def disable(self) -> bool:
        """禁用插件

        Returns:
            bool: 禁用是否成功
        """
        try:
            if not self._enabled:
                logger.warning(f"插件已禁用: {self.get_id()}")
                return True

            self._enabled = False
            logger.info(f"插件禁用成功: {self.get_id()}")
            return True

        except Exception as e:
            logger.error(f"插件禁用时出错: {self.get_id()}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def is_enabled(self) -> bool:
        """检查插件是否启用

        Returns:
            bool: 是否启用
        """
        return self._enabled

    def is_initialized(self) -> bool:
        """检查插件是否已初始化

        Returns:
            bool: 是否已初始化
        """
        return self._initialized

    def get_config(self) -> Dict[str, Any]:
        """获取插件配置

        Returns:
            Dict[str, Any]: 插件配置
        """
        return self._config

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """获取插件配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            Any: 配置值
        """
        return self._config.get(key, default)

    def get_info(self) -> Dict[str, Any]:
        """获取插件信息

        Returns:
            Dict[str, Any]: 插件信息
        """
        return {
            "id": self.get_id(),
            "name": self.get_name(),
            "version": self.get_version(),
            "description": self.get_description(),
            "author": self.get_author(),
            "enabled": self.is_enabled(),
            "initialized": self.is_initialized()
        }

    @abstractmethod
    def get_id(self) -> str:
        """获取插件ID

        Returns:
            str: 插件ID
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """获取插件名称

        Returns:
            str: 插件名称
        """
        pass

    @abstractmethod
    def get_version(self) -> str:
        """获取插件版本

        Returns:
            str: 插件版本
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """获取插件描述

        Returns:
            str: 插件描述
        """
        pass

    @abstractmethod
    def get_author(self) -> str:
        """获取插件作者

        Returns:
            str: 插件作者
        """
        pass

    @abstractmethod
    def setup(self) -> bool:
        """设置插件

        Returns:
            bool: 设置是否成功
        """
        pass

    @abstractmethod
    def teardown(self) -> bool:
        """清理插件

        Returns:
            bool: 清理是否成功
        """
        pass

    def reconfigure(self) -> bool:
        """重新配置插件

        Returns:
            bool: 重新配置是否成功
        """
        # 默认实现：清理后重新初始化
        if not self.cleanup():
            return False

        return self.initialize()

# 为了向后兼容，保留PluginInterface类
class PluginInterface(PluginBase):
    """插件接口类（向后兼容）"""

    def get_id(self) -> str:
        return "plugin_interface"

    def get_name(self) -> str:
        return "Plugin Interface"

    def get_version(self) -> str:
        return "1.0.0"

    def get_description(self) -> str:
        return "Plugin interface for backward compatibility"

    def get_author(self) -> str:
        return "System"

    def setup(self) -> bool:
        return True

    def teardown(self) -> bool:
        return True