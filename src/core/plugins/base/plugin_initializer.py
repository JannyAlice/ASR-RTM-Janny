from typing import Dict, Any, List, Optional
import json
import os
import logging
from .plugin_registry import PluginRegistry
from ..asr.vosk_plugin import VoskPlugin
from ..asr.sherpa_plugin import SherpaPlugin

logger = logging.getLogger(__name__)

class PluginInitializer:
    """插件初始化器，负责注册和加载系统所需的插件"""
    
    def __init__(self, plugins_config_path: str = None):
        self.registry = PluginRegistry()
        self.plugins_config_path = plugins_config_path or "config/plugins.json"
        self.core_config_path = "config/config.json"
        
        # 设置插件系统日志
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """设置插件系统日志"""
        try:
            plugins_config = self._load_config(self.plugins_config_path)
            log_config = plugins_config.get("plugin_system", {}).get("logging", {})
            
            if log_config.get("enabled", True):
                log_file = log_config.get("file", "logs/plugins.log")
                log_level = log_config.get("level", "INFO")
                
                # 确保日志目录存在
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
                # 配置日志处理器
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
                
                # 设置日志级别
                logger.setLevel(getattr(logging, log_level))
                logger.addHandler(file_handler)
                
                logger.info("Plugin system logging initialized")
                
        except Exception as e:
            print(f"Warning: Failed to setup plugin system logging: {e}")
        
    def register_builtin_plugins(self) -> None:
        """注册内置插件"""
        # 注册 ASR 插件
        self.registry.register_plugin("vosk_small", VoskPlugin)
        self.registry.register_plugin("sherpa_0626_std", SherpaPlugin)
        
        logger.info("Registered built-in plugins: vosk_small, sherpa_0626_std")
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if not os.path.exists(path):
                logger.warning(f"Config file not found: {path}")
                return {}
                
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            return {}
            
    def _get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """从core config中获取模型配置"""
        core_config = self._load_config(self.core_config_path)
        models = core_config.get("asr", {}).get("models", {})
        return models.get(model_id)
        
    def _merge_configs(self, 
                      plugin_config: Dict[str, Any], 
                      model_config: Dict[str, Any]
                      ) -> Dict[str, Any]:
        """合并插件配置和模型配置
        
        Args:
            plugin_config: plugins.json中的插件配置
            model_config: config.json中的模型配置
            
        Returns:
            Dict[str, Any]: 合并后的配置
        """
        merged = {
            "enabled": plugin_config.get("enabled", True),
            "model_path": model_config.get("path"),
            "type": model_config.get("type", "standard"),
            "config": {
                # 基础模型配置
                **model_config.get("config", {}),
                # 插件特定配置（会覆盖同名的模型配置）
                **plugin_config.get("plugin_config", {})
            }
        }
        return merged
            
    def initialize_plugins(self) -> List[str]:
        """初始化所有已配置的插件"""
        # 注册内置插件
        self.register_builtin_plugins()
        
        # 加载插件配置
        plugins_config = self._load_config(self.plugins_config_path)
        
        # 记录成功加载的插件
        loaded_plugins = []
        
        # 检查是否启用了自动重载
        auto_reload = plugins_config.get("plugin_system", {}).get("auto_reload", True)
        if auto_reload:
            logger.info("Plugin auto-reload is enabled")
        
        # 加载ASR插件
        asr_plugins = plugins_config.get("plugins", {}).get("asr", {})
        for plugin_id, plugin_config in asr_plugins.items():
            if plugin_config.get("enabled", True):
                try:
                    # 获取对应的模型配置
                    model_id = plugin_config.get("model_config")
                    model_config = self._get_model_config(model_id)
                    
                    if not model_config:
                        logger.error(f"Model config not found for plugin {plugin_id} (model_id: {model_id})")
                        continue
                        
                    # 合并配置
                    merged_config = self._merge_configs(plugin_config, model_config)
                    
                    # 加载插件
                    if self.registry.load_plugin(plugin_id, merged_config):
                        loaded_plugins.append(plugin_id)
                        logger.info(f"Successfully loaded plugin: {plugin_id}")
                    else:
                        logger.error(f"Failed to load plugin: {plugin_id}")
                        
                except Exception as e:
                    logger.error(f"Error loading plugin {plugin_id}: {e}")
                    continue
                    
        logger.info(f"Initialized {len(loaded_plugins)} plugins: {', '.join(loaded_plugins)}")
        return loaded_plugins
        
    def shutdown(self) -> None:
        """关闭插件系统"""
        self.registry.shutdown()
        logger.info("Plugin system shutdown complete")