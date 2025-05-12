"""
模型管理基类模块
为所有类型的模型管理器提供基础功能
"""
from typing import Dict, Any, Optional, List
import os
from src.utils.config_manager import config_manager

class ModelManagerBase:
    """模型管理器基类"""

    def __init__(self, model_type: str):
        """
        初始化模型管理器基类
        
        Args:
            model_type: 模型类型，如 'asr', 'translation' 等
        """
        self.model_type = model_type
        self.config = config_manager
        self.models_config = {}
        self.current_model = None
        self.model_path = None
        self.model_name = None
        
        # 从配置中加载模型配置
        self._load_models_config()
        
    def _load_models_config(self) -> None:
        """从配置中加载模型配置"""
        # 尝试从配置中获取模型配置
        if self.model_type in self.config.config and 'models' in self.config.config[self.model_type]:
            self.models_config = self.config.config[self.model_type]['models']
        else:
            # 尝试从顶级 'models' 配置中获取
            all_models = self.config.get_config('models', {})
            # 过滤出属于当前模型类型的模型
            self.models_config = {
                name: config for name, config in all_models.items()
                if config.get('type', '').startswith(self.model_type)
            }
        
        print(f"[DEBUG] {self.__class__.__name__}._load_models_config: models_config = {self.models_config}")
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定模型的配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 模型配置，如果不存在则返回 None
        """
        return self.models_config.get(model_name)
    
    def get_available_models(self) -> Dict[str, bool]:
        """
        获取可用的模型列表
        
        Returns:
            Dict[str, bool]: 模型名称到是否启用的映射
        """
        return {
            name: bool(config and config.get('enabled', False))
            for name, config in self.models_config.items()
        }
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        获取指定模型的路径
        
        Args:
            model_name: 模型名称
            
        Returns:
            str: 模型路径，如果不存在则返回 None
        """
        model_config = self.get_model_config(model_name)
        if not model_config:
            return None
        
        path = model_config.get('path', '')
        if not path:
            return None
        
        # 检查路径是否存在
        if not os.path.exists(path):
            print(f"模型路径不存在: {path}")
            return None
        
        return path
    
    def is_model_enabled(self, model_name: str) -> bool:
        """
        检查指定模型是否启用
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 模型是否启用
        """
        model_config = self.get_model_config(model_name)
        return bool(model_config and model_config.get('enabled', False))
    
    def get_model_names(self) -> List[str]:
        """
        获取所有模型的名称
        
        Returns:
            List[str]: 模型名称列表
        """
        return list(self.models_config.keys())
    
    def get_enabled_models(self) -> List[str]:
        """
        获取所有启用的模型的名称
        
        Returns:
            List[str]: 启用的模型名称列表
        """
        return [
            name for name, config in self.models_config.items()
            if config.get('enabled', False)
        ]
    
    def get_default_model(self) -> Optional[str]:
        """
        获取默认模型名称
        
        Returns:
            str: 默认模型名称，如果没有则返回 None
        """
        # 尝试从配置中获取默认模型
        default_model = self.config.get_nested_config(f"{self.model_type}.default_model")
        if default_model and default_model in self.models_config:
            return default_model
        
        # 如果没有配置默认模型，则返回第一个启用的模型
        enabled_models = self.get_enabled_models()
        if enabled_models:
            return enabled_models[0]
        
        return None
