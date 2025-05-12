#!/usr/bin/env python3
"""
测试Sherpa-ONNX 2023-06-26插件
"""
import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path

# 确保能够导入src目录下的模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 初始化日志系统
from src.utils.logger import configure_logging, get_logger
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
configure_logging(
    log_dir=str(log_dir),
    default_level="DEBUG"
)
logger = get_logger("test_sherpa_0626_plugin")

# 导入插件管理器和配置管理器
from src.utils.config_manager import config_manager
from src.core.plugins import PluginManager
from src.core.plugins.asr.sherpa_0626_plugin import Sherpa0626Plugin

def test_plugin_initialization():
    """测试插件初始化"""
    print("=" * 80)
    print("测试Sherpa-ONNX 2023-06-26插件初始化")
    print("=" * 80)
    
    # 创建插件实例
    plugin = Sherpa0626Plugin()
    
    # 打印插件信息
    print(f"插件ID: {plugin.get_id()}")
    print(f"插件名称: {plugin.get_name()}")
    print(f"插件版本: {plugin.get_version()}")
    print(f"插件描述: {plugin.get_description()}")
    print(f"插件作者: {plugin.get_author()}")
    
    # 加载配置
    config = config_manager.load_config()
    
    # 获取模型配置
    model_config = config.get('asr', {}).get('models', {}).get('sherpa_0626_std', {})
    
    # 设置插件配置
    plugin.configure({
        'path': model_config.get('path'),
        'type': model_config.get('type', 'standard'),
        'config': model_config.get('config', {})
    })
    
    # 初始化插件
    success = plugin.initialize()
    print(f"插件初始化结果: {'成功' if success else '失败'}")
    
    if success:
        # 清理插件资源
        plugin.teardown()
        print("插件资源已清理")
    
    return success

def test_file_transcription():
    """测试文件转录功能"""
    print("=" * 80)
    print("测试Sherpa-ONNX 2023-06-26插件文件转录功能")
    print("=" * 80)
    
    # 创建插件实例
    plugin = Sherpa0626Plugin()
    
    # 加载配置
    config = config_manager.load_config()
    
    # 获取模型配置
    model_config = config.get('asr', {}).get('models', {}).get('sherpa_0626_std', {})
    
    # 设置插件配置
    plugin.configure({
        'path': model_config.get('path'),
        'type': model_config.get('type', 'standard'),
        'config': model_config.get('config', {})
    })
    
    # 初始化插件
    if not plugin.initialize():
        print("插件初始化失败")
        return False
    
    # 测试文件路径
    test_file = config.get('app', {}).get('default_file', '')
    if not test_file or not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return False
    
    print(f"开始转录文件: {test_file}")
    start_time = time.time()
    
    # 转录文件
    result = plugin.transcribe_file(test_file)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if "error" in result:
        print(f"转录失败: {result['error']}")
        return False
    
    print(f"转录完成，耗时: {elapsed_time:.2f}秒")
    print(f"转录结果: {result['text']}")
    
    # 清理插件资源
    plugin.teardown()
    print("插件资源已清理")
    
    return True

def main():
    """主函数"""
    try:
        # 测试插件初始化
        if not test_plugin_initialization():
            print("插件初始化测试失败")
            return 1
        
        print("\n")
        
        # 测试文件转录
        if not test_file_transcription():
            print("文件转录测试失败")
            return 1
        
        print("\n所有测试通过！")
        return 0
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
