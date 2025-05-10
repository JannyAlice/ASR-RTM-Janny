#!/usr/bin/env python3
"""
新插件系统测试脚本
"""
import os
import sys
import json
import traceback
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入日志模块
from src.utils.logger import get_logger
logger = get_logger("new_plugin_system_test")

# 导入插件管理器
from src.core.plugins import PluginManager

# 确保不会导入错误的vosk模块
sys.modules.pop('vosk', None)

def test_plugin_manager():
    """测试插件管理器"""
    try:
        # 创建插件管理器
        plugin_manager = PluginManager()

        # 配置插件管理器
        plugin_manager.configure()

        # 获取所有插件
        plugins = plugin_manager.get_all_plugins()

        # 打印插件信息
        print("\n===== 插件信息 =====")
        for plugin_id, plugin_info in plugins.items():
            print(f"插件ID: {plugin_id}")
            print(f"  名称: {plugin_info.get('name', '')}")
            print(f"  版本: {plugin_info.get('version', '')}")
            print(f"  类型: {plugin_info.get('type', '')}")
            print(f"  描述: {plugin_info.get('description', '')}")
            print(f"  启用: {plugin_info.get('enabled', False)}")
            print(f"  加载: {plugin_info.get('loaded', False)}")
            print()

        # 获取可用的ASR模型
        asr_models = plugin_manager.get_available_models()

        # 打印ASR模型信息
        print("\n===== 可用ASR模型 =====")
        for model_id in asr_models:
            print(f"模型ID: {model_id}")

        # 加载Vosk插件
        print("\n===== 加载Vosk插件 =====")
        if plugin_manager.load_plugin("vosk_small"):
            print("Vosk插件加载成功")

            # 获取插件实例
            plugin = plugin_manager.registry.get_plugin("vosk_small")
            if plugin:
                print("获取Vosk插件实例成功")

                # 获取插件信息
                plugin_info = plugin.get_info()
                print(f"插件信息: {json.dumps(plugin_info, ensure_ascii=False, indent=2)}")

                # 禁用插件
                if plugin.disable():
                    print("Vosk插件禁用成功")

                # 启用插件
                if plugin.enable():
                    print("Vosk插件启用成功")

                # 卸载插件
                if plugin_manager.unload_plugin("vosk_small"):
                    print("Vosk插件卸载成功")
            else:
                print("获取Vosk插件实例失败")
        else:
            print("Vosk插件加载失败")

        # 重新加载所有插件
        print("\n===== 重新加载所有插件 =====")
        plugin_manager.reload_plugins()

        # 清理资源
        print("\n===== 清理资源 =====")
        plugin_manager.cleanup()

        print("\n测试完成")
        return True

    except Exception as e:
        logger.error(f"测试插件管理器失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    # 测试插件管理器
    if test_plugin_manager():
        print("插件系统测试成功")
    else:
        print("插件系统测试失败")

if __name__ == "__main__":
    main()
