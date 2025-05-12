import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 创建logs目录
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

# 生成带时间戳的日志文件名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f'test_model_manager_{timestamp}.log'

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from src.core.asr.model_manager import ASRModelManager
from src.utils.config_manager import config_manager

def test_model_manager():
    """测试模型管理器的加载和创建识别器功能"""
    try:
        logger.info("-" * 50)
        logger.info("开始测试模型管理器")
        logger.info(f"日志文件: {log_file}")
        logger.info(f"项目根目录: {project_root}")
        
        # 创建模型管理器实例
        logger.info("\n1. 创建 ASRModelManager 实例")
        model_manager = ASRModelManager()
        
        # 检查配置
        logger.info("\n2. 检查配置信息")
        config = config_manager.config
        logger.info(f"完整配置: {config}")

        # 获取默认模型
        logger.info("\n3. 获取默认模型")
        default_model = config_manager.get_default_model()
        logger.info(f"默认模型: {default_model}")

        # 获取模型配置
        logger.info("\n4. 获取模型配置")
        model_config = model_manager.models_config.get(default_model, {})
        logger.info(f"模型配置: {model_config}")

        # 检查模型文件
        logger.info("\n5. 检查模型文件")
        model_path = model_config.get('path', '')
        logger.info(f"模型路径: {model_path}")
        if os.path.exists(model_path):
            files = os.listdir(model_path)
            logger.info(f"目录内容: {files}")
        else:
            logger.error(f"模型路径不存在: {model_path}")

        # 加载模型
        logger.info("\n6. 尝试加载模型")
        if not model_manager.load_model(default_model):
            logger.error(f"加载模型失败: {default_model}")
            return False

        # 检查模型加载状态
        logger.info("\n7. 检查模型加载状态")
        logger.info(f"当前模型类型: {model_manager.model_type}")
        logger.info(f"当前模型路径: {model_manager.model_path}")
        logger.info(f"当前模型实例: {model_manager.current_model}")

        # 创建识别器
        logger.info("\n8. 尝试创建识别器")
        recognizer = model_manager.create_recognizer()
        if not recognizer:
            logger.error("创建识别器失败")
            return False

        logger.info("识别器创建成功")
        logger.info(f"识别器类型: {type(recognizer)}")
        logger.info("-" * 50)
        return True

    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_model_manager()
    if success:
        print("测试成功")
        sys.exit(0)
    else:
        print("测试失败")
        sys.exit(1)