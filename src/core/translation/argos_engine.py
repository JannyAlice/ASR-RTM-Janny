import os
import time
from typing import Optional, Tuple
import argostranslate
import argostranslate.package
import argostranslate.translate

class ArgosEngine:
    """ArgosTranslate 翻译引擎类"""
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        初始化 ArgosTranslate 翻译引擎
        
        Args:
            model_dir (str, optional): 模型目录路径。如果为 None，则使用默认路径
        """
        # 使用本地模型路径
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join("models", "translation", "argos"))
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 设置模型目录
        argostranslate.package.update_package_index()
        
        # 初始化翻译器
        self.translator = None
        self.setup()
    
    def setup(self) -> bool:
        """
        初始化翻译器
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            # 获取可用的语言包
            available_packages = argostranslate.package.get_available_packages()
            
            # 查找英语到中文的翻译包
            package_to_install = next(
                (pkg for pkg in available_packages 
                 if pkg.from_code == "en" and pkg.to_code == "zh"),
                None
            )
            
            if package_to_install:
                # 下载并安装语言包
                argostranslate.package.install_from_path(package_to_install.download())
                
                # 获取已安装的语言
                installed_languages = argostranslate.translate.get_installed_languages()
                
                # 获取英语和中文语言对象
                from_lang = next(
                    (lang for lang in installed_languages if lang.code == "en"),
                    None
                )
                to_lang = next(
                    (lang for lang in installed_languages if lang.code == "zh"),
                    None
                )
                
                if from_lang and to_lang:
                    # 创建翻译器
                    self.translator = from_lang.get_translation(to_lang)
                    return True
            
            return False
            
        except Exception as e:
            print(f"ArgosTranslate 初始化失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def translate(self, text: str, **kwargs) -> Tuple[Optional[str], float]:
        """
        翻译文本
        
        Args:
            text (str): 要翻译的文本
            **kwargs: 额外的参数（当前未使用）
            
        Returns:
            Tuple[Optional[str], float]: (翻译结果, 延迟时间)
        """
        if not text or not self.translator:
            return None, 0.0
            
        try:
            # 开始计时
            start_time = time.time()
            
            # 执行翻译
            translation = self.translator.translate(text)
            
            # 结束计时
            end_time = time.time()
            
            return translation, end_time - start_time
            
        except Exception as e:
            print(f"ArgosTranslate 翻译错误: {e}")
            return None, 0.0
    
    def get_supported_languages(self) -> list:
        """
        获取支持的语言列表
        
        Returns:
            list: 支持的语言代码列表
        """
        try:
            languages = argostranslate.translate.get_installed_languages()
            return [lang.code for lang in languages]
        except Exception:
            return []
