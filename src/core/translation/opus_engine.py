import os
import time
import torch
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSeq2SeqLM

class OpusMTEngine:
    """OPUS-MT 翻译引擎类"""
    # 类级别的缓存变量
    _tokenizer = None
    _pytorch_model = None
    _onnx_model = None
    
    def __init__(self, model_dir=None):
        """
        初始化 OPUS-MT 翻译引擎
        
        Args:
            model_dir (str, optional): 模型目录路径。如果为 None，则使用默认路径
        """
        # 使用本地模型路径
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join("models", "translation", "opus-mt", "en-zh"))
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.tokenizer = None
        self.pytorch_model = None
        self.onnx_model = None
        self.setup()
    
    def setup(self):
        """初始化模型"""
        try:
            # 使用缓存的分词器和 PyTorch 模型
            if OpusMTEngine._tokenizer is None:
                OpusMTEngine._tokenizer = MarianTokenizer.from_pretrained(self.model_dir)
            self.tokenizer = OpusMTEngine._tokenizer
            
            if OpusMTEngine._pytorch_model is None:
                OpusMTEngine._pytorch_model = MarianMTModel.from_pretrained(self.model_dir)
            self.pytorch_model = OpusMTEngine._pytorch_model
            
            # 使用缓存的 ONNX 模型
            try:
                if OpusMTEngine._onnx_model is None:
                    OpusMTEngine._onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                        self.model_dir,
                        use_io_binding=False
                    )
                self.onnx_model = OpusMTEngine._onnx_model
                
            except Exception as e:
                print(f"ONNX 模型准备失败: {e}")
                raise
            
            return True
            
        except Exception as e:
            print(f"初始化失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def convert_to_onnx(self):
        """将模型转换为 ONNX 格式"""
        try:
            print("\n开始 ONNX 转换...")
            print(f"目标路径: {self.model_dir}")
            
            # 使用最基本的转换配置
            self.onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                self.model_dir,
                export=True,
                use_io_binding=False
            )
            
            # 保存 ONNX 模型
            print("\n保存 ONNX 模型...")
            self.onnx_model.save_pretrained(self.model_dir)
            
            return True
            
        except Exception as e:
            print(f"ONNX 转换失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def translate(self, text, use_onnx=True):
        """
        翻译文本
        
        Args:
            text (str): 要翻译的文本
            use_onnx (bool): 是否使用 ONNX 模型进行翻译
            
        Returns:
            tuple: (翻译结果, 延迟时间)
        """
        if use_onnx and self.onnx_model is not None:
            return self._translate_onnx(text)
        else:
            return self._translate_pytorch(text)
    
    def _translate_pytorch(self, text):
        """使用 PyTorch 模型翻译"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            
            start_time = time.time()
            outputs = self.pytorch_model.generate(**inputs)
            end_time = time.time()
            
            translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return translation, end_time - start_time
            
        except Exception as e:
            print(f"PyTorch 翻译错误: {e}")
            return None, 0
    
    def _translate_onnx(self, text):
        """使用 ONNX 模型翻译"""
        try:
            # 准备输入
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            
            # 计算合适的最大长度（源文本长度的 2.5 倍，最小 256）
            src_len = inputs["input_ids"].shape[1]
            max_length = max(256, int(src_len * 2.5))
            
            # 开始计时
            start_time = time.time()
            
            # 使用 optimum 的生成方法（使用 PyTorch tensor）
            outputs = self.onnx_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,  # 使用动态最大长度
                num_beams=4,
                early_stopping=True,
                length_penalty=0.6  # 添加长度惩罚以避免过短输出
            )
            
            # 结束计时
            end_time = time.time()
            
            # 解码翻译结果
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translation, end_time - start_time
            
        except Exception as e:
            print(f"ONNX 翻译错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None, 0
