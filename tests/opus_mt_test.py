import os
import time
import torch
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSeq2SeqLM

class OpusMTTester:
    """OPUS-MT 翻译测试类"""
    # 类级别的缓存变量
    _tokenizer = None
    _pytorch_model = None
    _onnx_model = None
    
    def __init__(self):
        # 使用本地模型路径
        self.model_dir = os.path.abspath(os.path.join("models", "opus-mt", "en-zh"))
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 原始的文件检查代码保留但注释掉
        """
        self.check_model_files()
        """
        
        self.tokenizer = None
        self.pytorch_model = None
        self.onnx_model = None
        self.setup()
    
    def setup(self):
        """初始化模型"""
        try:
            # 使用缓存的分词器和 PyTorch 模型
            if OpusMTTester._tokenizer is None:
                OpusMTTester._tokenizer = MarianTokenizer.from_pretrained(self.model_dir)
            self.tokenizer = OpusMTTester._tokenizer
            
            if OpusMTTester._pytorch_model is None:
                OpusMTTester._pytorch_model = MarianMTModel.from_pretrained(self.model_dir)
            self.pytorch_model = OpusMTTester._pytorch_model
            
            # 使用缓存的 ONNX 模型
            try:
                if OpusMTTester._onnx_model is None:
                    OpusMTTester._onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                        self.model_dir,
                        use_io_binding=False
                    )
                self.onnx_model = OpusMTTester._onnx_model
                
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
        """仅在首次运行时转换为 ONNX 格式"""
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
            
            # 检查模型对象和保存的文件
            print("\nONNX 模型信息:")
            print(f"模型类型: {type(self.onnx_model)}")
            print(f"模型配置: {self.onnx_model.config}")
            
            # 检查可能的 ONNX 文件位置
            possible_paths = [
                self.model_dir,
                os.path.join(self.model_dir, "onnx"),
                os.path.join(os.getcwd(), "onnx"),
                os.path.expanduser("~/.cache/huggingface/hub")
            ]
            
            print("\n检查可能的 ONNX 文件位置:")
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"\n目录: {path}")
                    for file in os.listdir(path):
                        if file.endswith('.onnx'):
                            file_path = os.path.join(path, file)
                            size = os.path.getsize(file_path) / 1024 / 1024
                            print(f"- 找到 ONNX 文件: {file} ({size:.2f} MB)")
            
            # 简单检查是否成功
            if self.onnx_model is not None:
                print("\nONNX 模型转换成功！")
                return True
            else:
                raise Exception("转换后的模型为空")
            
        except Exception as e:
            print(f"ONNX 转换失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def test_translation(self, text, runs=5):
        """测试翻译性能"""
        print("\n=== 开始翻译测试 ===")
        print(f"测试文本: {text}")
        print(f"运行次数: {runs}")
        
        # PyTorch 测试
        pytorch_times = []
        pytorch_results = []
        for i in range(runs):
            result, latency = self.translate_pytorch(text)
            pytorch_times.append(latency)
            pytorch_results.append(result)
        
        # ONNX 测试
        onnx_times = []
        onnx_results = []
        for i in range(runs):
            result, latency = self.translate_onnx(text)
            onnx_times.append(latency)
            onnx_results.append(result)
        
        # 计算统计数据
        avg_pytorch = np.mean(pytorch_times)
        avg_onnx = np.mean(onnx_times)
        speedup = avg_pytorch / avg_onnx if avg_onnx > 0 else float('inf')
        
        # 打印结果
        print("\n=== 测试结果 ===")
        print(f"PyTorch 平均延迟: {avg_pytorch:.4f} 秒")
        print(f"ONNX 平均延迟: {avg_onnx:.4f} 秒")
        print(f"加速比: {speedup:.2f}x")
        print(f"\nPyTorch 翻译结果: {pytorch_results[0]}")
        print(f"ONNX 翻译结果: {onnx_results[0]}")
        
        return {
            'pytorch_latency': avg_pytorch,
            'onnx_latency': avg_onnx,
            'speedup': speedup,
            'pytorch_result': pytorch_results[0],
            'onnx_result': onnx_results[0]
        }
    
    def translate_pytorch(self, text):
        """PyTorch 版翻译"""
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
    
    def translate_onnx(self, text):
        """ONNX 版翻译"""
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

def main():
    """主函数"""
    print("✓ 所有依赖已安装")
    
    # 创建测试实例
    tester = OpusMTTester()
    
    # 简化的测试用例
    test_cases = [
        # 短句测试
        "Hello, nice to meet you.",
        # 中等长度句子
        "The technology we use today will shape our future.",
        # 长句测试
        "The rapid development of machine translation technology has made it possible.While many of my some during sleep so used to say okay lemme and your shifted are added other actually those some of the lab a look at which is as good as to doesn't mean they should not be used gingerly okay here we tell me about some like oh crap."
    ]
    
    # 每个样本只测试一次
    runs = 1
    
    # 运行测试
    results = []
    for text in test_cases:
        result = tester.test_translation(text, runs=runs)
        results.append(result)
        print("=" * 50 + "\n")
    
    # 简化的结果输出
    print("=== 测试总结 ===")
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"平均加速比: {avg_speedup:.2f}x")

if __name__ == "__main__":
    main() 