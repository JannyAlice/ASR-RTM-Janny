"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型文件检测
"""
import os
import sys
import time

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入必要的模块
from src.core.asr.sherpa_engine import SherpaOnnxASR
from src.utils.sherpa_logger import sherpa_logger

def test_sherpa_0626_file_detection():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型文件检测"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型文件检测")
    print("=" * 80)

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 模型路径
    model_path = r"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26"
    
    # 列出目录中的所有文件
    print(f"模型目录: {model_path}")
    print("目录中的文件:")
    for file in os.listdir(model_path):
        if file.endswith(".onnx"):
            print(f"  {file}")
    
    # 创建 SherpaOnnxASR 实例
    print(f"\n创建 SherpaOnnxASR 实例，路径: {model_path}")
    engine = SherpaOnnxASR(model_path, {"type": "standard", "name": "0626"})
    
    # 设置引擎
    print("设置引擎...")
    if engine.setup():
        print("引擎设置成功")
        
        # 测试文件路径
        test_file = input("请输入要转录的音频文件路径: ")
        if not os.path.exists(test_file):
            print(f"文件不存在: {test_file}")
            return
        
        # 转录文件
        print(f"开始转录文件: {test_file}")
        start_time = time.time()
        result = engine.transcribe_file(test_file)
        end_time = time.time()
        
        if result:
            print(f"转录成功，耗时: {end_time - start_time:.2f} 秒")
            print(f"转录结果: {result}")
            
            # 保存转录结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(project_root, "transcripts", f"transcript_0626_detection_{timestamp}.txt")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Sherpa-ONNX 2023-06-26 模型文件转录结果\n")
                f.write(f"# 文件: {test_file}\n")
                f.write(f"# 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
                f.write(f"# 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
                f.write(f"# 耗时: {end_time - start_time:.2f} 秒\n\n")
                f.write(result)
            print(f"转录结果已保存到: {output_file}")
        else:
            print("转录失败")
    else:
        print("引擎设置失败")

if __name__ == "__main__":
    test_sherpa_0626_file_detection()
