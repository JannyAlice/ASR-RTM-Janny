"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的在线转录功能
"""
import os
import sys
import time
import numpy as np
import soundcard as sc
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入 Sherpa-ONNX 日志工具
from src.utils.sherpa_logger import sherpa_logger

# 导入模型管理器
from src.core.asr.model_manager import ASRModelManager

def test_sherpa_0626_online_transcription():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的在线转录功能"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的在线转录功能")
    print("=" * 80)
    
    # 设置日志
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")
    
    # 创建模型管理器
    print("创建模型管理器...")
    model_manager = ASRModelManager()
    
    # 加载 sherpa_0626 模型
    print("加载 sherpa_0626 模型...")
    if not model_manager.load_model("sherpa_0626"):
        print("加载 sherpa_0626 模型失败")
        return
    
    # 创建识别器
    print("创建识别器...")
    recognizer = model_manager.create_recognizer()
    if not recognizer:
        print("创建识别器失败")
        return
    
    # 设置音频参数
    sample_rate = 16000
    channels = 1
    
    # 获取默认的系统音频输出设备
    print("获取系统音频输出设备...")
    try:
        # 尝试查找 "CABLE Output" 设备
        speakers = sc.all_speakers()
        default_speaker = None
        
        for speaker in speakers:
            if "CABLE" in speaker.name:
                default_speaker = speaker
                print(f"找到 CABLE 设备: {speaker.name}")
                break
        
        if not default_speaker:
            # 如果没有找到 CABLE 设备，使用默认设备
            default_speaker = sc.default_speaker()
            print(f"使用默认设备: {default_speaker.name}")
    except Exception as e:
        print(f"获取音频设备错误: {e}")
        return
    
    # 开始捕获音频
    print("开始捕获音频...")
    try:
        # 创建保存目录
        save_dir = os.path.join(project_root, "transcripts")
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(save_dir, f"online_test_{timestamp}.txt")
        
        # 打开文件用于保存转录结果
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"在线转录测试 - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
        
        print(f"转录结果将保存到: {txt_path}")
        
        # 设置捕获时间（秒）
        capture_time = 30
        start_time = time.time()
        end_time = start_time + capture_time
        
        print(f"将捕获 {capture_time} 秒的音频...")
        
        # 开始捕获
        with default_speaker.recorder(samplerate=sample_rate, channels=channels) as mic:
            # 重置识别器
            recognizer.Reset()
            
            # 循环捕获音频
            while time.time() < end_time:
                # 捕获音频数据
                data = mic.record(numframes=sample_rate // 4)  # 捕获 0.25 秒的音频
                
                # 转换为 numpy 数组
                audio_data = np.squeeze(data)
                
                # 确保数据是 float32 类型
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # 确保数据范围在 [-1, 1]
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # 转换为 int16
                audio_data_int16 = (audio_data * 32767).astype(np.int16)
                
                # 转换为字节
                audio_bytes = audio_data_int16.tobytes()
                
                # 处理音频数据
                if recognizer.AcceptWaveform(audio_bytes):
                    # 获取结果
                    result = recognizer.Result()
                    result_json = json.loads(result)
                    
                    if "text" in result_json and result_json["text"]:
                        text = result_json["text"]
                        print(f"最终结果: {text}")
                        
                        # 保存到文件
                        with open(txt_path, "a", encoding="utf-8") as f:
                            f.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")
                else:
                    # 获取部分结果
                    partial = recognizer.PartialResult()
                    partial_json = json.loads(partial)
                    
                    if "partial" in partial_json and partial_json["partial"]:
                        text = partial_json["partial"]
                        print(f"部分结果: {text}")
                
                # 显示剩余时间
                remaining = end_time - time.time()
                if remaining > 0:
                    print(f"剩余时间: {remaining:.1f} 秒", end="\r")
        
        # 获取最终结果
        final = recognizer.FinalResult()
        final_json = json.loads(final)
        
        if "text" in final_json and final_json["text"]:
            text = final_json["text"]
            print(f"最终结果: {text}")
            
            # 保存到文件
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")
        
        # 获取格式化的转录文本
        formatted_transcript = recognizer.get_formatted_transcript()
        if formatted_transcript:
            print("\n格式化的转录文本:")
            print(formatted_transcript)
            
            # 保存到文件
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write("\n\n格式化的转录文本:\n")
                f.write(formatted_transcript)
        
        print("\n转录完成")
        print(f"转录结果已保存到: {txt_path}")
        
    except Exception as e:
        print(f"捕获音频错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("测试完成")

if __name__ == "__main__":
    # 导入 json 模块
    import json
    
    test_sherpa_0626_online_transcription()
