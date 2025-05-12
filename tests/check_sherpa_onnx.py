"""
检查 sherpa-onnx 版本和 API
"""
import sys
import inspect

try:
    import sherpa_onnx
    
    # 打印版本信息
    print(f"Sherpa-ONNX 版本: {getattr(sherpa_onnx, '__version__', '未知')}")
    
    # 打印可用的类和方法
    print("\nSherpa-ONNX 可用的类和方法:")
    for name, obj in inspect.getmembers(sherpa_onnx):
        if not name.startswith('_'):  # 排除内部属性
            obj_type = type(obj).__name__
            print(f"  - {name}: {obj_type}")
            
            # 如果是类，打印其方法
            if inspect.isclass(obj):
                print(f"    类方法:")
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    if not method_name.startswith('_'):
                        print(f"      - {method_name}{inspect.signature(method)}")
    
    # 尝试创建一个识别器
    print("\n尝试创建一个识别器:")
    
    # 方法1: 尝试使用 OnlineRecognizerConfig
    try:
        print("方法1: 使用 OnlineRecognizerConfig")
        config = sherpa_onnx.OnlineRecognizerConfig()
        print("  成功创建 OnlineRecognizerConfig")
    except Exception as e:
        print(f"  创建 OnlineRecognizerConfig 失败: {e}")
    
    # 方法2: 尝试直接创建 OnlineRecognizer
    try:
        print("\n方法2: 直接创建 OnlineRecognizer")
        # 使用最简单的参数调用
        recognizer = sherpa_onnx.OnlineRecognizer(
            tokens="C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\sherpa-onnx\\tokens.txt",
            encoder="C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\sherpa-onnx\\encoder-epoch-99-avg-1.int8.onnx",
            decoder="C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\sherpa-onnx\\decoder-epoch-99-avg-1.int8.onnx",
            joiner="C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\sherpa-onnx\\joiner-epoch-99-avg-1.int8.onnx",
        )
        print("  成功创建 OnlineRecognizer")
        
        # 创建流
        stream = recognizer.create_stream()
        print("  成功创建 stream")
        
        # 打印 stream 的属性和方法
        print("  Stream 属性和方法:")
        for name, obj in inspect.getmembers(stream):
            if not name.startswith('_'):
                obj_type = type(obj).__name__
                if inspect.ismethod(obj):
                    print(f"    - {name}{inspect.signature(obj)}: {obj_type}")
                else:
                    print(f"    - {name}: {obj_type}")
        
    except Exception as e:
        print(f"  创建 OnlineRecognizer 失败: {e}")
        import traceback
        print(traceback.format_exc())

except ImportError:
    print("未安装 sherpa-onnx 模块")
    sys.exit(1)
