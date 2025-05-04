"""
详细检查 sherpa-onnx API
"""
import sys
import inspect

try:
    import sherpa_onnx
    
    # 打印版本信息
    print(f"Sherpa-ONNX 版本: {getattr(sherpa_onnx, '__version__', '未知')}")
    
    # 检查 online_recognizer 模块
    print("\n检查 online_recognizer 模块:")
    if hasattr(sherpa_onnx, 'online_recognizer'):
        online_recognizer = sherpa_onnx.online_recognizer
        print("  模块存在")
        
        # 打印模块属性和方法
        for name, obj in inspect.getmembers(online_recognizer):
            if not name.startswith('_'):  # 排除内部属性
                obj_type = type(obj).__name__
                print(f"  - {name}: {obj_type}")
                
                # 如果是函数或方法，打印其签名
                if inspect.isfunction(obj) or inspect.ismethod(obj):
                    print(f"    签名: {inspect.signature(obj)}")
                
                # 如果是类，打印其方法
                if inspect.isclass(obj):
                    print(f"    类方法:")
                    for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                        if not method_name.startswith('_'):
                            print(f"      - {method_name}{inspect.signature(method)}")
    else:
        print("  模块不存在")
    
    # 检查 OnlineRecognizer 类
    print("\n检查 OnlineRecognizer 类:")
    if hasattr(sherpa_onnx, 'OnlineRecognizer'):
        OnlineRecognizer = sherpa_onnx.OnlineRecognizer
        print("  类存在")
        
        # 打印类的构造函数签名
        print(f"  构造函数签名: {inspect.signature(OnlineRecognizer.__init__)}")
        
        # 打印类的方法
        print("  类方法:")
        for name, method in inspect.getmembers(OnlineRecognizer, inspect.isfunction):
            if not name.startswith('_'):
                print(f"    - {name}{inspect.signature(method)}")
        
        # 尝试创建一个实例
        print("\n尝试创建一个实例:")
        try:
            # 尝试不带参数创建
            recognizer = OnlineRecognizer()
            print("  成功创建实例（无参数）")
        except Exception as e:
            print(f"  创建实例失败（无参数）: {e}")
            
            # 尝试使用 from_conformer 方法
            if hasattr(OnlineRecognizer, 'from_conformer'):
                try:
                    print("\n尝试使用 from_conformer 方法:")
                    model_path = "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\sherpa-onnx"
                    recognizer = OnlineRecognizer.from_conformer(
                        encoder=f"{model_path}\\encoder-epoch-99-avg-1.int8.onnx",
                        decoder=f"{model_path}\\decoder-epoch-99-avg-1.int8.onnx",
                        joiner=f"{model_path}\\joiner-epoch-99-avg-1.int8.onnx",
                        tokens=f"{model_path}\\tokens.txt"
                    )
                    print("  成功使用 from_conformer 创建实例")
                except Exception as e:
                    print(f"  使用 from_conformer 创建实例失败: {e}")
            
            # 尝试使用 from_transducer 方法
            if hasattr(OnlineRecognizer, 'from_transducer'):
                try:
                    print("\n尝试使用 from_transducer 方法:")
                    model_path = "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\sherpa-onnx"
                    recognizer = OnlineRecognizer.from_transducer(
                        encoder=f"{model_path}\\encoder-epoch-99-avg-1.int8.onnx",
                        decoder=f"{model_path}\\decoder-epoch-99-avg-1.int8.onnx",
                        joiner=f"{model_path}\\joiner-epoch-99-avg-1.int8.onnx",
                        tokens=f"{model_path}\\tokens.txt"
                    )
                    print("  成功使用 from_transducer 创建实例")
                except Exception as e:
                    print(f"  使用 from_transducer 创建实例失败: {e}")
    else:
        print("  类不存在")

except ImportError:
    print("未安装 sherpa-onnx 模块")
    sys.exit(1)
