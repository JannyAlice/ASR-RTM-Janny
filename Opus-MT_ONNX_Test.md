### **OPUS-MT + ONNX 加速：测试方案**
为了测试 OPUS-MT 的 ONNX 加速效果，我们需要：
1. **转换 OPUS-MT 模型到 ONNX 格式**
2. **运行基准测试，测量 PyTorch 和 ONNX 运行速度**
3. **对比两者的翻译时间，评估 ONNX 加速效果**

---

## **1. 安装依赖**
首先，确保 `vosk-api` 这个 conda 环境中安装了以下依赖：
```bash
conda activate vosk-api
pip install torch transformers sentencepiece onnx onnxruntime
```

---

## **2. 转换 OPUS-MT 为 ONNX**
下面的代码会：
- **下载并加载 OPUS-MT 模型**
- **转换为 ONNX 格式**
- **保存为 `opus-mt-en-zh.onnx`**
```python
import torch
from transformers import MarianMTModel

# 加载 OPUS-MT（英语 -> 中文）
model_name = "Helsinki-NLP/opus-mt-en-zh"
model = MarianMTModel.from_pretrained(model_name)

# 定义 ONNX 模型保存路径
onnx_model_path = "opus-mt-en-zh.onnx"

# 创建示例输入（模拟 12 个 token 的句子）
dummy_input = torch.ones((1, 12), dtype=torch.int64)

# 转换模型
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_model_path,
    input_names=["input_ids"], 
    output_names=["output_ids"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}}
)

print(f"模型已转换并保存为 {onnx_model_path}")
```

---

## **3. 测试 PyTorch 和 ONNX 翻译速度**
该代码会：
1. **测试 PyTorch 版 OPUS-MT 的翻译时间**
2. **测试 ONNX 版 OPUS-MT 的翻译时间**
3. **对比两者的延迟**

```python
import time
import torch
import onnxruntime as ort
from transformers import MarianMTModel, MarianTokenizer

# 选择模型（英语 -> 中文）
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)

# 加载 PyTorch 版 OPUS-MT
model = MarianMTModel.from_pretrained(model_name)

# 加载 ONNX 版 OPUS-MT
onnx_model_path = "opus-mt-en-zh.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

def translate_pytorch(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    start_time = time.time()
    translated = model.generate(**inputs)
    end_time = time.time()

    translation = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    latency = end_time - start_time

    return translation, latency

def translate_onnx(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].numpy()  # 转换为 numpy 格式

    start_time = time.time()
    translated = ort_session.run(None, {"input_ids": input_ids})[0]
    end_time = time.time()

    translation = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    latency = end_time - start_time

    return translation, latency

# 测试文本
test_text = "Hello, how are you today?"

# 测试 PyTorch 版
pytorch_translation, pytorch_latency = translate_pytorch(test_text)

# 测试 ONNX 版
onnx_translation, onnx_latency = translate_onnx(test_text)

# 打印结果
print("=== PyTorch 版翻译 ===")
print(f"原文: {test_text}")
print(f"翻译: {pytorch_translation}")
print(f"翻译时间: {pytorch_latency:.4f} 秒")

print("\n=== ONNX 版翻译 ===")
print(f"原文: {test_text}")
print(f"翻译: {onnx_translation}")
print(f"翻译时间: {onnx_latency:.4f} 秒")

# 计算加速比例
speedup = pytorch_latency / onnx_latency if onnx_latency > 0 else float("inf")
print(f"\nONNX 加速比: {speedup:.2f} 倍")
```

---

## **4. 运行测试，评估 ONNX 版加速效果**
### **测试结果示例**
| 方式 | 翻译时间 |
|------|---------|
| PyTorch 版 OPUS-MT | **1.52 秒** |
| ONNX 版 OPUS-MT | **0.76 秒** |
| **加速比** | **2.0 倍** |

> **i5-7200U 上可能会稍慢**，但 ONNX **通常会快 1.5~3 倍**。

---

## **5. 结论**
| 方案 | 适合实时翻译？ | 说明 |
|------|--------------|-----------------------------------|
| **PyTorch 版 OPUS-MT** | ❌ **可能过慢** | 翻译短句需 **1~3 秒** |
| **ONNX 版 OPUS-MT** | ✅ **较适合** | 翻译短句约 **0.5~1.5 秒** |
| **Bergamot API** | ✅ **更优** | 翻译短句仅 **0.1~0.5 秒** |

如果你需要 **<1s 级别的实时翻译**，ONNX 是一个不错的选择。  
但如果 ONNX 仍然不够快，你可以考虑 **Bergamot Web API**，它的翻译速度会更快。

---

### **下一步建议**
1. **运行测试代码**，看看 ONNX 版在你的 i5-7200U 上是否足够快。
2. 如果 ONNX 仍然不够快，可以尝试 **Bergamot API**，我可以提供相关安装和优化方案。 🚀

测试完成后，把你的 **PyTorch vs ONNX 运行时间** 结果告诉我，我可以帮你优化方案！