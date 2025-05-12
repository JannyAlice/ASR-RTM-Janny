# Sherpa-ONNX 插件

本插件提供基于sherpa-onnx系列模型的语音识别功能。

## 支持的模型

- **sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20**
  - 支持中英双语识别
  - 提供标准模型和int8量化模型

- **sherpa-onnx-streaming-zipformer-en-2023-06-26**
  - 专为英语优化的模型
  - 提供标准模型和int8量化模型
  - 识别准确度更高

## 功能特点

- 支持在线实时语音识别
- 支持音频/视频文件转录
- 支持标准模型和int8量化模型
- 使用持久化流模式提高识别准确性

## 依赖项

- sherpa-onnx >= 1.11.2
- numpy
- soundcard (可选，用于处理非WAV格式音频)

## 配置选项

```json
{
    "sherpa_onnx_std": {
        "enabled": true,
        "type": "asr",
        "model_config": "sherpa_onnx_std",
        "plugin_config": {
            "feature_dim": 80,
            "num_threads": 1,
            "debug": false,
            "enable_endpoint_detection": true,
            "rule1_min_trailing_silence": 2.4,
            "rule2_min_trailing_silence": 1.2,
            "rule3_min_utterance_length": 20.0,
            "max_active_paths": 4,
            "buffer_size": 8000
        }
    }
}
```

## 使用方法

1. 确保模型文件已下载并放置在正确位置
2. 在配置文件中启用插件
3. 通过插件管理器加载插件
4. 使用ASR模型管理器创建识别器并处理音频数据

## 模型文件

### 2023-02-20 模型
模型文件应放置在以下路径：
`models/asr/sherpa-onnx`

包含以下文件：
- encoder-epoch-99-avg-1.onnx (或 encoder-epoch-99-avg-1.int8.onnx)
- decoder-epoch-99-avg-1.onnx (或 decoder-epoch-99-avg-1.int8.onnx)
- joiner-epoch-99-avg-1.onnx (或 joiner-epoch-99-avg-1.int8.onnx)
- tokens.txt

### 2023-06-26 模型
模型文件应放置在以下路径：
`models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26`

包含以下文件：
- encoder-epoch-99-avg-1-chunk-16-left-128.onnx (或 encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx)
- decoder-epoch-99-avg-1-chunk-16-left-128.onnx (或 decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx)
- joiner-epoch-99-avg-1-chunk-16-left-128.onnx (或 joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx)
- tokens.txt

## 注意事项

- 此插件需要sherpa-onnx库支持
- 推荐使用标准模型而非int8量化模型以获得更好的识别效果
- 2023-06-26模型在英语识别方面表现更好
