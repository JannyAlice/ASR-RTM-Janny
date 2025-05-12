# 系统音频字幕程序菜单结构说明
# 通用提示词：首先必须基于.cursorrules的规则约定，不能违背。其次，目前基于默认vosk small模型的在线与离线转录功能均已调试正常。接下来，计划逐步增加新模型开发，一个一个来，先从"ASR模型，sherpa-onnx int8量化模型: "开始（成功后再开始sherpa_std标准模型），其模型路径是："PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\sherpa-onnx"（注意int8 and std model are in the identical path），注意统一使用config配置文件来调度，config\config.json。

## 概述

本项目是一个实时音频转录系统，能够捕获系统音频或麦克风输入，并将其转换为文本字幕。该程序使用 PyQt5 构建图形界面，使用 Vosk and sherpa-onnx 2个ASR模型和 RTM 模型，使用 Argostranslate 和 Opus-Mt-ONNX  进行语音识别和翻译，支持多种功能和设置选项。目前ASR采用ASR语音识别模型，运行实时转录正常使用vosk small模型，使用sherpa-onnx int8量化模型，使用sherpa-onnx 标准模型，以上三者可选。RTM翻译模型，使用argostranslate模型，使用opus-mt-onnx模型，以上二者可选。具体模型存放位置与要求实现菜单的要求如下：
模型目录说明：一共要用到7个模型，其中5个属于ASR模型分别是vosk small模型，sherpa-onnx int8量化模型与sherpa-onnx 标准模型，以及sherpa_0626 int8 模型与sherpa_0626 标准模型；其它2个属于RTM模型分别是argostranslate模型，opus-mt-onnx模型。这些模型的路径如下：
ASR模型1，VOSK small 模型路径: PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\vosk\vosk-model-small-en-us-0.15"
ASR模型2，sherpa-onnx int8量化模型路径: PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx" 
ASR模型3，sherpa-onnx 标准模型路径，与其对应的int8量化模型路径相同: 
PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx" 
ASR模型4，sherpa_0626 int8模型路径）：C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26
"sherpa_0626_int8": {
                "path": "models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26",
                "type": "int8",
                "enabled": true,
                "config": {
                    "encoder": "/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                    "decoder": "/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                    "joiner": "/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                    "tokens": "/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt",
....
ASR模型5，sherpa_0626 标准模型路径，与其对应的int8量化模型路径相同：C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26
"sherpa_0626_std": {
                "path": "models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26",
                "type": "standard",
                "enabled": true,
                "config": {
                    "encoder": "/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                    "decoder": "/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                    "joiner": "/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
                    "tokens": "/models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt",
....
RTM模型1，Argostranslate 模型路径: 
PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\translation\argos-translate\packages\translate-en_zh-1_9"
RTM模型2，Opus-MT-onnx 模型路径: PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\translation\opus-mt\en-zh"
根据以上7个模型的实际路径，配置好config.json文件，在配置文件中模型要名称要一一对应。
## 菜单结构图

```计划更改
Menu (主菜单)
├── 转录模式
│   ├── 系统音频模式
│   │   ├── 英文识别
│   │   ├── 中文识别
│   │   └── 自动识别
│   └── 选择音频/视频文件
├── 模型选择
│   ├── ASR 转录模型
│   │   ├── VOSK Small 模型
│   │   ├── Sherpa-ONNX int8量化模型
│   │   └── Sherpa-ONNX 标准模型
│   ├── RTM 翻译模型
│   │   ├── Argostranslate 模型
│   └───└── Opus-Mt-ONNX   模型
├── 背景模式
│   ├── 不透明
│   ├── 半透明
│   └── 全透明
├── 字体大小
│   ├── 小
│   ├── 中
│   └── 大
├── 模型管理
│   ├── 显示系统信息
│   ├── 检查模型目录
│   └── 搜索模型文档
└── 附加功能
│   └── 启用说话人识别
└── 帮助(H)
    ├── 使用说明(F1)
    └── 关于(A)     
```

## 菜单功能说明

### 转录模式

- **系统音频**：捕获系统播放的声音进行转录
  - **英文识别**：使用英文模型进行识别
  - **中文识别**：使用中文模型进行识别
  - **自动识别**：自动检测语言并选择合适的模型

- **音频/视频文件**：从文件中读取音频进行转录

### 音频设备

- 动态生成的系统音频设备列表，用户可以选择要捕获的音频设备



### 背景模式

- **不透明**：窗口完全不透明
- **半透明**：窗口半透明，可以看到下方内容
- **全透明**：窗口几乎全透明，只显示文字

### 字体大小

- **小**：小字体显示
- **中**：中等字体显示
- **大**：大字体显示

### 调试

- **显示系统信息**：显示系统资源使用情况
- **检查模型目录**：检查模型文件是否完整
- **测试音频设备**：测试音频捕获功能
- **搜索模型文档**：查找模型相关文档

### 说话人识别

- **启用说话人识别**：开启/关闭说话人识别功能，可以区分不同说话人

## 界面控件

除了菜单外，主界面还包含以下控件：

1. **字幕显示区域**：显示转录的文本，支持滚动查看历史记录
2. **开始/停止按钮**：控制转录的开始和停止
3. **退出按钮**：退出程序
4. **进度条**：显示文件转录的进度

## 快捷键

- **ESC**：退出程序
- **Ctrl+Q**：退出程序
- **空格键**：开始/停止转录

## 技术实现

程序使用多线程架构，主线程负责界面交互，后台线程负责音频捕获和转录处理。使用 ASR技术 进行语音识别，支持实时转录和文件转录两种模式。程序还支持说话人识别功能，可以区分不同的说话人。 
以下是程序的主要结构：目前还在不断变化中，后续会逐步完善。AI要以自己搜索实际的目录为准。以下仅供参考。
project_root/
├── config/                      # 配置文件目录
│   ├── config.json             # 主配置文件
│   ├── config20250402.json     # 历史配置备份
│   └── default_config.json     # 默认配置
│
├── models/                      # 模型目录
│   ├── asr/                    # ASR 模型
│   │   ├── vosk/              # VOSK 模型
│   │   │   └── small/         # Small 模型
│   │   └── sherpa/            # Sherpa-ONNX 模型
│   │       ├── int8/          # int8量化模型
│   │       └── standard/      # 标准模型
│   │
│   └── translation/            # 翻译模型
│       ├── argos/             # Argostranslate 模型
│       └── opus/              # Opus-Mt-ONNX 模型
│
├── src/                        # 源代码目录
│   ├── asr/                   # ASR 模块
│   │   ├── __init__.py
│   │   ├── manager.py        # ASRManager
│   │   ├── vosk_engine.py    # VOSK 引擎实现
│   │   └── sherpa_engine.py  # Sherpa-ONNX 引擎实现
│   │
│   ├── translation/          # 翻译模块
│   │   ├── __init__.py
│   │   ├── manager.py       # TranslationManager
│   │   ├── argos_engine.py  # Argostranslate 引擎
│   │   └── opus_engine.py   # Opus-Mt-ONNX 引擎
│   │
│   ├── audio/               # 音频处理模块
│   │   ├── __init__.py
│   │   ├── processor.py     # 音频处理
│   │   └── recorder.py      # 音频录制
│   │
│   ├── ui/                  # UI 模块
│   │   ├── __init__.py
│   │   ├── main_window.py   # 主窗口
│   │   ├── menu/           # 菜单模块
│   │   │   ├── __init__.py
│   │   │   ├── transcription_menu.py  # 转录模式菜单
│   │   │   ├── model_menu.py         # 模型选择菜单
│   │   │   ├── background_menu.py    # 背景模式菜单
│   │   │   ├── font_menu.py          # 字体大小菜单
│   │   │   ├── model_management_menu.py # 模型管理菜单
│   │   │   ├── extra_menu.py         # 附加功能菜单
│   │   │   └── help_menu.py          # 帮助菜单
│   │   │
│   │   ├── windows/        # 窗口模块
│   │   │   ├── __init__.py
│   │   │   └── subtitle_window.py    # 字幕窗口
│   │   │
│   │   └── widgets/        # 控件模块
│   │       ├── __init__.py
│   │       ├── status_bar.py         # 状态栏
│   │       └── settings_panel.py     # 设置面板
│   │
│   └── utils/              # 工具模块
│       ├── __init__.py
│       ├── config.py      # 配置管理
│       ├── logger.py      # 日志管理
│       └── common.py      # 通用工具函数
│
├── tests/                  # 测试目录
│   ├── __init__.py
│   ├── test_asr.py        # ASR 测试
│   ├── test_translation.py # 翻译测试
│   └── test_audio.py      # 音频测试
│
├── docs/                  # 文档目录
│   ├── user_guide.md      # 用户指南
│   └── model_docs/        # 模型文档
│
└── main.py               # 主程序入口
```
