# 系统音频字幕程序菜单结构说明

## 概述

`system_audio_subtitles.py` 是一个实时音频转录系统，能够捕获系统音频或麦克风输入，并将其转换为文本字幕。该程序使用 PyQt5 构建图形界面，使用 Vosk and sherpa-onnx 2个ASR模型和 RTM 模型，使用 Argostranslate 和 Opus-Mt-ONNX  进行语音识别和翻译，支持多种功能和设置选项。本次修改，是对system_audio_subtitles.py文件进行功能改造，此文件的ASR采用ASR语音识别模型，运行实时转录正常。这次修改，采用ASR语音识别模型，使用vosk small模型，使用sherpa-onnx模型，以上二者可选，采用RTM翻译模型，使用argostranslate模型，使用opus-mt-onnx模型，以上二者可选。具体模型存放位置与要求实现菜单的要求如下：
模型目录说明：一共要用到5个模型，分别是vosk small模型，sherpa-onnx int8量化模型，sherpa-onnx 标准模型，前3个属于ASR模型，argostranslate模型，opus-mt-onnx模型，最后2个属于RTM模型。
ASR模型，VOSK small 模型路径: PATH:"C:\Users\crige\RealtimeTrans\vosk-api\model\vosk-model-small-en-us-0.15"
ASR模型，sherpa-onnx int8量化模型路径: PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\sherpa-onnx" 
ASR模型，sherpa-onnx 标准模型路径: PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\sherpa-onnx" 
RTM模型，Argostranslate 模型路径: PATH:"C:\Users\crige\.local\share\argos-translate\packages\translate-en_zh-1_9\model"
RTM模型，Opus-MT-onnx 模型路径: PATH:"c:\Users\crige\RealtimeTrans\vosk-api\models\opus-mt\en-zh"
## 菜单结构图

```计划更改
Menu (主菜单)
├── 转录模式
│   ├── 系统音频
│   │   ├── 英文识别
│   │   ├── 中文识别
│   │   └── 自动识别
│   └── 音频/视频文件
├── 音频设备
│   └── [动态生成的系统音频设备列表]
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
├── 调试
│   ├── 显示系统信息
│   ├── 检查模型目录
│   ├── 测试音频设备
│   └── 搜索模型文档
└── 说话人识别
    └── 启用说话人识别
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