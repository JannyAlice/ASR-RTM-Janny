# 更新日志

本文件记录项目的所有重要变更。

## [未发布]

### 新增

- 添加了新的生命周期信号：`transcription_started`、`transcription_paused`、`transcription_resumed`
- 添加了信号系统使用指南文档：`docs/signal_guide.md`
- 添加了信号测试文件：`tests/test_signals.py`

### 修改

- 修改了`src/core/signals.py`中的`TranscriptionSignals`类，添加了详细的文档和新的生命周期信号
- 修改了`src/ui/main_window.py`中的`_connect_signals`方法，添加了信号存在性检查
- 修改了信号处理方法，添加了try-except块和详细的日志记录
- 修改了`src/core/audio/file_transcriber.py`中的信号使用，添加了新的生命周期信号发射

### 修复

- 修复了`TranscriptionSignals`类缺少`transcription_finished`信号的问题
- 修复了信号连接没有检查信号是否存在的问题
- 修复了信号处理没有错误捕获的问题

## [2025-05-05] - UI布局修复

### 修复

- 修复了UI布局混乱的问题，使用`src/ui/main_window.py`中的`MainWindow`类替代`main.py`中的自定义`MainWindow`类
- 修复了COM初始化和清理的问题，确保COM只被初始化和清理一次
- 修复了`ConfigManager`类缺少`get_ui_config`方法的问题

## [2025-05-01] - 初始版本

### 新增

- 实现了基本的实时语音转录功能
- 支持Vosk和Sherpa-ONNX模型
- 支持系统音频捕获和文件转录
- 实现了基本的UI界面，包括菜单栏、字幕窗口和控制面板
