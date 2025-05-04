# 测试目录结构说明

## tests 和 test_tools 的关系

在本项目中，测试相关的文件被分为两个文件夹：

- **tests 文件夹**（当前文件夹）：包含实际的测试代码（测试用例）。这是"测试内容"，定义了要测试什么。
- **test_tools 文件夹**：包含用于运行测试的工具和脚本。这是"测试工具"，定义了如何运行这些测试。

简单来说，`tests` 包含测试代码，而 `test_tools` 包含运行这些测试的工具。

## 测试目录结构

本测试目录按照以下结构组织：

```
tests/
├── unit/                  # 单元测试
│   ├── core/              # 核心功能测试
│   │   ├── audio/         # 音频处理测试
│   │   └── ...
│   ├── ui/                # 用户界面测试
│   └── utils/             # 工具函数测试
│       └── test_config_manager.py  # 配置管理器测试
├── integration/           # 集成测试
└── data/                  # 测试数据
```

## 测试命名约定

- 测试文件名应以 `test_` 开头
- 测试类名应以 `Test` 开头
- 测试方法名应以 `test_` 开头

例如：
- 文件名：`test_config_manager.py`
- 类名：`TestConfigManager`
- 方法名：`test_load_config`

## 运行测试

要运行测试，请使用 `test_tools` 目录中的工具：

1. 从项目根目录运行 `run_tests.bat`
2. 或者直接使用 `test_tools` 目录中的批处理文件：
   - `test_tools\run_tests_in_vosk.bat` - 运行所有测试
   - `test_tools\run_config_tests_in_vosk.bat` - 运行配置测试
   - `test_tools\run_audio_tests_in_vosk.bat` - 运行音频测试

## 编写新测试

添加新测试时，请遵循以下步骤：

1. 确定测试类型（单元测试、集成测试等）
2. 在相应的目录中创建测试文件
3. 遵循测试命名约定
4. 使用 pytest 框架编写测试
5. 运行测试以确保它们正常工作

## 测试依赖

本项目使用 pytest 作为测试框架。如果尚未安装，可以使用以下命令安装：

```
test_tools\install_to_vosk.bat pytest
```
