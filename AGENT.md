## 一、代码规范
## 1. 总体说明－有关模型和参数说明
本项目是一个实时音频转录系统，能够捕获系统音频或麦克风输入，并将其转换为文本字幕。该程序使用 PyQt5 构建图形界面，ASR模型使用 Vosk and sherpa-onnx 和 RTM 模型使用 Argostranslate 和 Opus-Mt-ONNX  进行语音识别和翻译，支持多种功能和设置选项。目前ASR语音识别模型，运行实时转录正常使用vosk small模型，sherpa_0220_int8量化模型，sherpa_0220_std 标准模型，sherpa_0226_int8量化模型，sherpa_0226_std 标准模型共5个模型。RTM翻译模型使用argostranslate模型，opus-mt-onnx模型等共2个模型。具体模型存放位置与要求实现菜单的要求如下：
模型目录说明：一共要用到7个模型，其中5个属于ASR模型，其它2个属于RTM模型分别是argostranslate模型，opus-mt-onnx模型。这些模型的路径如下：
全部模型目录路径：PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models"
ASR模型1，VOSK small 模型路径: PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\vosk\vosk-model-small-en-us-0.15"
ASR模型2，sherpa_0220_int8量化模型路径: PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx" 
ASR模型3，sherpa_0220_std 标准模型路径：PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx" 
ASR模型4，sherpa_0626 int8模型路径）：PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26"
ASR模型5，sherpa_0626 std 标准模型路径：PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26"
RTM模型1，Argostranslate 模型路径: 
PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\translation\argos-translate\packages\translate-en_zh-1_9"
RTM模型2，Opus-MT-onnx 模型路径: PATH:"C:\Users\crige\RealtimeTrans\vosk-api\models\translation\opus-mt\en-zh"

模型使用说明：不要修改上述模型路径与模型名称，否则可能会导致程序无法运行。引用模型路径时，使用绝对路径，不要使用相对路径。
1. **通用编码规则**
   - [PEP8]  AI 代码补全时，必须遵循 PEP8 规范
   - [STRUCT] 本项目采用PyQt5框架，其代码必须保持清晰的 MVC 结构
   - [SAFE] 不能随意删除、修改、重构已有代码，除非特别要求。
   - [REFERENCE] 遵循"扩展而非修改"的原则，是解决当前问题的最佳方式，如果想要借用共用方法，在共用方法不修改的前提下，是可以借用的，
   但是为了防止互相影响，不能对共用方法进行修改，可以不通过修改而采取单独建立方法的方式来解决问题。例如，不能因修改其它代码影响
   在线音频转录与音视频文件的转录功能。
   - [CONFIRM] 代码修改完成后，必须弹出 "Accept All" or "Keep All" 提示，不允许直接覆盖代码。
   - [REFERENCE] 有关模型参数，请参考 config/config.json 文件。不允许随意修改以及硬编码，除非明确指示。
   - [Comment] 代码注释必须使用中文,注意不允许采用unicode的数字串形式，并遵循 PEP8 规范。
   - [SYSTEM] 本机平台使用的是 Windows 10 64位操作系统。不允许使用其它操作系统的命令或函数方法，比如不能使用unix/linux的命令或函数方法。
2. **ASR 模块规范** (Vosk/Sherpa-ONNX)
- [ASR-FLOW] 保持现有 ASR 处理流程不变：
     - 遵守插件式架构设计原则
     - 模型加载顺序
     - 音频处理管道
     - 推理调用方式     
     - [ASR-API] 禁止删除/修改现有公共方法
     - 未得到允许的前提下不能改动 ASR 处理流程（如 Vosk Model 加载、Sherpa-ONNX 推理）。
     - 代码优化时，**不能删除** 任何已有函数或方法，除非得到明确同意。
- [ASR-MODEL] 使用的模型如下，共5个模型，其名称与路径均应按照文件中的配置以绝对路径进行加载。"asr": {"vosk_small","sherpa_0220_int8","sherpa_0220_std","sherpa_0626_int8","sherpa_0626_std"}
3.**RTM 模块规范** (ArgosTranslate/Opus-MT)
   - [MOD-SEP] ASR 与 RTM 代码必须物理分离：
        - 遵守插件式架构设计原则
        - 独立目录结构
        - 禁止共享处理管道
   - [RTM-CORE] 翻译模型核心逻辑禁止修改：
        - 模型加载方式
        - 文本预处理流程
        - 结果后处理
- [RTM-MODEL] 使用的模型如下，共2个模型，其名称与路径均应按照此文件中的配置以绝对路径进行加载。"translation": {"opus","argos"}     
- ASR 代码和 RTM 代码必须遵守插件式设计原则，单独分开处理，不能混用。
- AI 不能调整 Opus-MT 模型的翻译管道，除非明确指示。

## 二、版本控制
1. **代码提交**
   - [LINT] 提交前必须通过 `flake8 --max-line-length=120` 检查
   - [TEST] 涉及核心功能需通过现有测试用例
2. **Commit Message**
   - Git commit message 必须包含：
   - `[ASR]` - 变更涉及 ASR 代码。
   - `[RTM]` - 变更涉及翻译模块。
   - `[UI]`  - 变更 PyQt5 图形界面。
## 三、交互规则
# 交互规则
- 最好的解决问题的办法，让AI先把所有怀疑的存在问题可能先全部列出来，然后我们都全部检查一遍，从头至尾，这样再往下进行解决问题，先把问题搞清楚。比如，请马上列一下所有检查清单给我，然后一步一步来检查，我将每一步的检查结果告诉你，过程中间绝对不能跳，只有我将测试结果反馈给你，然后AI才能进行下一步，另外AI每次收到我反馈的测试结果后，给出解决方案的同时要明确提示我下一步要做什么！
- 代码修改完成后，**总是**让编辑器弹出提示 "Accept All" and "Reject"按钮。
- 代码修改必须 **逐步进行，并给出详细解释**，不能一次性覆盖多个文件。
- 代码补全时，**优先复用现有函数**，避免重复实现相同功能。
- 代码测试，测试文件不创建在根目录下的，必须创建在\tests\下。
- 所有测试文件，必须生成带有时间戳及日志分析功能来分析测试结果的日志文件（日志文件存放在\logs\下），以便于调试。
- 你回答问题时始终使用“中文”。
## 四、维护规则
# 维护规则
- 变更需记录日志，确保团队成员理解最新规则。
- 重要修改需同步到 README.md 说明，确保一致性。

## 五、构建与测试命令
- 运行全部测试：`python tests/run_tests.py`
- 运行单个测试文件：`pytest tests/test_file.py -v`
- 运行指定测试函数：`pytest tests/test_file.py::test_function -v`
- 代码格式检查：`flake8 --max-line-length=120`
- 生成带时间戳的测试日志：运行测试时自动生成在`logs/`目录下