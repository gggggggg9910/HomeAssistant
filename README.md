# 家庭智能助手 (Home Assistant)

基于树莓派5的智能语音助手，支持关键词唤醒、语音识别、AI对话和语音合成功能。

## 功能特性

- 🎤 **关键词唤醒**: 使用sherpa-onnx-kws实现低功耗关键词检测
- 🗣️ **语音识别**: 基于sherpa-onnx的中文语音识别
- 🤖 **AI对话**: 集成OpenAI GPT等大语言模型
- 🔊 **语音合成**: 支持pyttsx3和edge-tts两种TTS引擎
- 🏗️ **模块化设计**: 高度模块化的架构，易于扩展和维护

## 技术架构

### 核心组件
- **关键词唤醒**: sherpa-onnx-kws-zipformer-wenetspeech-3.3M
- **语音识别**: 阿里SenseVoice本地模型 (离线语音识别)
- **语音合成**: 阿里CosyVoice2-0.5B (离线语音合成)
- **大模型接口**: 阿里Qwen模型 (本地/云端API)
- **音频处理**: PyAudio (跨平台音频I/O)

### 系统架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │ -> │ Keyword Spotting│ -> │Speech Recognition│
│   (麦克风)      │    │   (唤醒检测)    │    │   (语音转文本)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                         │
                                                         v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM Query     │ <- │   AI Dialogue   │ -> │  Text-to-Speech │
│   (外部大模型)  │    │   (对话处理)    │    │   (文本转语音)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                         │
                                                         v
                                               ┌─────────────────┐
                                               │  Audio Output   │
                                               │   (扬声器)      │
                                               └─────────────────┘
```

### 模块设计
```
core/
├── audio/          # 底层音频输入输出接口
├── kws/           # 关键词唤醒模块
├── asr/           # 语音识别模块
├── tts/           # 语音合成模块
├── llm/           # 大模型接口模块
└── controller.py  # 主控制逻辑和状态管理

config/            # 配置文件和设置
utils/             # 工具函数和辅助模块
```

## 快速开始

### 1. 环境准备

确保树莓派已安装必要的依赖：

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python和音频相关依赖
sudo apt install python3 python3-pip python3-venv portaudio19-dev

# 安装sherpa-onnx (如果尚未安装)
# 请参考sherpa-onnx官方文档进行安装
```

### 2. 下载项目

```bash
git clone <repository-url>
cd HomeAssistant
```

### 3. 安装依赖

#### Ubuntu/Debian 用户（推荐）：
```bash
# 使用专用安装脚本（会自动处理环境问题）
chmod +x install_deps.sh
./install_deps.sh
```

#### 其他系统或手动安装：
```bash
# 使用启动脚本（推荐）
chmod +x start.sh
./start.sh

# 或手动安装
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 如果遇到 externally-managed-environment 错误，运行修复脚本
chmod +x fix_env.sh
./fix_env.sh
```

### 4. 配置模型文件

模型会自动从ModelScope下载，你也可以手动指定路径：

```bash
# 创建模型目录
mkdir -p ~/models

# 模型会自动下载到以下位置：
# - 关键词唤醒: ~/models/kws-onnx (sherpa-onnx-kws)
# - 语音识别: ~/models/sensevoice (阿里SenseVoice)
# - 语音合成: ~/models/cosyvoice (阿里CosyVoice2-0.5B)
# - LLM本地模型: ~/models/qwen (如果使用本地Qwen模型)
```

### 5. 配置环境变量

复制配置文件模板：

```bash
cp config_template.txt .env
```

编辑 `.env` 文件，配置你的API密钥和其他设置：

```bash
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-3.5-turbo

# 关键词设置
KWS_KEYWORD=你好小助手

# 其他配置...
```

### 6. 测试配置

运行配置测试脚本：

```bash
python test_config.py
```

### 7. 运行助手

```bash
# 激活虚拟环境（如果使用启动脚本则自动激活）
source venv/bin/activate

# 启动语音助手
python main.py
```

## 配置说明

### 环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里云DashScope API密钥 | 必需 |
| `LLM_MODEL` | Qwen模型 | qwen-turbo |
| `LLM_USE_LOCAL` | 是否使用本地Qwen模型 | false |
| `KWS_KEYWORD` | 唤醒关键词 | 你好小助手 |
| `KWS_MODEL_PATH` | KWS模型路径 | ~/models/kws-onnx |
| `ASR_MODEL_ID` | SenseVoice模型ID | iic/SenseVoiceSmall |
| `TTS_MODEL_ID` | CosyVoice模型ID | iic/CosyVoice2-0.5B |
| `TTS_VOICE` | TTS语音类型 | 中文女 |
| `ASR_USE_GPU` | ASR使用GPU | false |
| `TTS_USE_GPU` | TTS使用GPU | false |
| `LOG_LEVEL` | 日志级别 | INFO |

### 音频配置

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `AUDIO_SAMPLE_RATE` | 采样率 | 16000 |
| `AUDIO_CHANNELS` | 声道数 | 1 |
| `AUDIO_CHUNK_SIZE` | 音频块大小 | 1024 |

## 使用方法

1. **启动助手**: 运行 `python main.py`
2. **唤醒**: 说唤醒词 "你好小助手"
3. **对话**: 等待提示音后说话
4. **响应**: 助手会语音回复

### 状态指示

助手有以下状态：
- `idle`: 空闲，等待唤醒
- `listening_for_keyword`: 监听关键词
- `recognizing_speech`: 正在识别语音
- `processing_request`: 处理请求
- `speaking_response`: 播放响应

## 故障排除

### 常见问题

1. **externally-managed-environment 错误**
   ```bash
   # Ubuntu/Debian的新安全特性，尝试以下解决方案：

   # 方案1：使用修复脚本
   chmod +x fix_env.sh
   ./fix_env.sh

   # 方案2：手动创建新的虚拟环境
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # 方案3：检查环境状态
   echo $VIRTUAL_ENV
   which pip
   which python

   # 如果必须使用系统环境（不推荐）
   pip install --break-system-packages -r requirements.txt
   ```

2. **音频设备问题**
   ```bash
   # 检查音频设备
   arecord -l  # 列出录音设备
   aplay -l    # 列出播放设备
   ```

3. **模型文件问题**
   - ModelScope会自动下载模型到~/models目录
   - 如果下载失败，检查网络连接和磁盘空间
   - 验证模型ID是否正确

4. **API连接问题**
   - 检查网络连接
   - 验证DashScope API密钥是否正确
   - 检查阿里云账户余额

5. **阿里云服务问题**
   - SenseVoice: 检查模型ID (iic/SenseVoiceSmall)
   - CosyVoice: 检查模型ID (iic/CosyVoice2-0.5B)
   - Qwen: 检查模型名称 (qwen-turbo, qwen-plus, etc.)

4. **性能问题**
   - 树莓派5有足够性能运行此应用
   - 如果遇到延迟，考虑使用更小的模型

### 日志调试

查看日志文件：
```bash
tail -f logs/assistant.log
```

调整日志级别：
```bash
# 在.env中设置
LOG_LEVEL=DEBUG
```

## 开发和扩展

### 添加新功能

1. **自定义关键词**: 修改 `KWS_KEYWORD` 环境变量
2. **更换TTS引擎**: 设置 `TTS_ENGINE=edge-tts`
3. **集成新模型**: 扩展 `core/llm/` 模块
4. **添加新命令**: 在控制器中添加处理逻辑

### 模块接口

每个核心模块都遵循标准接口：
- `initialize()`: 初始化模块
- `cleanup()`: 清理资源
- `is_initialized()`: 检查状态

## 许可证

[MIT License](LICENSE)

## 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - 优秀的语音处理工具
- [OpenAI](https://openai.com) - 大语言模型API
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) - Python音频库
