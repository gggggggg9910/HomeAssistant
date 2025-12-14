# Piper TTS 安装指南 (树莓派5优化版)

Piper TTS 是一个快速、轻量级的本地神经网络语音合成系统，专门针对树莓派等嵌入式设备优化。

## 系统要求

- 树莓派5 (8GB 推荐，4GB 可用)
- Raspberry Pi OS (64-bit)
- Python 3.9+

## 安装步骤

### 1. 安装 Piper TTS

```bash
# 安装 piper-tts
pip install piper-tts

# 或者使用 pip3
pip3 install piper-tts
```

### 2. 创建模型目录

```bash
mkdir -p ~/models/piper
cd ~/models/piper
```

### 3. 下载中文语音模型

**方法一：直接下载（推荐）**

```bash
# 下载华妍中文女声模型（中等质量，约50MB）
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json
```

**方法二：使用 curl（如果 wget 不可用）**

```bash
curl -L -o zh_CN-huayan-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx

curl -L -o zh_CN-huayan-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json
```

**方法三：使用国内镜像（如果 HuggingFace 访问慢）**

```bash
# 使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 然后使用方法一的命令
wget https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx
wget https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json
```

### 4. 验证安装

```bash
# 测试 piper 命令
echo "你好，这是一个测试。" | piper --model ~/models/piper/zh_CN-huayan-medium.onnx --output_file test.wav

# 播放测试
aplay test.wav
```

### 5. 运行测试脚本

```bash
cd ~/HomeAssistant
python test_piper_tts.py
```

## 可用的中文语音模型

| 模型名称 | 说明 | 质量 | 大小 |
|---------|------|------|------|
| zh_CN-huayan-medium | 华妍女声 | 中等 | ~50MB |
| zh_CN-huayan-x_low | 华妍女声 | 较低 | ~20MB |

更多语音模型请访问：https://github.com/rhasspy/piper/blob/master/VOICES.md

## 多角色支持

要添加更多角色，只需下载更多模型文件到 `~/models/piper/` 目录，然后在代码中切换：

```python
from core.tts import PiperTTSConfig, TextToSpeechPiper

config = PiperTTSConfig(
    model_dir="~/models/piper",
    voice="zh_CN_huayan_medium",  # 默认声音
)

tts = TextToSpeechPiper(config)
await tts.initialize()

# 切换声音
await tts.load_voice("another_voice_id")
tts.set_voice("another_voice_id")
```

## 性能优化

Piper TTS 在树莓派5上的典型性能：

| 文本长度 | 合成时间 | RTF |
|---------|---------|-----|
| 10字 | ~0.5秒 | <0.5 |
| 50字 | ~2秒 | <0.5 |
| 100字 | ~4秒 | <0.5 |

RTF (Real-Time Factor) < 1 表示合成速度比播放速度快。

### 进一步优化

```python
# 在 PiperTTSConfig 中调整参数
config = PiperTTSConfig(
    length_scale=0.9,      # 稍快的语速
    sentence_silence=0.1,  # 减少句子间停顿
    num_threads=4,         # 使用所有4个CPU核心
)
```

## 常见问题

### Q: 提示 "piper: command not found"

```bash
# 确保 pip 安装的命令在 PATH 中
export PATH="$HOME/.local/bin:$PATH"

# 或者使用 python 模块方式
python -m piper --help
```

### Q: 下载模型很慢

使用国内镜像或手动下载后上传到树莓派。

### Q: 播放没有声音

```bash
# 检查音频设备
aplay -l

# 尝试指定设备
aplay -D default test.wav
aplay -D hw:0,0 test.wav
```

### Q: 内存不足

Piper TTS 内存占用很低（约200MB），如果仍有问题：

```bash
# 增加 swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # 设置 CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## 依赖列表

```
piper-tts>=1.2.0
numpy>=1.21.0
```

## 相关链接

- Piper TTS GitHub: https://github.com/rhasspy/piper
- 语音模型列表: https://github.com/rhasspy/piper/blob/master/VOICES.md
- HuggingFace 模型: https://huggingface.co/rhasspy/piper-voices
