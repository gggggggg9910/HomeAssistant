#!/usr/bin/env python3
"""
Debug script for testing SenseVoice speech recognition with Hikvision device.
"""

import asyncio
import logging
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    logger.error("PyAudio not available")
    PYAUDIO_AVAILABLE = False

async def test_sensevoice_recognition():
    """Test SenseVoice recognition with recorded audio."""

    if not PYAUDIO_AVAILABLE:
        return

    # Import after availability check
    from core.asr.speech_recognizer import SpeechRecognizer, ASRConfig
    from core.audio.audio_interface import AudioConfig
    from core.audio.pyaudio_interface import PyAudioInputInterface

    # Initialize ASR
    asr_config = ASRConfig(
        model_path="~/models/sensevoice",
        language="zh",
        sample_rate=16000,
        disable_update=True
    )

    asr = SpeechRecognizer(asr_config)
    if not await asr.initialize():
        logger.error("Failed to initialize ASR")
        return

    # Initialize audio input
    audio_config = AudioConfig(
        sample_rate=16000,
        channels=1,
        chunk_size=1024,
        input_device=0,  # Hikvision device
        input_sample_rate=16000,
        output_sample_rate=48000
    )

    audio_input = PyAudioInputInterface(audio_config)
    if not await audio_input.initialize():
        logger.error("Failed to initialize audio input")
        return

    print("🎤 请说话（5秒）...")

    # Record 5 seconds of audio
    audio_data = await audio_input.record_chunk(5.0)

    if audio_data is None or len(audio_data) == 0:
        print("❌ 没有录制到音频数据")
        return

    print(f"📊 录制了 {len(audio_data)} 个采样点，RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")

    # Test recognition
    print("🔍 正在识别语音...")
    start_time = time.time()
    result = await asr.recognize_speech(audio_data)
    end_time = time.time()

    print(".2f"
    if result:
        print(f"✅ 识别结果: '{result}'")
    else:
        print("❌ 未能识别语音")

    # Cleanup
    await audio_input.cleanup()

if __name__ == "__main__":
    print("🧪 SenseVoice ASR 调试测试")
    print("=" * 50)
    asyncio.run(test_sensevoice_recognition())
