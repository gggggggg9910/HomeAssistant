#!/usr/bin/env python3
"""
Test script to verify configuration and basic functionality.
"""
import asyncio
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings, initialize_logging


async def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        # Test audio
        from core.audio import AudioManager, AudioConfig
        print("✓ Audio imports OK")

        # Test KWS
        from core.kws import KeywordSpotter, KWSConfig
        print("✓ KWS imports OK")

        # Test SenseVoice ASR
        from core.asr import SpeechRecognizer, ASRConfig
        print("✓ ASR imports OK")

        # Test Qwen LLM
        from core.llm import LLMClient, LLMConfig
        print("✓ LLM imports OK")

        # Test CosyVoice TTS
        from core.tts import TextToSpeech, TTSConfig
        print("✓ TTS imports OK")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


async def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        print(f"✓ Audio sample rate: {settings.audio.sample_rate}")
        print(f"✓ KWS keyword: {settings.kws.keyword}")
        print(f"✓ ASR model: {settings.asr.model_id}")
        print(f"✓ LLM model: {settings.llm.model}")
        print(f"✓ TTS model: {settings.tts.model_id}")
        print(f"✓ TTS voice: {settings.tts.voice}")

        return True

    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


async def test_modelscope():
    """Test ModelScope availability."""
    print("\nTesting ModelScope...")

    try:
        import modelscope
        print(f"✓ ModelScope version: {modelscope.__version__}")

        from modelscope import snapshot_download
        print("✓ ModelScope snapshot_download available")

        return True

    except ImportError:
        print("✗ ModelScope not available - install with: pip install modelscope")
        return False


async def test_dashscope():
    """Test DashScope availability."""
    print("\nTesting DashScope...")

    try:
        import dashscope
        print("✓ DashScope available")

        if settings.llm.api_key:
            print("✓ DashScope API key configured")
        else:
            print("⚠ DashScope API key not configured - set DASHSCOPE_API_KEY in .env")

        return True

    except ImportError:
        print("✗ DashScope not available - install with: pip install dashscope")
        return False


async def main():
    """Run all tests."""
    print("Home Assistant Configuration Test")
    print("=" * 40)

    results = []

    # Test imports
    results.append(await test_imports())

    # Test config
    results.append(await test_config())

    # Test ModelScope
    results.append(await test_modelscope())

    # Test DashScope
    results.append(await test_dashscope())

    print("\n" + "=" * 40)
    print("Test Results:")

    if all(results):
        print("✓ All tests passed! Configuration looks good.")
        print("\nNext steps:")
        print("1. Set your DASHSCOPE_API_KEY in .env file")
        print("2. Run: python main.py")
    else:
        print("✗ Some tests failed. Please check dependencies and configuration.")

    return 0 if all(results) else 1


if __name__ == "__main__":
    initialize_logging()
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
