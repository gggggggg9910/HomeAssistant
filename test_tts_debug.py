#!/usr/bin/env python3
"""
Debug script for TTS functionality
"""
import asyncio
import logging
import sys
import os

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from tts import TextToSpeech, TTSConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_tts():
    """Test TTS functionality with debug info."""
    try:
        logger.info("Testing TTS functionality...")

        # Create TTS config
        config = TTSConfig(
            model_id="damo/CosyVoice2-0.5B",
            voice="中文女",
            speed=1.0,
            volume=0.8
        )

        # Create TTS instance
        tts = TextToSpeech(config)

        # Initialize
        logger.info("Initializing TTS...")
        if not await tts.initialize():
            logger.error("Failed to initialize TTS")
            return False

        # Test text
        test_text = "没有听到您的语音，请重试"
        logger.info(f"Testing with text: '{test_text}' (type: {type(test_text)})")

        # Try to synthesize
        audio = await tts.synthesize_speech(test_text)
        if audio is not None:
            logger.info(f"✓ TTS synthesis successful, audio shape: {audio.shape}")
            return True
        else:
            logger.error("✗ TTS synthesis failed")
            return False

    except Exception as e:
        logger.error(f"TTS test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'tts' in locals():
            await tts.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_tts())
    sys.exit(0 if success else 1)
