"""
Text-to-speech implementation using Alibaba CosyVoice2-0.5B.
"""
import asyncio
import logging
from typing import Optional
import numpy as np

try:
    from modelscope.pipelines import pipeline
    MODELScope_AVAILABLE = True
except ImportError:
    MODELScope_AVAILABLE = False
    pipeline = None

logger = logging.getLogger(__name__)


class TTSConfig:
    """Configuration for Alibaba CosyVoice2-0.5B TTS."""
    def __init__(
        self,
        model_id: str = "damo/CosyVoice2-0.5B",
        voice: str = "中文女",
        speed: float = 1.0,
        volume: float = 0.8
    ):
        self.model_id = model_id
        self.voice = voice
        self.speed = speed
        self.volume = volume


class TextToSpeech:
    """Text-to-speech using Alibaba CosyVoice2-0.5B."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.pipeline = None
        self._is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the CosyVoice TTS engine."""
        if not MODELScope_AVAILABLE:
            logger.error("modelscope not available. Please install with: pip install modelscope")
            return False

        try:
            # Direct pipeline initialization with ModelScope ID
            self.pipeline = pipeline('text-to-speech', model=self.config.model_id)
            self._is_initialized = True
            logger.info(f"CosyVoice2-0.5B TTS initialized with model: {self.config.model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize CosyVoice TTS: {e}")
            return False

    async def cleanup(self):
        """Clean up TTS resources."""
        self.pipeline = None
        self._is_initialized = False
        logger.info("CosyVoice TTS cleaned up")

    def is_initialized(self) -> bool:
        """Check if TTS is initialized."""
        return self._is_initialized

    async def synthesize_speech(self, text: str) -> Optional[np.ndarray]:
        """Convert text to speech audio using CosyVoice.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as numpy array or None if failed
        """
        if not self._is_initialized or not self.pipeline:
            logger.error("CosyVoice TTS not initialized")
            return None

        try:
            # Prepare input for CosyVoice2-0.5B
            input_data = {'text': text}

            # Generate speech
            result = self.pipeline(input_data)

            # Extract audio from result
            if isinstance(result, dict) and 'output' in result:
                audio_data = result['output']
            elif hasattr(result, 'numpy'):
                audio_data = result.numpy()
            elif isinstance(result, np.ndarray):
                audio_data = result
            else:
                logger.error(f"Unexpected result format: {type(result)}")
                return None

            # Ensure proper format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val

            # Apply volume
            audio_data = audio_data * self.config.volume

            logger.info(f"Synthesized: '{text[:50]}...' -> {len(audio_data)} samples")
            return audio_data

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return None

    async def speak_text(self, text: str) -> bool:
        """Convert text to speech and play it immediately.

        Args:
            text: Text to speak

        Returns:
            True if successful, False otherwise
        """
        try:
            audio_data = await self.synthesize_speech(text)
            if audio_data is not None:
                # Import here to avoid circular imports
                from ..audio import AudioManager, AudioConfig

                # Create audio manager for playback
                audio_config = AudioConfig(sample_rate=22050)
                audio_manager = AudioManager(audio_config)

                # Initialize and play
                if await audio_manager.initialize():
                    success = await audio_manager.speak(audio_data)
                    await audio_manager.cleanup()
                    return success
                else:
                    logger.error("Failed to initialize audio manager")
                    return False
            else:
                logger.error("Failed to synthesize speech")
                return False

        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
            return False

    def get_available_voices(self) -> list:
        """Get list of available CosyVoice voices."""
        if not self._is_initialized:
            return []

        return [
            "中文女", "中文男", "英文女", "英文男",
            "日语女", "韩语女", "粤语女"
        ]
