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

# Try to import pyttsx3 as fallback TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None

logger = logging.getLogger(__name__)


class TTSConfig:
    """Configuration for Alibaba CosyVoice2-0.5B TTS."""
    def __init__(
        self,
        model_id: str = "iic/CosyVoice2-0.5B",
        model_path: Optional[str] = None,
        voice: str = "中文女",
        speed: float = 1.0,
        volume: float = 0.8
    ):
        self.model_id = model_id
        self.model_path = model_path  # Local model path, e.g., ~/models/iic/CosyVoice2-0.5B/
        self.voice = voice
        self.speed = speed
        self.volume = volume


class TextToSpeech:
    """Text-to-speech using pyttsx3."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.engine = None  # pyttsx3 engine
        self._is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the pyttsx3 TTS engine."""
        if not PYTTSX3_AVAILABLE:
            logger.error("pyttsx3 not available. Please install with: pip install pyttsx3")
            return False

        try:
            logger.info("Initializing pyttsx3 TTS engine")
            self.engine = pyttsx3.init()
            self._is_initialized = True
            logger.info("pyttsx3 TTS initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 TTS: {e}")
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
            # Debug logging
            logger.info(f"TTS synthesize_speech called with text: '{text}' (type: {type(text)})")

            # Prepare input for CosyVoice2-0.5B
            # CosyVoice2-0.5B expects text directly, not in a dictionary
            logger.info(f"TTS input_data: '{text}'")

            # Generate speech - try different input formats for CosyVoice
            logger.debug(f"TTS calling CosyVoice pipeline with text: '{text}'")

            # Try different input formats
            input_data = {'text': text}
            logger.debug(f"TTS trying input format: {input_data}")

            try:
                result = self.pipeline(input_data)
                logger.debug(f"TTS pipeline returned result of type: {type(result)}")
            except Exception as e:
                logger.warning(f"Dict input failed: {e}, trying direct text input")
                try:
                    result = self.pipeline(text)
                    logger.debug(f"TTS direct text input succeeded, result type: {type(result)}")
                except Exception as e2:
                    logger.error(f"Both input formats failed: {e2}")
                    return None

            # Extract audio from result
            logger.debug(f"TTS result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            if isinstance(result, dict):
                if 'output' in result:
                    audio_data = result['output']
                elif 'audio' in result:
                    audio_data = result['audio']
                elif 'wav' in result:
                    audio_data = result['wav']
                else:
                    logger.error(f"Unexpected dict result format, keys: {list(result.keys())}")
                    return None
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
            logger.info(f"TTS speak_text called with: '{text}'")
            audio_data = await self.synthesize_speech(text)
            logger.info(f"TTS synthesis result: audio_data is {'not ' if audio_data is None else ''}None")
            if audio_data is not None:
                # Import here to avoid circular imports
                from ..audio import AudioManager, AudioConfig

                # Create audio manager for playback
                audio_config = AudioConfig(
                    sample_rate=22050,
                    input_sample_rate=16000,  # For Hikvision device
                    output_sample_rate=48000, # For Hikvision device
                    input_device=0,           # Use device 0
                    output_device=0           # Use device 0
                )
                audio_manager = AudioManager(audio_config)

                # Initialize and play
                if await audio_manager.initialize():
                    logger.info(f"TTS playing audio: shape={audio_data.shape}, dtype={audio_data.dtype}, sample_rate=22050")
                    success = await audio_manager.speak(audio_data)
                    logger.info(f"TTS audio playback result: {success}")
                    await audio_manager.cleanup()
                    return success
                else:
                    logger.error("Failed to initialize audio manager for TTS playback")
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
