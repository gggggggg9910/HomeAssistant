"""
Text-to-speech implementation using Alibaba CosyVoice2-0.5B.
"""
import asyncio
import logging
import io
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import torch
    from modelscope import snapshot_download
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    MODELScope_AVAILABLE = True
except ImportError:
    MODELScope_AVAILABLE = False
    torch = None
    snapshot_download = None
    pipeline = None
    Tasks = None

logger = logging.getLogger(__name__)


class TTSConfig:
    """Configuration for Alibaba CosyVoice2-0.5B TTS."""
    def __init__(
        self,
        model_id: str = "iic/CosyVoice2-0.5B",
        model_path: str = "~/models/cosyvoice",
        voice: str = "中文女",  # Voice style for CosyVoice
        speed: float = 1.0,
        volume: float = 0.8,
        use_gpu: bool = False,
        sample_rate: int = 16000
    ):
        self.model_id = model_id
        self.model_path = Path(model_path).expanduser()
        self.voice = voice
        self.speed = speed
        self.volume = volume
        self.use_gpu = use_gpu
        self.sample_rate = sample_rate


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
            # Download model if not exists
            model_path = self.config.model_path
            if not model_path.exists():
                logger.info(f"Downloading CosyVoice2-0.5B model: {self.config.model_id}")
                model_path = snapshot_download(
                    self.config.model_id,
                    cache_dir=str(model_path.parent)
                )
                logger.info(f"Model downloaded to: {model_path}")

            # Set device
            device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"

            # Create TTS pipeline
            self.pipeline = pipeline(
                task=Tasks.text_to_speech,
                model=self.config.model_id,
                device=device,
                model_revision="v1.0.0"  # Specify version for stability
            )

            self._is_initialized = True
            logger.info(f"CosyVoice2-0.5B TTS initialized (device: {device})")
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
            # Prepare input for CosyVoice
            # CosyVoice expects specific input format
            input_data = {
                'text': text,
                'voice': self.voice
            }

            # Generate speech
            result = self.pipeline(input_data)

            # Extract audio from result
            if isinstance(result, dict) and 'output' in result:
                audio_data = result['output']
            elif hasattr(result, 'numpy'):  # If it's a tensor
                audio_data = result.numpy()
            elif isinstance(result, np.ndarray):
                audio_data = result
            else:
                logger.error(f"Unexpected CosyVoice result format: {type(result)}")
                return None

            # Ensure audio is in the right format (float32, normalized)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize to [-1, 1] range if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val

            # Apply volume
            audio_data = audio_data * self.config.volume

            logger.info(f"CosyVoice synthesized: '{text[:50]}...' -> {len(audio_data)} samples")
            return audio_data

        except Exception as e:
            logger.error(f"CosyVoice synthesis failed: {e}")
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
                audio_config = AudioConfig(sample_rate=22050)  # CosyVoice typically uses 22kHz
                audio_manager = AudioManager(audio_config)

                # Initialize and play
                if await audio_manager.initialize():
                    success = await audio_manager.speak(audio_data)
                    await audio_manager.cleanup()
                    return success
                else:
                    logger.error("Failed to initialize audio manager for CosyVoice playback")
                    return False
            else:
                logger.error("Failed to synthesize speech with CosyVoice")
                return False

        except Exception as e:
            logger.error(f"Failed to speak text with CosyVoice: {e}")
            return False

    def get_available_voices(self) -> list:
        """Get list of available CosyVoice voices."""
        if not self._is_initialized:
            return []

        # CosyVoice2-0.5B supports various Chinese voices
        return [
            "中文女",  # Chinese Female
            "中文男",  # Chinese Male
            "英文女",  # English Female
            "英文男",  # English Male
            "日语女",  # Japanese Female
            "韩语女",  # Korean Female
            "粤语女",  # Cantonese Female
        ]
