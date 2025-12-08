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
        """Convert text to speech audio using pyttsx3.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as numpy array or None if failed
        """
        if not self._is_initialized or not self.engine:
            logger.error("pyttsx3 TTS not initialized")
            return None

        try:
            import tempfile
            import wave
            import subprocess
            import os

            logger.debug(f"TTS synthesizing text: '{text}'")

            # Try using espeak directly to generate WAV file
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Use espeak directly instead of pyttsx3
                logger.debug(f"Using espeak to generate WAV file: {temp_path}")
                cmd = [
                    'espeak-ng',
                    '-v', 'zh',  # Chinese voice
                    '-s', '180',  # Speed
                    '-a', str(int(self.config.volume * 100)),  # Amplitude (volume)
                    '-w', temp_path,  # Output file
                    text
                ]

                logger.debug(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if result.returncode != 0:
                    logger.error(f"espeak-ng failed: {result.stderr}")
                    return None

                # Check if file was created and has content
                if not os.path.exists(temp_path):
                    logger.error("espeak-ng did not create WAV file")
                    return None

                file_size = os.path.getsize(temp_path)
                logger.debug(f"espeak-ng WAV file created, size: {file_size} bytes")

                if file_size == 0:
                    logger.error("espeak-ng generated empty WAV file")
                    return None

                # Read the WAV file and convert to numpy array
                logger.debug("Reading espeak-ng WAV file...")
                with wave.open(temp_path, 'rb') as wf:
                    channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    frame_rate = wf.getframerate()
                    n_frames = wf.getnframes()

                    logger.debug(f"WAV info: channels={channels}, sample_width={sample_width}, "
                               f"frame_rate={frame_rate}, n_frames={n_frames}")

                    # Read all frames
                    frames = wf.readframes(n_frames)
                    logger.debug(f"Read {len(frames)} bytes from WAV file")

                    # Convert to numpy array
                    if sample_width == 2:  # 16-bit
                        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    else:
                        logger.error(f"Unsupported sample width: {sample_width}")
                        return None

                    # If stereo, convert to mono
                    if channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                    elif channels != 1:
                        logger.error(f"Unsupported number of channels: {channels}")
                        return None

                    # Normalize to [-1, 1]
                    audio_data = audio_data / 32768.0

                # Apply volume
                audio_data = audio_data * self.config.volume

                logger.debug(f"TTS synthesis successful: shape={audio_data.shape}, dtype={audio_data.dtype}")
                return audio_data

            finally:
                # Clean up temp file
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        logger.debug(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

        except Exception as e:
            logger.error(f"Error during speech synthesis: {e}")
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
