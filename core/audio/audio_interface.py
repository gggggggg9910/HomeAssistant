"""
Low-level audio input/output interfaces for voice assistant.
Provides abstraction over different audio backends (pyaudio, sounddevice).
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
import numpy as np

logger = logging.getLogger(__name__)


class AudioConfig:
    """Audio configuration parameters."""
    def __init__(
        self,
        sample_rate: int = 16000,          # 向后兼容
        input_sample_rate: int = 16000,    # 输入采样率
        output_sample_rate: int = 48000,   # 输出采样率
        channels: int = 1,
        chunk_size: int = 1024,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        dtype: str = 'float32',
        enable_input: bool = True,
        enable_output: bool = True
    ):
        # 为了向后兼容，如果没有指定input/output_sample_rate，使用sample_rate
        self.sample_rate = sample_rate
        self.input_sample_rate = input_sample_rate or sample_rate
        self.output_sample_rate = output_sample_rate or sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.input_device = input_device
        self.output_device = output_device
        self.dtype = dtype
        self.enable_input = enable_input
        self.enable_output = enable_output


class AudioInterface(ABC):
    """Abstract base class for audio interfaces."""

    @abstractmethod
    def __init__(self, config: AudioConfig):
        self.config = config
        self._is_initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the audio interface."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Clean up audio resources."""
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if interface is initialized."""
        return self._is_initialized


class AudioInputInterface(AudioInterface):
    """Abstract audio input interface."""

    @abstractmethod
    async def start_recording(self, callback: Callable[[np.ndarray], None]) -> bool:
        """Start audio recording with callback for audio chunks."""
        pass

    @abstractmethod
    async def stop_recording(self):
        """Stop audio recording."""
        pass

    @abstractmethod
    def is_recording(self) -> bool:
        """Check if currently recording."""
        pass

    @abstractmethod
    async def record_chunk(self, duration_seconds: float) -> Optional[np.ndarray]:
        """Record a single chunk of audio for specified duration."""
        pass


class AudioOutputInterface(AudioInterface):
    """Abstract audio output interface."""

    @abstractmethod
    async def play_audio(self, audio_data: np.ndarray) -> bool:
        """Play audio data synchronously."""
        pass

    @abstractmethod
    async def play_audio_async(self, audio_data: np.ndarray):
        """Play audio data asynchronously."""
        pass

    @abstractmethod
    def stop_playback(self):
        """Stop current playback."""
        pass

    @abstractmethod
    def is_playing(self) -> bool:
        """Check if currently playing audio."""
        pass


class AudioManager:
    """High-level audio manager coordinating input and output."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.input_interface: Optional[AudioInputInterface] = None
        self.output_interface: Optional[AudioOutputInterface] = None
        self._is_initialized = False

    async def initialize(self, input_interface_cls=None, output_interface_cls=None) -> bool:
        """Initialize audio interfaces."""
        try:
            # Import here to avoid circular imports
            if input_interface_cls is None:
                from .pyaudio_interface import PyAudioInputInterface
                input_interface_cls = PyAudioInputInterface

            if output_interface_cls is None:
                from .pyaudio_interface import PyAudioOutputInterface
                output_interface_cls = PyAudioOutputInterface

            input_ok = False
            output_ok = False

            # Initialize input interface (if enabled)
            if getattr(self.config, 'enable_input', True):
                try:
                    self.input_interface = input_interface_cls(self.config)
                    input_ok = await self.input_interface.initialize()
                    if input_ok:
                        logger.info("Audio input interface initialized successfully")
                    else:
                        logger.warning("Audio input interface initialization failed - voice input will be disabled")
                except Exception as e:
                    logger.warning(f"Audio input interface initialization failed: {e} - voice input will be disabled")
                    self.input_interface = None
            else:
                logger.info("Audio input disabled by configuration")
                self.input_interface = None

            # Initialize output interface (if enabled)
            if getattr(self.config, 'enable_output', True):
                try:
                    self.output_interface = output_interface_cls(self.config)
                    output_ok = await self.output_interface.initialize()
                    if output_ok:
                        logger.info("Audio output interface initialized successfully")
                    else:
                        logger.warning("Audio output interface initialization failed - voice output will be disabled")
                except Exception as e:
                    logger.warning(f"Audio output interface initialization failed: {e} - voice output will be disabled")
                    self.output_interface = None
            else:
                logger.info("Audio output disabled by configuration")
                self.output_interface = None

            # Allow initialization if at least one interface works
            self._is_initialized = input_ok or output_ok
            if self._is_initialized:
                logger.info("Audio manager initialized (some features may be disabled)")
                if not input_ok:
                    logger.warning("Voice input (microphone) is not available")
                if not output_ok:
                    logger.warning("Voice output (speakers) is not available")
            else:
                logger.error("Failed to initialize any audio interfaces")

            return self._is_initialized

        except Exception as e:
            logger.error(f"Failed to initialize audio manager: {e}")
            return False

    async def cleanup(self):
        """Clean up all audio resources."""
        if self.input_interface:
            await self.input_interface.cleanup()
        if self.output_interface:
            await self.output_interface.cleanup()
        self._is_initialized = False
        logger.info("Audio manager cleaned up")

    def is_initialized(self) -> bool:
        """Check if audio manager is initialized."""
        return self._is_initialized

    async def start_listening(self, callback: Callable[[np.ndarray], None]) -> bool:
        """Start continuous audio listening."""
        if not self.input_interface or not self.input_interface.is_initialized():
            logger.error("Audio input interface not initialized")
            return False

        return await self.input_interface.start_recording(callback)

    async def stop_listening(self):
        """Stop continuous audio listening."""
        if self.input_interface:
            await self.input_interface.stop_recording()

    async def speak(self, audio_data: np.ndarray, async_play: bool = False) -> bool:
        """Play audio through output interface."""
        if not self.output_interface or not self.output_interface.is_initialized():
            logger.error("Audio output interface not initialized")
            return False

        if async_play:
            await self.output_interface.play_audio_async(audio_data)
            return True
        else:
            return await self.output_interface.play_audio(audio_data)

    async def record_once(self, duration_seconds: float) -> Optional[np.ndarray]:
        """Record a single audio chunk."""
        if not self.input_interface or not self.input_interface.is_initialized():
            logger.error("Audio input interface not initialized")
            return None

        return await self.input_interface.record_chunk(duration_seconds)
