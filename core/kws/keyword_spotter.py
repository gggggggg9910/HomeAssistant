"""
Keyword spotting implementation using sherpa-onnx-kws.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Callable, List
import numpy as np

# Try to import sherpa_onnx from custom path if not available in standard path
try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    try:
        # Try to load from ~/models/sherpa-onnx
        sherpa_path = Path.home() / "models" / "sherpa-onnx"
        if sherpa_path.exists():
            sys.path.insert(0, str(sherpa_path))
            sys.path.insert(0, str(sherpa_path / "lib"))
            sys.path.insert(0, str(sherpa_path / "python"))
            import sherpa_onnx
            SHERPA_AVAILABLE = True
        else:
            SHERPA_AVAILABLE = False
            sherpa_onnx = None
    except ImportError:
        SHERPA_AVAILABLE = False
        sherpa_onnx = None

from ..audio import AudioConfig

logger = logging.getLogger(__name__)


class KWSConfig:
    """Configuration for keyword spotting."""
    def __init__(
        self,
        model_path: str = "~/models/kws-onnx",
        keyword: str = "你好小助手",
        threshold: float = 0.5,
        max_wait_seconds: int = 30,
        sample_rate: int = 16000
    ):
        # Handle different path formats
        if model_path.startswith("~/"):
            self.model_path = Path(model_path).expanduser()
        elif model_path.startswith("/models/"):
            # Convert /models/ to ~/models/
            self.model_path = Path.home() / Path(model_path).relative_to(Path("/models"))
        else:
            self.model_path = Path(model_path).expanduser()

        self.keyword = keyword
        self.threshold = threshold
        self.max_wait_seconds = max_wait_seconds
        self.sample_rate = sample_rate


class KeywordSpotter:
    """Keyword spotting using sherpa-onnx-kws."""

    def __init__(self, config: KWSConfig):
        self.config = config
        self.kws = None
        self._is_initialized = False
        self._listening = False
        self._audio_buffer = []
        self._buffer_lock = asyncio.Lock()
        self._use_simple_mode = False

    async def initialize(self) -> bool:
        """Initialize the keyword spotter."""
        if SHERPA_AVAILABLE:
            try:
                # Check if model files exist
                model_path = self.config.model_path
                if not model_path.exists():
                    logger.warning(f"KWS model path does not exist: {model_path}, falling back to simple mode")
                    return await self._initialize_simple_mode()

                # Try to find the model files
                encoder_files = list(model_path.glob("*encoder*.onnx"))
                decoder_files = list(model_path.glob("*decoder*.onnx"))
                joiner_files = list(model_path.glob("*joiner*.onnx"))

                if not (encoder_files and decoder_files and joiner_files):
                    logger.warning(f"Required model files not found in {model_path}, falling back to simple mode")
                    return await self._initialize_simple_mode()

                # Create KWS config
                kws_config = sherpa_onnx.KeywordSpotterConfig(
                    encoder=encoder_files[0].as_posix(),
                    decoder=decoder_files[0].as_posix(),
                    joiner=joiner_files[0].as_posix(),
                    keywords=self.config.keyword,
                    num_threads=2,
                    provider="cpu",
                    max_active_paths=4
                )

                # Initialize keyword spotter
                self.kws = sherpa_onnx.KeywordSpotter(kws_config)
                self._is_initialized = True
                logger.info(f"Keyword spotter initialized with sherpa-onnx, keyword: '{self.config.keyword}'")
                return True

            except Exception as e:
                logger.warning(f"Failed to initialize sherpa-onnx keyword spotter: {e}, falling back to simple mode")
                return await self._initialize_simple_mode()
        else:
            logger.warning("sherpa-onnx not available, using simple keyword detection mode")
            return await self._initialize_simple_mode()

    async def _initialize_simple_mode(self) -> bool:
        """Initialize simple keyword detection mode (fallback)."""
        try:
            self.kws = None  # No sherpa-onnx instance
            self._is_initialized = True
            self._use_simple_mode = True
            logger.info(f"Simple keyword spotter initialized with keyword: '{self.config.keyword}'")
            logger.info("Note: This mode provides basic functionality. Install sherpa-onnx for better performance.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize simple keyword spotter: {e}")
            return False

    async def cleanup(self):
        """Clean up keyword spotter resources."""
        self._listening = False
        self.kws = None
        self._is_initialized = False
        async with self._buffer_lock:
            self._audio_buffer.clear()
        logger.info("Keyword spotter cleaned up")

    def is_initialized(self) -> bool:
        """Check if keyword spotter is initialized."""
        return self._is_initialized

    async def process_audio_chunk(self, audio_data: np.ndarray) -> Optional[str]:
        """Process an audio chunk for keyword detection.

        Args:
            audio_data: Audio data as numpy array (float32, normalized to [-1, 1])

        Returns:
            Detected keyword if found, None otherwise
        """
        if not self._is_initialized:
            return None

        try:
            if self._use_simple_mode:
                # Simple mode: detect loud sounds as potential keywords
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # sherpa-onnx expects 1D array
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()

                # Calculate RMS (Root Mean Square) as a simple volume indicator
                rms = np.sqrt(np.mean(audio_data ** 2))

                # If volume is above threshold, consider it a keyword detection
                # This is a very basic implementation - in real usage you'd want
                # to implement proper voice activity detection
                if rms > 0.1:  # Adjust threshold as needed
                    logger.info(f"Simple mode: Sound detected (RMS: {rms:.3f}), triggering keyword")
                    return self.config.keyword

                return None
            else:
                # Full sherpa-onnx mode
                if not self.kws:
                    return None

                # Ensure audio data is in the correct format
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # sherpa-onnx expects 1D array
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()

                # Process the audio chunk
                stream = self.kws.create_stream()
                stream.accept_waveform(self.config.sample_rate, audio_data)

                # Check for keyword
                while self.kws.is_ready(stream):
                    self.kws.decode_stream(stream)

                result = self.kws.get_result(stream)

                if result.keyword != "" and result.confidence >= self.config.threshold:
                    logger.info(f"Keyword detected: '{result.keyword}' (confidence: {result.confidence:.3f})")
                    return result.keyword

                return None

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None

    async def listen_for_keyword(
        self,
        audio_callback: Callable[[np.ndarray], None],
        keyword_callback: Callable[[str], None],
        timeout_seconds: Optional[float] = None
    ) -> bool:
        """Listen for keyword with callbacks.

        Args:
            audio_callback: Called with each audio chunk
            keyword_callback: Called when keyword is detected
            timeout_seconds: Maximum time to listen, None for config default

        Returns:
            True if keyword was detected, False if timeout or error
        """
        if not self._is_initialized:
            logger.error("Keyword spotter not initialized")
            return False

        timeout = timeout_seconds or self.config.max_wait_seconds
        self._listening = True
        start_time = asyncio.get_event_loop().time()

        logger.info(f"Starting keyword listening (timeout: {timeout}s)")

        try:
            while self._listening:
                current_time = asyncio.get_event_loop().time()
                if current_time - start_time > timeout:
                    logger.info("Keyword listening timeout")
                    return False

                # Wait for audio chunk (this would be called by audio interface)
                # In a real implementation, this would be integrated with the audio input
                await asyncio.sleep(0.1)  # Placeholder

                # Check if we have buffered audio to process
                async with self._buffer_lock:
                    if self._audio_buffer:
                        audio_chunk = self._audio_buffer.pop(0)

                        # Call audio callback
                        audio_callback(audio_chunk)

                        # Process for keyword
                        keyword = await self.process_audio_chunk(audio_chunk)
                        if keyword:
                            keyword_callback(keyword)
                            return True

            return False

        except Exception as e:
            logger.error(f"Error during keyword listening: {e}")
            return False
        finally:
            self._listening = False

    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to processing buffer (called by audio interface)."""
        async def _add_chunk():
            async with self._buffer_lock:
                self._audio_buffer.append(audio_data.copy())
                # Keep buffer size reasonable
                if len(self._audio_buffer) > 10:
                    self._audio_buffer.pop(0)

        # Schedule the addition
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_add_chunk())
        except RuntimeError:
            # No running loop, add synchronously (not ideal but works)
            self._audio_buffer.append(audio_data.copy())
            if len(self._audio_buffer) > 10:
                self._audio_buffer.pop(0)

    def stop_listening(self):
        """Stop keyword listening."""
        self._listening = False
        logger.info("Keyword listening stopped")

    def is_listening(self) -> bool:
        """Check if currently listening for keywords."""
        return self._listening
