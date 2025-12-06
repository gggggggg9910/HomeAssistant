"""
Text-to-speech implementation supporting multiple TTS engines.
"""
import asyncio
import logging
import io
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class TTSConfig:
    """Configuration for text-to-speech."""
    def __init__(
        self,
        engine: str = "pyttsx3",  # "pyttsx3" or "edge-tts"
        voice: str = "zh-CN-XiaoxiaoNeural",  # for edge-tts
        speed: float = 1.0,
        volume: float = 0.8,
        sample_rate: int = 16000
    ):
        self.engine = engine
        self.voice = voice
        self.speed = speed
        self.volume = volume
        self.sample_rate = sample_rate


class TextToSpeech:
    """Text-to-speech using various engines."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.engine = None
        self._is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the TTS engine."""
        try:
            if self.config.engine == "pyttsx3":
                return await self._init_pyttsx3()
            elif self.config.engine == "edge-tts":
                return await self._init_edge_tts()
            else:
                logger.error(f"Unsupported TTS engine: {self.config.engine}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            return False

    async def _init_pyttsx3(self) -> bool:
        """Initialize pyttsx3 engine."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', int(200 * self.config.speed))  # Default is 200
            self.engine.setProperty('volume', self.config.volume)

            # Try to set Chinese voice if available
            voices = self.engine.getProperty('voices')
            chinese_voice = None
            for voice in voices:
                if 'Chinese' in voice.name or 'zh' in voice.name.lower():
                    chinese_voice = voice
                    break

            if chinese_voice:
                self.engine.setProperty('voice', chinese_voice.id)
                logger.info(f"Using Chinese voice: {chinese_voice.name}")

            self._is_initialized = True
            logger.info("pyttsx3 TTS engine initialized")
            return True
        except ImportError:
            logger.error("pyttsx3 not available. Please install with: pip install pyttsx3")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            return False

    async def _init_edge_tts(self) -> bool:
        """Initialize edge-tts engine."""
        try:
            import edge_tts
            # edge-tts doesn't need initialization, just check availability
            self._is_initialized = True
            logger.info("edge-tts TTS engine initialized")
            return True
        except ImportError:
            logger.error("edge-tts not available. Please install with: pip install edge-tts")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize edge-tts: {e}")
            return False

    async def cleanup(self):
        """Clean up TTS resources."""
        if self.engine and self.config.engine == "pyttsx3":
            try:
                self.engine.stop()
            except:
                pass
        self.engine = None
        self._is_initialized = False
        logger.info("TTS engine cleaned up")

    def is_initialized(self) -> bool:
        """Check if TTS is initialized."""
        return self._is_initialized

    async def synthesize_speech(self, text: str) -> Optional[np.ndarray]:
        """Convert text to speech audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as numpy array or None if failed
        """
        if not self._is_initialized:
            logger.error("TTS engine not initialized")
            return None

        try:
            if self.config.engine == "pyttsx3":
                return await self._synthesize_pyttsx3(text)
            elif self.config.engine == "edge-tts":
                return await self._synthesize_edge_tts(text)
            else:
                logger.error(f"Unsupported TTS engine: {self.config.engine}")
                return None
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            return None

    async def _synthesize_pyttsx3(self, text: str) -> Optional[np.ndarray]:
        """Synthesize speech using pyttsx3."""
        try:
            # pyttsx3 works synchronously, so we run it in a thread
            import threading

            audio_data = None
            error = None

            def _synthesize():
                nonlocal audio_data, error
                try:
                    # Save to temporary buffer
                    buffer = io.BytesIO()
                    self.engine.save_to_file(text, buffer)
                    self.engine.runAndWait()

                    # This is a simplified approach - pyttsx3 doesn't easily give raw audio
                    # In practice, you might need to save to a temporary file and read it back
                    logger.warning("pyttsx3 synthesis completed (file-based approach needed)")
                    return None

                except Exception as e:
                    error = e

            # Run synthesis in thread
            synthesize_thread = threading.Thread(target=_synthesize)
            synthesize_thread.start()
            synthesize_thread.join(timeout=10)  # 10 second timeout

            if error:
                raise error

            # For now, return None as pyttsx3 doesn't easily provide raw audio data
            # In a real implementation, you'd save to a temp file and read it back
            logger.warning("pyttsx3 synthesis not fully implemented - use edge-tts for better results")
            return None

        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return None

    async def _synthesize_edge_tts(self, text: str) -> Optional[np.ndarray]:
        """Synthesize speech using edge-tts."""
        try:
            import edge_tts
            import io

            # Create edge-tts communicate object
            communicate = edge_tts.Communicate(text, self.config.voice)

            # Collect audio data
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            if not audio_chunks:
                logger.error("No audio data received from edge-tts")
                return None

            # Combine audio chunks
            audio_data = b''.join(audio_chunks)

            # Convert to numpy array (assuming WAV format)
            # This is a simplified conversion - real implementation would need proper WAV parsing
            try:
                import wave
                wav_buffer = io.BytesIO(audio_data)
                with wave.open(wav_buffer, 'rb') as wav_file:
                    # Read audio data
                    raw_audio = wav_file.readframes(wav_file.getnframes())
                    sample_width = wav_file.getsampwidth()
                    num_channels = wav_file.getnchannels()
                    frame_rate = wav_file.getframerate()

                    # Convert to numpy array
                    if sample_width == 2:  # 16-bit
                        audio_array = np.frombuffer(raw_audio, dtype=np.int16)
                    elif sample_width == 4:  # 32-bit
                        audio_array = np.frombuffer(raw_audio, dtype=np.int32)
                    else:
                        logger.error(f"Unsupported sample width: {sample_width}")
                        return None

                    # Convert to float32 and normalize
                    if audio_array.dtype == np.int16:
                        audio_array = audio_array.astype(np.float32) / 32768.0
                    elif audio_array.dtype == np.int32:
                        audio_array = audio_array.astype(np.float32) / 2147483648.0

                    # Resample if necessary
                    if frame_rate != self.config.sample_rate:
                        logger.warning(f"Sample rate mismatch: {frame_rate} vs {self.config.sample_rate}")
                        # In a real implementation, you'd resample here
                        # For now, we'll proceed with the original rate

                    # Convert to mono if stereo
                    if num_channels == 2:
                        audio_array = audio_array.reshape(-1, 2).mean(axis=1)

                    logger.info(f"Synthesized speech: {len(audio_array)} samples")
                    return audio_array

            except Exception as e:
                logger.error(f"Failed to parse WAV data: {e}")
                return None

        except Exception as e:
            logger.error(f"edge-tts synthesis failed: {e}")
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
                audio_config = AudioConfig(sample_rate=self.config.sample_rate)
                audio_manager = AudioManager(audio_config)

                # Initialize and play
                if await audio_manager.initialize():
                    success = await audio_manager.speak(audio_data)
                    await audio_manager.cleanup()
                    return success
                else:
                    logger.error("Failed to initialize audio manager for playback")
                    return False
            else:
                logger.error("Failed to synthesize speech")
                return False

        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
            return False

    def get_available_voices(self) -> list:
        """Get list of available voices."""
        if not self._is_initialized:
            return []

        try:
            if self.config.engine == "pyttsx3" and self.engine:
                voices = self.engine.getProperty('voices')
                return [voice.name for voice in voices]
            elif self.config.engine == "edge-tts":
                # edge-tts has many voices, return some common Chinese ones
                return [
                    "zh-CN-XiaoxiaoNeural",
                    "zh-CN-XiaoyiNeural",
                    "zh-CN-YunjianNeural",
                    "zh-CN-YunxiNeural",
                    "zh-CN-YunxiaNeural",
                    "zh-CN-YunyangNeural"
                ]
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return []
