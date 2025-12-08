"""
PyAudio-based implementation of audio interfaces.
"""
import asyncio
import logging
import threading
import time
from typing import Optional, Callable, Any
import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

from .audio_interface import AudioConfig, AudioInputInterface, AudioOutputInterface

logger = logging.getLogger(__name__)


class PyAudioInputInterface(AudioInputInterface):
    """PyAudio-based audio input implementation."""

    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self.audio = None
        self.stream = None
        self._recording = False
        self._callback: Optional[Callable[[np.ndarray], None]] = None
        self._loop = None

    async def initialize(self) -> bool:
        """Initialize PyAudio input interface."""
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio not available. Please install with: pip install pyaudio")
            return False

        try:
            self.audio = pyaudio.PyAudio()
            self._loop = asyncio.get_event_loop()
            self._is_initialized = True
            logger.info("PyAudio input interface initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio input: {e}")
            return False

    async def cleanup(self):
        """Clean up PyAudio input resources."""
        await self.stop_recording()
        if self.stream:
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        self._is_initialized = False
        logger.info("PyAudio input interface cleaned up")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Internal callback for PyAudio stream."""
        if self._recording and self._callback:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)

            # Convert to float32 and normalize
            if self.config.dtype == 'float32':
                audio_data = audio_data.astype(np.float32) / 32768.0

            # Call user callback in the event loop
            if self._loop:
                self._loop.call_soon_threadsafe(self._callback, audio_data)

        return (in_data, pyaudio.paContinue)

    async def start_recording(self, callback: Callable[[np.ndarray], None]) -> bool:
        """Start continuous audio recording."""
        if not self._is_initialized:
            logger.error("PyAudio input not initialized")
            return False

        try:
            self._callback = callback

            # 尝试使用ALSA设备名称直接访问海康威视摄像头麦克风
            try:
                # 首先尝试使用设备索引（如果设置了的话）
                if self.config.input_device is not None:
                    self.stream = self.audio.open(
                        format=pyaudio.paInt16,
                        channels=self.config.channels,
                        rate=self.config.input_sample_rate,  # 使用输入采样率
                        input=True,
                        input_device_index=self.config.input_device,
                        frames_per_buffer=self.config.chunk_size,
                        stream_callback=self._audio_callback
                    )
                else:
                    # 如果没有设置设备索引，尝试直接使用ALSA设备名称
                    import pyaudio
                    host_api_info = self.audio.get_host_api_info_by_type(pyaudio.paALSA)
                    for i in range(host_api_info.get('deviceCount', 0)):
                        device_info = self.audio.get_device_info_by_host_api_device_index(host_api_info['index'], i)
                        if 'hw:2,0' in device_info.get('name', '').lower() and device_info.get('maxInputChannels', 0) > 0:
                            self.stream = self.audio.open(
                                format=pyaudio.paInt16,
                                channels=self.config.channels,
                                rate=self.config.input_sample_rate,  # 使用输入采样率
                                input=True,
                                input_device_index=device_info['index'],
                                frames_per_buffer=self.config.chunk_size,
                                stream_callback=self._audio_callback
                            )
                            break
                    else:
                        # 如果找不到匹配的设备，使用默认输入设备
                        self.stream = self.audio.open(
                            format=pyaudio.paInt16,
                            channels=self.config.channels,
                            rate=self.config.input_sample_rate,  # 使用输入采样率
                            input=True,
                            frames_per_buffer=self.config.chunk_size,
                            stream_callback=self._audio_callback
                        )
            except Exception as e:
                logger.warning(f"Direct device access failed, trying default: {e}")
                # 回退到默认设备
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=self.config.channels,
                    rate=self.config.input_sample_rate,  # 使用输入采样率
                    input=True,
                    frames_per_buffer=self.config.chunk_size,
                    stream_callback=self._audio_callback
                )

            self.stream.start_stream()
            self._recording = True
            logger.info("Started audio recording")
            return True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False

    async def stop_recording(self):
        """Stop audio recording."""
        self._recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        logger.info("Stopped audio recording")

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def is_initialized(self) -> bool:
        """Check if interface is initialized."""
        return self._is_initialized

    async def record_chunk(self, duration_seconds: float) -> Optional[np.ndarray]:
        """Record a single chunk of audio."""
        if not self._is_initialized:
            return None

        try:
            frames = []
            chunk_duration = self.config.chunk_size / self.config.input_sample_rate  # 使用输入采样率
            num_chunks = int(duration_seconds / chunk_duration)

            # Create a temporary stream for recording
            try:
                # 首先尝试使用设备索引（如果设置了的话）
                if self.config.input_device is not None:
                    stream = self.audio.open(
                        format=pyaudio.paInt16,
                        channels=self.config.channels,
                        rate=self.config.input_sample_rate,  # 使用输入采样率
                        input=True,
                        input_device_index=self.config.input_device,
                        frames_per_buffer=self.config.chunk_size
                    )
                else:
                    # 如果没有设置设备索引，尝试直接使用ALSA设备名称
                    import pyaudio
                    host_api_info = self.audio.get_host_api_info_by_type(pyaudio.paALSA)
                    for i in range(host_api_info.get('deviceCount', 0)):
                        device_info = self.audio.get_device_info_by_host_api_device_index(host_api_info['index'], i)
                        if 'hw:2,0' in device_info.get('name', '').lower() and device_info.get('maxInputChannels', 0) > 0:
                            stream = self.audio.open(
                                format=pyaudio.paInt16,
                                channels=self.config.channels,
                                rate=self.config.input_sample_rate,  # 使用输入采样率
                                input=True,
                                input_device_index=device_info['index'],
                                frames_per_buffer=self.config.chunk_size
                            )
                            break
                    else:
                        # 如果找不到匹配的设备，使用默认输入设备
                        stream = self.audio.open(
                            format=pyaudio.paInt16,
                            channels=self.config.channels,
                            rate=self.config.input_sample_rate,  # 使用输入采样率
                            input=True,
                            frames_per_buffer=self.config.chunk_size
                        )
            except Exception as e:
                logger.warning(f"Direct device access failed, trying default: {e}")
                # 回退到默认设备
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=self.config.channels,
                    rate=self.config.input_sample_rate,  # 使用输入采样率
                    input=True,
                    frames_per_buffer=self.config.chunk_size
                )

            for _ in range(num_chunks):
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                frames.append(data)

            stream.close()

            # Concatenate all frames
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

            # Convert to float32 if needed
            if self.config.dtype == 'float32':
                audio_data = audio_data.astype(np.float32) / 32768.0

            return audio_data

        except Exception as e:
            logger.error(f"Failed to record audio chunk: {e}")
            return None


class PyAudioOutputInterface(AudioOutputInterface):
    """PyAudio-based audio output implementation."""

    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self.audio = None
        self.stream = None
        self._playing = False
        self._stop_event = threading.Event()

    async def initialize(self) -> bool:
        """Initialize PyAudio output interface."""
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio not available. Please install with: pip install pyaudio")
            return False

        try:
            self.audio = pyaudio.PyAudio()
            self._is_initialized = True
            logger.info("PyAudio output interface initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio output: {e}")
            return False

    async def cleanup(self):
        """Clean up PyAudio output resources."""
        self.stop_playback()
        if self.stream:
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        self._is_initialized = False
        logger.info("PyAudio output interface cleaned up")

    def _play_audio_thread(self, audio_data: np.ndarray):
        """Thread function for playing audio."""
        try:
            # Convert float32 back to int16 if needed
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)

            # Ensure correct shape
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(-1, self.config.channels)

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.output_sample_rate,  # 使用输出采样率
                output=True,
                output_device_index=self.config.output_device
            )

            self._playing = True
            self._stop_event.clear()

            # Write audio data in chunks
            chunk_size = self.config.chunk_size
            for i in range(0, len(audio_data), chunk_size):
                if self._stop_event.is_set():
                    break
                chunk = audio_data[i:i + chunk_size]
                self.stream.write(chunk.tobytes())

            self.stream.close()
            self.stream = None

        except Exception as e:
            logger.error(f"Error playing audio: {e}")
        finally:
            self._playing = False

    async def play_audio(self, audio_data: np.ndarray) -> bool:
        """Play audio data synchronously."""
        if not self._is_initialized:
            logger.error("PyAudio output not initialized")
            return False

        try:
            # Create and start playback thread
            play_thread = threading.Thread(
                target=self._play_audio_thread,
                args=(audio_data,),
                daemon=True
            )
            play_thread.start()

            # Wait for playback to complete
            while self._playing and play_thread.is_alive():
                await asyncio.sleep(0.1)

            return True

        except Exception as e:
            logger.error(f"Failed to play audio synchronously: {e}")
            return False

    async def play_audio_async(self, audio_data: np.ndarray):
        """Play audio data asynchronously."""
        if not self._is_initialized:
            logger.error("PyAudio output not initialized")
            return

        try:
            # Start playback in background thread
            play_thread = threading.Thread(
                target=self._play_audio_thread,
                args=(audio_data,),
                daemon=True
            )
            play_thread.start()
            logger.debug("Started asynchronous audio playback")

        except Exception as e:
            logger.error(f"Failed to play audio asynchronously: {e}")

    def stop_playback(self):
        """Stop current playback."""
        self._stop_event.set()
        self._playing = False
        if self.stream:
            try:
                self.stream.stop_stream()
            except:
                pass
        logger.info("Audio playback stopped")

    def is_playing(self) -> bool:
        """Check if currently playing audio."""
        return self._playing

    def is_initialized(self) -> bool:
        """Check if interface is initialized."""
        return self._is_initialized
