"""
Speech recognition implementation using Alibaba SenseVoice.
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, List, Union
import numpy as np

from ..utils import PerformanceMonitor

try:
    import torch
    from modelscope import snapshot_download
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    MODELScope_AVAILABLE = True
except ImportError as e:
    MODELScope_AVAILABLE = False
    torch = None
    snapshot_download = None
    pipeline = None
    Tasks = None
    # Store error for later debugging
    _import_error = str(e)
except Exception as e:
    MODELScope_AVAILABLE = False
    torch = None
    snapshot_download = None
    pipeline = None
    Tasks = None
    _import_error = str(e)
else:
    _import_error = None

logger = logging.getLogger(__name__)


class ASRConfig:
    """Configuration for automatic speech recognition using SenseVoice."""
    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        model_path: str = "~/models/sensevoice",
        language: str = "zh",
        max_wait_seconds: int = 10,
        use_gpu: bool = False,
        sample_rate: int = 16000,
        disable_update: bool = True  # Disable model update checks for faster startup
    ):
        self.model_id = model_id
        self.model_path = Path(model_path).expanduser()
        self.language = language
        self.max_wait_seconds = max_wait_seconds
        self.use_gpu = use_gpu
        self.sample_rate = sample_rate
        self.disable_update = disable_update


class SpeechRecognizer:
    """Speech recognition using Alibaba SenseVoice."""

    def __init__(self, config: ASRConfig):
        self.config = config
        self.pipeline = None
        self._is_initialized = False
        self._is_recording = False
        self._audio_buffer = []
        self._buffer_lock = asyncio.Lock()
        self.asr_monitor = PerformanceMonitor("ASR")

    async def initialize(self) -> bool:
        """Initialize the SenseVoice speech recognizer."""
        # Declare global variables at the start to avoid "used prior to global declaration" error
        global torch, snapshot_download, pipeline, Tasks, MODELScope_AVAILABLE

        # Debug: Check current state and try re-importing if needed
        if not MODELScope_AVAILABLE:
            logger.warning(f"MODELScope_AVAILABLE is False, attempting to import modelscope again...")
            if '_import_error' in globals():
                logger.debug(f"Previous import error: {_import_error}")

            try:
                import torch as t
                from modelscope import snapshot_download as sd
                from modelscope.pipelines import pipeline as p
                from modelscope.utils.constant import Tasks as T

                # Update global variables
                torch = t
                snapshot_download = sd
                pipeline = p
                Tasks = T
                MODELScope_AVAILABLE = True

                logger.info("✅ modelscope imported successfully on retry!")
            except ImportError as e:
                logger.error(f"❌ modelscope import failed: {e}")
                logger.error("Common fixes:")
                logger.error("  pip install 'datasets<3.0.0'  # Downgrade datasets")
                logger.error("  pip install oss2 addict simplejson sortedcontainers  # Missing dependencies")
                import traceback
                logger.debug(traceback.format_exc())
                return False
            except Exception as e:
                logger.error(f"❌ Unexpected error importing modelscope: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return False

        if not MODELScope_AVAILABLE:
            logger.error("modelscope not available. Please install with: pip install modelscope torch")
            return False

        try:
            # Download model if not exists
            model_path = self.config.model_path
            if not model_path.exists():
                logger.info(f"Downloading SenseVoice model: {self.config.model_id}")
                model_path = snapshot_download(
                    self.config.model_id,
                    cache_dir=str(model_path.parent)
                )
                logger.info(f"Model downloaded to: {model_path}")

            # Set device
            device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"

            # Create ASR pipeline
            pipeline_kwargs = {
                'task': Tasks.auto_speech_recognition,
                'model': self.config.model_id,
                'device': device,
                'disable_update': self.config.disable_update
            }

            try:
                # Try with default revision first
                self.pipeline = pipeline(**pipeline_kwargs)
            except Exception as revision_error:
                logger.warning(f"Failed to load with default revision: {revision_error}")
                # Try with latest revision
                try:
                    pipeline_kwargs['model_revision'] = "master"  # Use master branch
                    self.pipeline = pipeline(**pipeline_kwargs)
                except Exception as master_error:
                    logger.warning(f"Failed to load with master revision: {master_error}")
                    # Try without revision parameter
                    del pipeline_kwargs['model_revision']
                    self.pipeline = pipeline(**pipeline_kwargs)

            self._is_initialized = True
            logger.info(f"SenseVoice speech recognizer initialized (device: {device})")
            return True

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialize SenseVoice recognizer: {e}")
            
            # Check if it's the audio extension error
            if "modelscope[audio]" in error_msg or "audio model import failed" in error_msg.lower() or "funasr" in error_msg.lower():
                logger.error("=" * 60)
                logger.error("SenseVoice requires FunASR audio extension!")
                logger.error("")
                logger.error("Try installing modelscope[audio] but skip pysptk:")
                logger.error("  pip install 'modelscope[audio]>=1.20.0' --no-deps")
                logger.error("  pip install funasr funasr-runtime")
                logger.error("  # Then manually install other audio dependencies (except pysptk)")
                logger.error("")
                logger.error("Or install SPTK first (if you need pysptk):")
                logger.error("  sudo apt-get install -y build-essential libfftw3-dev")
                logger.error("  wget http://download.sourceforge.net/sp-tk/SPTK-3.11.tar.gz")
                logger.error("  tar -xzf SPTK-3.11.tar.gz && cd SPTK-3.11")
                logger.error("  ./configure --prefix=/usr/local && make && sudo make install")
                logger.error("  pip install pysptk")
                logger.error("=" * 60)
            
            return False

    async def cleanup(self):
        """Clean up speech recognizer resources."""
        self._is_recording = False
        self.pipeline = None
        self._is_initialized = False
        async with self._buffer_lock:
            self._audio_buffer.clear()
        logger.info("SenseVoice speech recognizer cleaned up")

    def is_initialized(self) -> bool:
        """Check if speech recognizer is initialized."""
        return self._is_initialized

    async def recognize_speech(self, audio_data: Union[np.ndarray, str]) -> Optional[str]:
        """Recognize speech from audio data.

        Args:
            audio_data: Audio data as numpy array (float32, 16kHz) or audio file path

        Returns:
            Recognized text or None if recognition failed
        """
        if not self._is_initialized or not self.pipeline:
            logger.error("SenseVoice recognizer not initialized")
            return None

        self.asr_monitor.start()
        
        try:
            # Run inference
            if isinstance(audio_data, np.ndarray):
                with self.asr_monitor.stage("音频预处理"):
                    # Convert numpy array to the format expected by modelscope
                    # SenseVoice expects audio data in the correct format
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)

                    # Ensure proper shape (should be 1D for single channel)
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.flatten()

                    # Try different input formats for SenseVoice
                    # Some versions expect numpy array directly, others expect dict
                    try:
                        # First try direct numpy array
                        audio_input = audio_data
                    except Exception:
                        # If direct array fails, try dict format
                        audio_input = {
                            'array': audio_data,
                            'sampling_rate': 16000
                        }
                    
                    audio_duration = len(audio_data) / self.config.sample_rate
                    logger.info(f"[ASR] 音频长度: {audio_duration:.2f}秒, 采样数: {len(audio_data)}")
            else:
                # Assume it's a file path
                audio_input = audio_data

            # Run recognition
            with self.asr_monitor.stage("模型推理"):
                result = self.pipeline(audio_input)

            # Extract text from result
            with self.asr_monitor.stage("结果后处理"):
                if isinstance(result, dict) and 'text' in result:
                    text = result['text'].strip()
                elif isinstance(result, str):
                    text = result.strip()
                elif isinstance(result, list) and len(result) > 0:
                    # Sometimes result is a list of segments
                    text = ' '.join([seg.get('text', '') for seg in result if isinstance(seg, dict)]).strip()
                else:
                    text = str(result).strip()

                if text:
                    # Clean up SenseVoice special tokens
                    # Remove <|zh|>, <|NEUTRAL|>, <|Speech|>, etc.
                    import re
                    cleaned_text = re.sub(r'<\|[^>]+\|>', '', text).strip()
                    logger.info(f"SenseVoice recognized speech: '{text}' -> cleaned: '{cleaned_text}'")
                    final_text = cleaned_text if cleaned_text else text
                else:
                    logger.debug("SenseVoice: No speech recognized")
                    final_text = None

            self.asr_monitor.end()
            return final_text

        except Exception as e:
            self.asr_monitor.end()
            logger.error(f"Error during SenseVoice speech recognition: {e}")
            return None

    async def start_continuous_recognition(
        self,
        audio_callback: Optional[callable] = None,
        text_callback: Optional[callable] = None,
        timeout_seconds: Optional[float] = None
    ) -> Optional[str]:
        """Start continuous speech recognition using SenseVoice.

        For SenseVoice, we collect audio for the specified duration and then
        process it as a complete utterance.

        Args:
            audio_callback: Called with each audio chunk
            text_callback: Called with recognized text (not used for SenseVoice)
            timeout_seconds: Maximum recording time

        Returns:
            Final recognized text or None
        """
        if not self._is_initialized:
            logger.error("SenseVoice recognizer not initialized")
            return None

        timeout = timeout_seconds or self.config.max_wait_seconds
        self._is_recording = True
        start_time = asyncio.get_event_loop().time()

        # Collect all audio data during the recording period
        collected_audio = []

        logger.info(f"Starting SenseVoice continuous recognition (timeout: {timeout}s)")

        try:
            while self._is_recording:
                current_time = asyncio.get_event_loop().time()
                if current_time - start_time > timeout:
                    logger.info("SenseVoice recognition timeout")
                    break

                # Check for buffered audio
                async with self._buffer_lock:
                    # Collect all available audio chunks
                    while self._audio_buffer:
                        audio_chunk = self._audio_buffer.pop(0)
                        collected_audio.append(audio_chunk)

                        if audio_callback:
                            audio_callback(audio_chunk)

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.05)

            # Debug: Log collected audio status
            logger.info(f"Audio collection completed: collected_audio={len(collected_audio) if collected_audio else 0} chunks")

            # Process collected audio
            if collected_audio:
                # Concatenate all audio chunks
                full_audio = np.concatenate(collected_audio)
                logger.info(f"SenseVoice processing audio: {len(collected_audio)} chunks, "
                          f"total_samples={len(full_audio)}, duration={len(full_audio)/self.config.sample_rate:.2f}s, "
                          f"rms={np.sqrt(np.mean(full_audio**2)):.4f}")

                # Recognize the complete utterance
                text = await self.recognize_speech(full_audio)
                logger.info(f"SenseVoice result: '{text}' (None={text is None})")

                if text and text_callback:
                    text_callback(text)

                return text
            else:
                logger.warning("No audio chunks collected during recognition period!")
                return None

        except Exception as e:
            logger.error(f"Error during SenseVoice continuous recognition: {e}")
            return None
        finally:
            self._is_recording = False

    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to processing buffer."""
        # Add synchronously to avoid timing issues
        async def _add_chunk():
            async with self._buffer_lock:
                self._audio_buffer.append(audio_data.copy())
                # Keep buffer size reasonable for SenseVoice
                if len(self._audio_buffer) > 50:  # SenseVoice can handle longer audio
                    self._audio_buffer.pop(0)
                # Debug: Log buffer status occasionally
                if len(self._audio_buffer) % 20 == 0:
                    logger.debug(f"ASR buffer size: {len(self._audio_buffer)} chunks")

        # Add synchronously to avoid asyncio complications
        self._audio_buffer.append(audio_data.copy())
        # Keep buffer size reasonable for SenseVoice
        if len(self._audio_buffer) > 50:  # SenseVoice can handle longer audio
            self._audio_buffer.pop(0)
        # Debug: Log buffer status occasionally
        if len(self._audio_buffer) % 20 == 0:
            logger.debug(f"ASR buffer size: {len(self._audio_buffer)} chunks")

    def stop_recognition(self):
        """Stop continuous speech recognition."""
        self._is_recording = False
        logger.info("SenseVoice speech recognition stopped")

    def is_recognizing(self) -> bool:
        """Check if currently recognizing speech."""
        return self._is_recording
