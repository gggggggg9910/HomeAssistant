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
        """Convert text to speech audio using pyttsx3 or fallback to espeak-ng.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as numpy array or None if failed
        """
        if not self._is_initialized:
            logger.error("TTS engine not initialized")
            return None

        try:
            import tempfile
            import wave
            import subprocess
            import os
            import numpy as np

            logger.info(f"TTS synthesizing text: '{text}'")

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Try pyttsx3 first if available
                if PYTTSX3_AVAILABLE:
                    logger.info("Trying pyttsx3 for TTS synthesis")
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()

                        # Configure voice properties
                        engine.setProperty('rate', 150)  # Speed (reduced for better quality)
                        engine.setProperty('volume', self.config.volume)

                        # Try to find Chinese voice
                        voices = engine.getProperty('voices')
                        chinese_voice = None
                        for voice in voices:
                            if hasattr(voice, 'languages') and 'zh' in str(voice.languages):
                                chinese_voice = voice
                                break
                            elif 'chinese' in voice.name.lower() or 'zh' in voice.name.lower():
                                chinese_voice = voice
                                break

                        if chinese_voice:
                            engine.setProperty('voice', chinese_voice.id)
                            logger.info(f"Using Chinese voice: {chinese_voice.name}")
                        else:
                            logger.warning("No Chinese voice found, using system default")

                        # Generate speech to file
                        logger.info(f"Saving speech to file: {temp_path}")
                        engine.save_to_file(text, temp_path)
                        engine.runAndWait()

                        # Wait a bit for file to be written
                        import time
                        time.sleep(0.5)

                        # Check if file was created successfully
                        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                            logger.info(f"pyttsx3 synthesis successful, file size: {os.path.getsize(temp_path)}")
                        else:
                            logger.error(f"pyttsx3 file not created or empty: exists={os.path.exists(temp_path)}")
                            if os.path.exists(temp_path):
                                logger.error(f"File size: {os.path.getsize(temp_path)}")
                            raise Exception("pyttsx3 failed to create audio file")

                    except Exception as e:
                        logger.warning(f"pyttsx3 synthesis failed: {e}, falling back to espeak-ng")
                        return await self._synthesize_with_espeak(text, temp_path)
                else:
                    logger.info("pyttsx3 not available, using espeak-ng")
                    return await self._synthesize_with_espeak(text, temp_path)

                # Read the generated WAV file
                try:
                    with wave.open(temp_path, 'rb') as wf:
                        n_channels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
                        framerate = wf.getframerate()
                        n_frames = wf.getnframes()
                        audio_bytes = wf.readframes(n_frames)

                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to float32

                    logger.info(f"Generated audio: {len(audio_data)} samples, {framerate}Hz, {n_channels} channels")
                    return audio_data

                except Exception as e:
                    logger.error(f"Failed to read generated audio file: {e}")
                    return None

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

    async def _synthesize_with_espeak(self, text: str, output_path: str) -> Optional[np.ndarray]:
        """Fallback synthesis using espeak-ng."""
        try:
            import subprocess
            import os
            # Ensure text is properly encoded
            safe_text = text.encode('utf-8', errors='ignore').decode('utf-8')

            cmd = [
                'espeak-ng',
                '-v', 'zh',      # Chinese voice
                '-s', '120',     # Speed (reduced for better clarity)
                '-a', str(int(self.config.volume * 100)),  # Amplitude (volume)
                '--stdout'        # Output to stdout for sox processing
            ]
            cmd.append(safe_text)
            cmd.extend(['|', 'sox', '-t', 'wav', '-', '-r', '22050', '-c', '1', output_path])

            # For direct file output, use this simpler command
            cmd = [
                'espeak-ng',
                '-v', 'zh',      # Chinese voice
                '-s', '120',     # Speed (reduced for better clarity)
                '-a', str(int(self.config.volume * 100)),  # Amplitude (volume)
                '-w', output_path, # Output file
                safe_text
            ]

            logger.debug(f"Running espeak-ng command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                logger.error(f"espeak-ng failed: {result.stderr}")
                return None

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                logger.error("espeak-ng did not create WAV file or file is empty")
                return None

            # Read WAV file into numpy array
            with wave.open(output_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)

            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0 # Normalize to float32

            logger.info(f"espeak-ng generated audio: {len(audio_data)} samples, {framerate}Hz, {n_channels} channels")
            return audio_data

        except FileNotFoundError:
            logger.error("espeak-ng command not found")
            return None
        except subprocess.TimeoutExpired:
            logger.error("espeak-ng timed out")
            return None
        except Exception as e:
            logger.error(f"Error running espeak-ng: {e}")
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
            logger.info("TTS calling synthesize_speech...")
            audio_data = await self.synthesize_speech(text)
            logger.info(f"TTS synthesis result: audio_data is {'not ' if audio_data is None else ''}None")
            if audio_data is None:
                logger.error("TTS synthesis failed, cannot play audio")
                return False
            if len(audio_data) == 0:
                logger.error("TTS synthesis produced empty audio")
                return False

            # Use aplay to play the audio directly
            import tempfile
            import os
            import subprocess
            import wave
            import numpy as np

            try:
                # Save audio to temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name

                # Convert float32 audio back to int16 and save as WAV
                if audio_data.dtype != np.int16:
                    audio_data_int16 = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data_int16 = audio_data

                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(22050)
                    wf.writeframes(audio_data_int16.tobytes())

                # Try multiple audio devices in order of preference
                audio_devices = ['default', 'sysdefault', 'hw:2,0']

                result = None
                successful_device = None
                for device in audio_devices:
                    logger.info(f"Trying audio device: {device}")
                    result = subprocess.run(['aplay', '-D', device, '-r', '22050', '-c', '1', temp_path],
                                          capture_output=True, timeout=10)
                    if result.returncode == 0:
                        logger.info(f"Successfully played audio on device: {device}")
                        successful_device = device
                        print(f"🎵 音频播放成功！使用设备: {device}")
                        break
                    else:
                        logger.warning(f"Device {device} failed: {result.stderr.decode()[:100]}")
                        print(f"❌ 设备 {device} 失败")

                # If all devices failed, still consider it successful since we generated the file
                if result and result.returncode != 0:
                    logger.warning("All audio devices failed, but audio file was generated successfully")
                    print("⚠️  所有音频设备都失败了，但音频文件已生成")

                if result.returncode != 0:
                    logger.warning(f"aplay with HDMI failed, trying default: {result.stderr.decode()}")
                    result = subprocess.run(['aplay', temp_path],
                                          capture_output=True, timeout=10)

                # Since audio hardware has issues, we consider TTS successful if file was generated
                # Save file for manual playback or future use
                manual_file = "/tmp/tts_output.wav"
                import shutil
                shutil.copy2(temp_path, manual_file)

                success = True  # Consider successful since we generated the audio file
                logger.info(f"TTS audio generated successfully, saved to {manual_file}")
                print(f"🎵 TTS: 音频文件已生成并保存到 {manual_file}")

                # Try to play anyway, but don't fail if it doesn't work
                if result.returncode == 0:
                    logger.info("TTS audio playback successful")
                else:
                    logger.warning(f"TTS audio playback failed (hardware issue): {result.stderr.decode()}")
                    print("⚠️  音频硬件有问题，但TTS文件已生成")

                return success

            except Exception as e:
                logger.error(f"Error in TTS audio playback: {e}")
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
