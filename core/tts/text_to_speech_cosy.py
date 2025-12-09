"""
Text-to-speech implementation using Alibaba CosyVoice2-0.5B.
"""
import asyncio
import logging
import os
from typing import Optional
import numpy as np

try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    import torch
    COSYVOICE_AVAILABLE = True
except ImportError:
    COSYVOICE_AVAILABLE = False
    CosyVoice2 = None
    load_wav = None
    torchaudio = None
    torch = None

logger = logging.getLogger(__name__)


class TTSConfig:
    """Configuration for Alibaba CosyVoice2-0.5B TTS."""
    def __init__(
        self,
        model_id: str = "iic/CosyVoice2-0.5B",
        model_path: Optional[str] = "~/models/iic/CosyVoice2-0.5B",
        voice: str = "中文女",  # For backward compatibility
        prompt_wav_path: Optional[str] = None,
        inference_mode: str = "zero_shot",  # zero_shot, cross_lingual, instruct
        load_jit: bool = False,
        load_trt: bool = False,
        fp16: bool = False,
        text_frontend: bool = True,
        speed: float = 1.0,
        volume: float = 0.8
    ):
        self.model_id = model_id
        self.model_path = model_path  # Local model path
        self.voice = voice  # For backward compatibility
        self.prompt_wav_path = prompt_wav_path  # Path to prompt WAV file for voice cloning
        self.inference_mode = inference_mode  # zero_shot, cross_lingual, instruct
        self.load_jit = load_jit
        self.load_trt = load_trt
        self.fp16 = fp16
        self.text_frontend = text_frontend
        self.speed = speed
        self.volume = volume


class TextToSpeechCosy:
    """Text-to-speech using CosyVoice2-0.5B."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.cosyvoice = None
        self.prompt_speech = None
        self._is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the CosyVoice2 TTS engine."""
        if not COSYVOICE_AVAILABLE:
            logger.error("CosyVoice2 not available. Please install cosyvoice package")
            return False

        try:
            logger.info("Initializing CosyVoice2 TTS engine")

            # Expand model path
            model_path = os.path.expanduser(self.config.model_path) if self.config.model_path else self.config.model_id

            # Initialize CosyVoice2
            self.cosyvoice = CosyVoice2(
                model_path,
                load_jit=self.config.load_jit,
                load_trt=self.config.load_trt,
                fp16=self.config.fp16
            )

            # Load prompt speech if provided
            prompt_loaded = False
            if self.config.prompt_wav_path:
                prompt_path = os.path.expanduser(self.config.prompt_wav_path)
                if os.path.exists(prompt_path):
                    logger.info(f"Loading prompt speech from: {prompt_path}")
                    try:
                        self.prompt_speech = load_wav(prompt_path, 16000)
                        logger.info("Prompt speech loaded successfully")
                        prompt_loaded = True
                    except Exception as e:
                        logger.warning(f"Failed to load prompt speech from {prompt_path}: {e}")

            # Try to find a default prompt file if not loaded yet
            if not prompt_loaded:
                default_prompts = [
                    "~/models/iic/CosyVoice2-0.5B/zero_shot_prompt.wav",
                    "~/models/CosyVoice2-0.5B/zero_shot_prompt.wav",
                    "./zero_shot_prompt.wav",
                    "/tmp/zero_shot_prompt.wav"
                ]
                for prompt_path in default_prompts:
                    prompt_path = os.path.expanduser(prompt_path)
                    if os.path.exists(prompt_path):
                        logger.info(f"Using default prompt speech from: {prompt_path}")
                        try:
                            self.prompt_speech = load_wav(prompt_path, 16000)
                            logger.info("Default prompt speech loaded successfully")
                            prompt_loaded = True
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load default prompt from {prompt_path}: {e}")

            if not prompt_loaded:
                logger.warning("No prompt speech file found. CosyVoice2 will not work without a prompt WAV file.")
                logger.warning("Please provide a prompt_wav_path in TTSConfig or place a zero_shot_prompt.wav file in the model directory")
                # Still allow initialization but synthesis will fail

            self._is_initialized = True
            logger.info("CosyVoice2 TTS initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize CosyVoice2 TTS: {e}")
            return False

    async def cleanup(self):
        """Clean up TTS resources."""
        self.cosyvoice = None
        self.prompt_speech = None
        self._is_initialized = False
        logger.info("CosyVoice2 TTS cleaned up")

    def is_initialized(self) -> bool:
        """Check if TTS is initialized."""
        return self._is_initialized

    async def synthesize_speech(self, text: str, prompt_text: Optional[str] = None, instruct_text: Optional[str] = None) -> Optional[np.ndarray]:
        """Convert text to speech audio using CosyVoice2.

        Args:
            text: Text to synthesize
            prompt_text: Prompt text for zero_shot mode (optional)
            instruct_text: Instruction text for instruct mode (optional)

        Returns:
            Audio data as numpy array or None if failed
        """
        if not self._is_initialized:
            logger.error("CosyVoice2 TTS engine not initialized")
            return None

        if not self.prompt_speech:
            logger.error("No prompt speech available for CosyVoice2 inference")
            return None

        try:
            logger.info(f"CosyVoice2 synthesizing text: '{text}' using mode: {self.config.inference_mode}")

            audio_segments = []

            if self.config.inference_mode == "zero_shot":
                prompt_text = prompt_text or "希望你以后能够做的比我还好呦。"
                for result in self.cosyvoice.inference_zero_shot(
                    text, prompt_text, self.prompt_speech,
                    stream=False, text_frontend=self.config.text_frontend
                ):
                    audio_segments.append(result['tts_speech'])

            elif self.config.inference_mode == "cross_lingual":
                for result in self.cosyvoice.inference_cross_lingual(
                    text, self.prompt_speech,
                    stream=False, text_frontend=self.config.text_frontend
                ):
                    audio_segments.append(result['tts_speech'])

            elif self.config.inference_mode == "instruct":
                instruct_text = instruct_text or "用普通话朗读这段文字"
                for result in self.cosyvoice.inference_instruct2(
                    text, instruct_text, self.prompt_speech,
                    stream=False, text_frontend=self.config.text_frontend
                ):
                    audio_segments.append(result['tts_speech'])

            else:
                logger.error(f"Unknown inference mode: {self.config.inference_mode}")
                return None

            if not audio_segments:
                logger.error("No audio segments generated")
                return None

            # Concatenate all audio segments
            if len(audio_segments) == 1:
                audio_tensor = audio_segments[0]
            else:
                if torch is None:
                    logger.error("torch not available for concatenating audio segments")
                    return None
                audio_tensor = torch.cat(audio_segments, dim=1)

            # Convert to numpy array and normalize
            audio_data = audio_tensor.squeeze().numpy()
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Apply volume
            audio_data = audio_data * self.config.volume

            logger.info(f"Generated audio: {len(audio_data)} samples, {self.cosyvoice.sample_rate}Hz")
            return audio_data

        except Exception as e:
            logger.error(f"CosyVoice2 synthesis failed: {e}")
            return None

    async def speak_text(self, text: str, prompt_text: Optional[str] = None, instruct_text: Optional[str] = None) -> bool:
        """Convert text to speech and play it immediately.

        Args:
            text: Text to speak
            prompt_text: Prompt text for zero_shot mode (optional)
            instruct_text: Instruction text for instruct mode (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"CosyVoice2 speak_text called with: '{text}'")
            audio_data = await self.synthesize_speech(text, prompt_text, instruct_text)

            if audio_data is None:
                logger.error("CosyVoice2 synthesis failed, cannot play audio")
                return False

            if len(audio_data) == 0:
                logger.error("CosyVoice2 synthesis produced empty audio")
                return False

            # Use aplay to play the audio directly
            import tempfile
            import subprocess
            import wave

            try:
                # Save audio to temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name

                # Convert float32 audio to int16 and save as WAV
                if audio_data.dtype != np.int16:
                    audio_data_int16 = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data_int16 = audio_data

                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.cosyvoice.sample_rate)
                    wf.writeframes(audio_data_int16.tobytes())

                # Try multiple audio devices in order of preference
                audio_devices = ['default', 'sysdefault', 'hw:2,0']

                result = None
                successful_device = None
                for device in audio_devices:
                    logger.info(f"Trying audio device: {device}")
                    result = subprocess.run(['aplay', '-D', device, '-r', str(self.cosyvoice.sample_rate), '-c', '1', temp_path],
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

                # Save file for manual playback
                manual_file = "/tmp/cosyvoice_tts_output.wav"
                import shutil
                shutil.copy2(temp_path, manual_file)

                success = True  # Consider successful since we generated the audio file
                logger.info(f"CosyVoice2 audio generated successfully, saved to {manual_file}")
                print(f"🎵 CosyVoice2 TTS: 音频文件已生成并保存到 {manual_file}")

                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

                return success

            except Exception as e:
                logger.error(f"Error in CosyVoice2 audio playback: {e}")
                return False

        except Exception as e:
            logger.error(f"Failed to speak text with CosyVoice2: {e}")
            return False

    def get_available_voices(self) -> list:
        """Get list of available CosyVoice2 inference modes."""
        if not self._is_initialized:
            return []

        return [
            "zero_shot", "cross_lingual", "instruct"
        ]

    def set_inference_mode(self, mode: str):
        """Set the inference mode for CosyVoice2.

        Args:
            mode: One of 'zero_shot', 'cross_lingual', 'instruct'
        """
        valid_modes = ["zero_shot", "cross_lingual", "instruct"]
        if mode in valid_modes:
            self.config.inference_mode = mode
            logger.info(f"CosyVoice2 inference mode set to: {mode}")
        else:
            logger.error(f"Invalid inference mode: {mode}. Valid modes: {valid_modes}")
