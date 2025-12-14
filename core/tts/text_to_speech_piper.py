"""
Text-to-speech implementation using Piper TTS.
Optimized for Raspberry Pi 5 - fast, lightweight, and natural sounding.

Piper TTS is a fast, local neural text to speech system that sounds great 
and is optimized for the Raspberry Pi 4/5.

GitHub: https://github.com/rhasspy/piper
"""
import asyncio
import logging
import os
import json
import wave
import tempfile
import subprocess
import shutil
from typing import Optional, Dict, List, Any
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default model directory - HARDCODED, never changes
# Piper TTS uses its own dedicated directory, independent from other TTS engines
DEFAULT_MODEL_DIR = os.path.expanduser("~/models/piper")

# Pre-defined voice configurations
# Format: voice_id -> (model_file, config_file, description)
BUILT_IN_VOICES = {
    # Chinese voices
    "zh_CN_huayan_medium": {
        "model": "zh_CN-huayan-medium.onnx",
        "config": "zh_CN-huayan-medium.onnx.json",
        "description": "中文女声 - 华妍 (中等质量)",
        "language": "zh-CN",
        "sample_rate": 22050,
    },
    # You can add more voices here after downloading
    # "zh_CN_male_medium": {
    #     "model": "zh_CN-male-medium.onnx",
    #     "config": "zh_CN-male-medium.onnx.json", 
    #     "description": "中文男声 (中等质量)",
    #     "language": "zh-CN",
    #     "sample_rate": 22050,
    # },
}


class PiperTTSConfig:
    """Configuration for Piper TTS.
    
    IMPORTANT: Model directory is hardcoded to ~/models/piper and cannot be changed.
    This ensures Piper TTS is completely independent from other TTS engines (like CosyVoice).
    Piper TTS has its own dedicated directory and doesn't interfere with other models.
    
    Supports legacy parameters (model_id, model_path, model_dir) for backward compatibility
    with controller.py, but these are ignored - the path is always ~/models/piper.
    """
    
    def __init__(
        self,
        # Voice settings
        voice: Optional[str] = None,
        custom_model_path: Optional[str] = None,
        custom_config_path: Optional[str] = None,
        
        # Synthesis settings
        speaker_id: int = 0,
        length_scale: Optional[float] = None,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sentence_silence: float = 0.2,
        
        # Audio settings
        volume: float = 0.9,
        sample_rate: int = 22050,
        
        # Performance settings
        use_cuda: bool = False,
        num_threads: int = 4,
        
        # Legacy parameters (for backward compatibility with controller.py)
        # These are accepted but IGNORED - path is always hardcoded to ~/models/piper
        model_id: Optional[str] = None,  # Ignored - Piper doesn't use model_id
        model_path: Optional[str] = None,  # Ignored - path is hardcoded, always ~/models/piper
        model_dir: Optional[str] = None,  # Ignored - path is hardcoded, always ~/models/piper
        speed: Optional[float] = None,  # Maps to length_scale
        
        # Additional voice configs
        voice_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize Piper TTS configuration.
        
        Model directory is ALWAYS hardcoded to ~/models/piper and cannot be changed.
        This ensures Piper TTS is completely independent from other TTS engines.
        
        Args:
            voice: Voice ID (default: zh_CN_huayan_medium). Can be "中文女" (auto-converted)
            speed: Speech speed (1.0 = normal, <1.0 = faster, >1.0 = slower). Maps to length_scale
            length_scale: Speech speed (alternative to speed parameter)
            volume: Output volume (0.0 - 1.0)
            model_id: Legacy parameter, IGNORED (Piper doesn't use model_id)
            model_path: Legacy parameter, IGNORED (path is always ~/models/piper)
            model_dir: Legacy parameter, IGNORED (path is always ~/models/piper)
            ... other parameters ...
        """
        # Model directory is HARDCODED - always ~/models/piper, never uses external config
        # This ensures Piper TTS is completely independent from CosyVoice and other engines
        self.model_dir = os.path.expanduser(DEFAULT_MODEL_DIR)
        logger.info(f"📁 Piper TTS model directory (HARDCODED, independent): {self.model_dir}")
        
        # Ignore any model_path/model_dir parameters passed from external config
        if model_path or model_dir:
            logger.debug(f"Ignoring external model_path/model_dir parameters (using hardcoded path)")
        
        # Voice mapping: convert legacy "中文女" to Piper voice ID
        if voice:
            if voice == "中文女":
                self.voice = "zh_CN_huayan_medium"
            else:
                self.voice = voice
        else:
            self.voice = "zh_CN_huayan_medium"
        
        self.custom_model_path = custom_model_path
        self.custom_config_path = custom_config_path
        
        # Speed/length_scale mapping
        if speed is not None:
            self.length_scale = speed
        elif length_scale is not None:
            self.length_scale = length_scale
        else:
            self.length_scale = 1.0
        
        self.speaker_id = speaker_id
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence
        self.volume = volume
        self.sample_rate = sample_rate
        self.use_cuda = use_cuda
        self.num_threads = num_threads
        self.voice_configs = voice_configs or {}


class TextToSpeechPiper:
    """Text-to-speech using Piper TTS - optimized for Raspberry Pi."""

    def __init__(self, config: PiperTTSConfig):
        self.config = config
        self._is_initialized = False
        self._piper_available = False
        self._model_path: Optional[str] = None
        self._config_path: Optional[str] = None
        self._model_config: Dict = {}
        self._voices: Dict[str, Dict] = {}  # Loaded voice configurations
        self._current_voice: str = config.voice
        
    async def initialize(self) -> bool:
        """Initialize the Piper TTS engine."""
        try:
            logger.info("Initializing Piper TTS engine")
            
            # Check if piper is available (command line tool)
            if not self._check_piper_available():
                logger.error("Piper TTS not available. Please install with: pip install piper-tts")
                return False
            
            # Set up model directory
            model_dir = Path(os.path.expanduser(self.config.model_dir))
            if not model_dir.exists():
                logger.info(f"Creating model directory: {model_dir}")
                model_dir.mkdir(parents=True, exist_ok=True)
            
            # Load the default voice model
            if not await self._load_voice(self.config.voice):
                logger.error(f"Failed to load voice: {self.config.voice}")
                return False
            
            self._is_initialized = True
            logger.info(f"Piper TTS initialized successfully with voice: {self.config.voice}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_piper_available(self) -> bool:
        """Check if piper command is available."""
        try:
            # Try piper command
            result = subprocess.run(
                ['piper', '--help'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                self._piper_available = True
                logger.info("Piper TTS command found")
                return True
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            logger.debug(f"Piper check error: {e}")
        
        # Try python -m piper
        try:
            result = subprocess.run(
                ['python', '-m', 'piper', '--help'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                self._piper_available = True
                logger.info("Piper TTS available via python -m piper")
                return True
        except Exception as e:
            logger.debug(f"Piper module check error: {e}")
        
        logger.error("Piper TTS not found. Install with: pip install piper-tts")
        return False
    
    async def _load_voice(self, voice_id: str) -> bool:
        """Load a voice model.
        
        Args:
            voice_id: Voice identifier (from BUILT_IN_VOICES or custom model name)
            
        Returns:
            True if successful
        """
        model_dir = Path(os.path.expanduser(self.config.model_dir))
        logger.debug(f"Loading voice '{voice_id}' from model_dir: {model_dir}")
        logger.debug(f"Model directory exists: {model_dir.exists()}")
        
        if model_dir.exists():
            logger.debug(f"Contents of {model_dir}: {list(model_dir.iterdir())}")
        
        # Check if using custom model paths
        if self.config.custom_model_path:
            model_path = Path(os.path.expanduser(self.config.custom_model_path))
            config_path = Path(os.path.expanduser(self.config.custom_config_path)) if self.config.custom_config_path else None
            
            if not config_path:
                # Try to find config file next to model
                config_path = model_path.with_suffix('.onnx.json')
            
            if not model_path.exists():
                logger.error(f"Custom model not found: {model_path}")
                return False
                
        elif voice_id in BUILT_IN_VOICES:
            # Use built-in voice configuration
            voice_config = BUILT_IN_VOICES[voice_id]
            model_path = model_dir / voice_config["model"]
            config_path = model_dir / voice_config["config"]
            
            logger.debug(f"Looking for model: {model_path}")
            logger.debug(f"Looking for config: {config_path}")
            logger.debug(f"Model file exists: {model_path.exists()}")
            logger.debug(f"Config file exists: {config_path.exists()}")
            
            if not model_path.exists():
                logger.error(f"Voice model not found: {model_path}")
                logger.error(f"Model directory: {model_dir}")
                logger.error(f"Expected model file: {voice_config['model']}")
                logger.error(f"Expected config file: {voice_config['config']}")
                if model_dir.exists():
                    logger.error(f"Files in directory: {[f.name for f in model_dir.iterdir()]}")
                logger.info(f"Please download the model. See instructions below.")
                self._print_download_instructions(voice_id, voice_config)
                return False
            
            # Config file is optional, warn but don't fail
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}, trying to find alternative")
                # Try alternative config file names
                alt_config_paths = [
                    model_dir / model_path.with_suffix('.onnx.json').name,  # Same name as model with .json
                    model_dir / f"{voice_id}.onnx.json",  # Voice ID based
                ]
                for alt_path in alt_config_paths:
                    if alt_path.exists():
                        logger.info(f"Found alternative config file: {alt_path}")
                        config_path = alt_path
                        break
                else:
                    logger.warning("No config file found, continuing without it (Piper may work without config)")
                    config_path = None
                
        else:
            # Try to find model file directly
            model_path = model_dir / f"{voice_id}.onnx"
            config_path = model_dir / f"{voice_id}.onnx.json"
            
            logger.debug(f"Trying direct model path: {model_path}")
            
            if not model_path.exists():
                logger.error(f"Voice model not found: {model_path}")
                logger.info("Available built-in voices:")
                for vid, vconfig in BUILT_IN_VOICES.items():
                    logger.info(f"  - {vid}: {vconfig['description']}")
                return False
        
        # Load model config (optional)
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._model_config = json.load(f)
                    self.config.sample_rate = self._model_config.get('audio', {}).get('sample_rate', 22050)
                    logger.info(f"Loaded model config, sample_rate: {self.config.sample_rate}")
            except Exception as e:
                logger.warning(f"Failed to load model config: {e}")
                self._model_config = {}
        else:
            logger.debug("No config file found, using defaults")
            self._model_config = {}
        
        self._model_path = str(model_path)
        self._config_path = str(config_path) if config_path else None
        self._current_voice = voice_id
        
        # Add to loaded voices
        self._voices[voice_id] = {
            "model_path": self._model_path,
            "config_path": self._config_path,
            "config": self._model_config,
        }
        
        logger.info(f"Voice loaded successfully: {voice_id} from {model_path}")
        return True
    
    def _print_download_instructions(self, voice_id: str, voice_config: Dict):
        """Print download instructions for a voice model."""
        model_name = voice_config["model"].replace(".onnx", "")
        logger.info("=" * 60)
        logger.info(f"Download instructions for voice: {voice_id}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Method 1: Download from Piper releases")
        logger.info(f"  wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-{model_name}.tar.gz")
        logger.info(f"  tar -xzf voice-{model_name}.tar.gz -C {self.config.model_dir}")
        logger.info("")
        logger.info("Method 2: Download individual files")
        logger.info(f"  wget -O {self.config.model_dir}/{voice_config['model']} \\")
        logger.info(f"    https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/{voice_config['model']}")
        logger.info(f"  wget -O {self.config.model_dir}/{voice_config['config']} \\")
        logger.info(f"    https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/{voice_config['config']}")
        logger.info("")
        logger.info("Method 3: Use piper to download automatically (if supported)")
        logger.info(f"  echo 'test' | piper --model {model_name} --download-dir {self.config.model_dir}")
        logger.info("=" * 60)

    async def cleanup(self):
        """Clean up TTS resources."""
        self._is_initialized = False
        self._model_path = None
        self._config_path = None
        self._voices.clear()
        logger.info("Piper TTS cleaned up")

    def is_initialized(self) -> bool:
        """Check if TTS is initialized."""
        return self._is_initialized

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS synthesis.
        
        Removes Markdown formatting, special characters, and normalizes text
        to ensure Piper TTS can process it correctly.
        
        Args:
            text: Raw text with possible Markdown/special characters
            
        Returns:
            Cleaned text suitable for TTS
        """
        import re
        
        # Remove Markdown formatting
        # Remove bold/italic: **text**, *text*, __text__, _text_
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
        text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
        text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
        
        # Remove code blocks: `code`, ```code```
        text = re.sub(r'```[^`]*```', '', text)  # Multi-line code blocks
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
        
        # Remove links: [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove headers: # Header, ## Header, etc.
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove list markers: -, *, 1., etc. (keep the text)
        text = re.sub(r'^[\s]*[-*•]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r'[ \t]+', ' ', text)     # Multiple spaces to one
        
        # Remove special characters that might cause issues
        # Keep Chinese characters, English letters, numbers, and common punctuation
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：、""''（）【】《》\.,!?;:\-]', '', text)
        
        # Normalize punctuation
        text = text.replace('，', '，').replace('。', '。')
        text = text.replace('！', '！').replace('？', '？')
        
        # Trim whitespace
        text = text.strip()
        
        # Limit text length (Piper may have issues with very long text)
        max_length = 2000  # Characters
        if len(text) > max_length:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_length} chars")
            # Try to truncate at sentence boundary
            sentences = re.split(r'[。！？\n]', text[:max_length])
            if len(sentences) > 1:
                text = '。'.join(sentences[:-1]) + '。'
            else:
                text = text[:max_length]
        
        return text

    async def synthesize_speech(self, text: str, voice: Optional[str] = None) -> Optional[np.ndarray]:
        """Convert text to speech audio using Piper.

        Args:
            text: Text to synthesize
            voice: Optional voice ID to use (defaults to current voice)

        Returns:
            Audio data as numpy array (float32, normalized) or None if failed
        """
        if not self._is_initialized:
            logger.error("Piper TTS engine not initialized")
            return None

        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        # Clean text before synthesis (remove Markdown, special chars, etc.)
        cleaned_text = self._clean_text_for_tts(text)
        if not cleaned_text or not cleaned_text.strip():
            logger.error("Text is empty after cleaning")
            return None
        
        if cleaned_text != text:
            logger.debug(f"Text cleaned: original length={len(text)}, cleaned length={len(cleaned_text)}")

        # Switch voice if needed
        if voice and voice != self._current_voice:
            if voice in self._voices:
                self._model_path = self._voices[voice]["model_path"]
                self._config_path = self._voices[voice]["config_path"]
                self._current_voice = voice
            elif not await self._load_voice(voice):
                logger.error(f"Failed to switch to voice: {voice}")
                return None

        try:
            logger.info(f"Piper synthesizing: '{cleaned_text[:50]}...' with voice: {self._current_voice}")
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                output_path = temp_file.name
            
            try:
                # Build piper command (text is passed via stdin, not command line)
                cmd = self._build_piper_command(output_path)
                logger.debug(f"Running piper command: {' '.join(cmd[:5])}...")
                
                # Run piper
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Send cleaned text to piper
                stdout, stderr = process.communicate(
                    input=cleaned_text.encode('utf-8'),
                    timeout=60  # 60 second timeout
                )
                
                if process.returncode != 0:
                    logger.error(f"Piper synthesis failed: {stderr.decode()}")
                    return None
                
                # Check if output file was created
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    logger.error("Piper did not generate audio file")
                    return None
                
                # Read the generated WAV file
                audio_data = self._read_wav_file(output_path)
                
                if audio_data is None:
                    return None
                
                # Apply volume
                audio_data = audio_data * self.config.volume
                
                logger.info(f"Generated audio: {len(audio_data)} samples, {self.config.sample_rate}Hz")
                return audio_data
                
            finally:
                # Clean up temp file
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except:
                        pass

        except subprocess.TimeoutExpired:
            logger.error("Piper synthesis timed out")
            return None
        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_piper_command(self, output_path: str) -> List[str]:
        """Build the piper command line. Text is passed via stdin."""
        cmd = [
            'piper',
            '--model', self._model_path,
            '--output_file', output_path,
        ]
        
        # Add config file if available
        if self._config_path and os.path.exists(self._config_path):
            cmd.extend(['--config', self._config_path])
        
        # Add synthesis parameters
        cmd.extend(['--length-scale', str(self.config.length_scale)])
        cmd.extend(['--noise-scale', str(self.config.noise_scale)])
        cmd.extend(['--noise-w', str(self.config.noise_w)])
        cmd.extend(['--sentence-silence', str(self.config.sentence_silence)])
        
        # Add speaker ID for multi-speaker models
        if self.config.speaker_id > 0:
            cmd.extend(['--speaker', str(self.config.speaker_id)])
        
        return cmd
    
    def _read_wav_file(self, wav_path: str) -> Optional[np.ndarray]:
        """Read WAV file and return normalized float32 numpy array."""
        try:
            with wave.open(wav_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
            
            # Update sample rate from actual file
            self.config.sample_rate = framerate
            
            # Convert bytes to numpy array
            if sampwidth == 2:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif sampwidth == 4:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int32)
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            
            # Convert stereo to mono if needed
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to read WAV file: {e}")
            return None

    async def speak_text(self, text: str, voice: Optional[str] = None) -> bool:
        """Convert text to speech and play it immediately.

        Args:
            text: Text to speak
            voice: Optional voice ID to use

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Piper speak_text: '{text[:50]}...'")
            
            audio_data = await self.synthesize_speech(text, voice)
            
            if audio_data is None:
                logger.error("Piper synthesis failed")
                return False

            if len(audio_data) == 0:
                logger.error("Piper synthesis produced empty audio")
                return False

            # Save and play audio
            return await self._play_audio(audio_data)

        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
            return False

    async def _play_audio(self, audio_data: np.ndarray) -> bool:
        """Save audio to file and play it."""
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Save WAV file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            # Try to play audio
            audio_devices = ['default', 'sysdefault', 'hw:0,0', 'hw:1,0', 'hw:2,0']
            played = False
            
            for device in audio_devices:
                try:
                    logger.debug(f"Trying audio device: {device}")
                    result = subprocess.run(
                        ['aplay', '-D', device, '-r', str(self.config.sample_rate), '-c', '1', temp_path],
                        capture_output=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        logger.info(f"Audio played on device: {device}")
                        played = True
                        print(f"🎵 音频播放成功！使用设备: {device}")
                        break
                except Exception as e:
                    logger.debug(f"Device {device} failed: {e}")
            
            # Save a copy for manual playback
            output_file = "/tmp/piper_tts_output.wav"
            shutil.copy2(temp_path, output_file)
            logger.info(f"Audio saved to: {output_file}")
            print(f"🎵 Piper TTS: 音频文件已保存到 {output_file}")
            
            # Clean up
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            return True  # Consider successful if audio was generated
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            return False

    def get_available_voices(self) -> List[str]:
        """Get list of available voices."""
        voices = []
        
        # Add built-in voices
        for voice_id, config in BUILT_IN_VOICES.items():
            voices.append(f"{voice_id}: {config['description']}")
        
        # Add any custom loaded voices
        for voice_id in self._voices:
            if voice_id not in BUILT_IN_VOICES:
                voices.append(f"{voice_id}: (custom)")
        
        return voices

    def set_voice(self, voice_id: str) -> bool:
        """Set the current voice.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            True if successful
        """
        if voice_id in self._voices:
            self._model_path = self._voices[voice_id]["model_path"]
            self._config_path = self._voices[voice_id]["config_path"]
            self._current_voice = voice_id
            logger.info(f"Switched to voice: {voice_id}")
            return True
        
        logger.error(f"Voice not loaded: {voice_id}. Call load_voice() first.")
        return False

    async def load_voice(self, voice_id: str) -> bool:
        """Load a voice model for use.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            True if successful
        """
        return await self._load_voice(voice_id)

    def get_current_voice(self) -> str:
        """Get the current voice ID."""
        return self._current_voice

    def set_speed(self, speed: float):
        """Set speech speed.
        
        Args:
            speed: Speed factor (1.0 = normal, 0.5 = twice as fast, 2.0 = half speed)
        """
        self.config.length_scale = speed
        logger.info(f"Speech speed set to: {speed}")

    def set_volume(self, volume: float):
        """Set output volume.
        
        Args:
            volume: Volume level (0.0 - 1.0)
        """
        self.config.volume = max(0.0, min(1.0, volume))
        logger.info(f"Volume set to: {self.config.volume}")


# Convenience alias for consistency with other TTS implementations
TTSConfig = PiperTTSConfig
TextToSpeech = TextToSpeechPiper


# Test function
async def test_piper_tts():
    """Test Piper TTS functionality."""
    print("=" * 60)
    print("Testing Piper TTS")
    print("=" * 60)
    
    config = PiperTTSConfig(
        model_dir="~/models/piper",
        voice="zh_CN_huayan_medium",
        length_scale=1.0,
        volume=0.9,
    )
    
    tts = TextToSpeechPiper(config)
    
    print("\n1. Initializing Piper TTS...")
    if not await tts.initialize():
        print("❌ Failed to initialize Piper TTS")
        print("\nPlease download the model first:")
        print("  mkdir -p ~/models/piper")
        print("  cd ~/models/piper")
        print("  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx")
        print("  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json")
        return
    
    print("✅ Piper TTS initialized")
    
    print("\n2. Testing speech synthesis...")
    test_texts = [
        "你好，欢迎回家。",
        "今天天气真不错，适合出去散步。",
        "我是你的智能助手，有什么可以帮助你的？",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Test {i}: '{text}'")
        success = await tts.speak_text(text)
        if success:
            print(f"   ✅ Synthesis successful")
        else:
            print(f"   ❌ Synthesis failed")
    
    print("\n3. Cleaning up...")
    await tts.cleanup()
    print("✅ Cleanup complete")
    
    print("\n" + "=" * 60)
    print("Piper TTS test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_piper_tts())
