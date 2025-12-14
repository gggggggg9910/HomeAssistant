"""
Text-to-Speech module for voice output.

Available TTS engines:
- TextToSpeechPiper: Fast, lightweight, optimized for Raspberry Pi (recommended)
- TextToSpeechCosy: High quality CosyVoice2, but slow on Pi
- TextToSpeech: Basic pyttsx3/espeak fallback

Usage:
    # For Raspberry Pi (recommended)
    from core.tts import PiperTTSConfig, TextToSpeechPiper
    
    # For high-end hardware with CosyVoice2
    from core.tts import TTSConfig, TextToSpeechCosy
"""

# Import all TTS implementations
from .text_to_speech_piper import (
    PiperTTSConfig,
    TextToSpeechPiper,
)

from .text_to_speech_cosy import (
    TTSConfig as CosyTTSConfig,
    TextToSpeechCosy,
)

from .text_to_speech import (
    TTSConfig as BasicTTSConfig,
    TextToSpeech as TextToSpeechBasic,
)

# Default exports - Use Piper for Raspberry Pi
TTSConfig = PiperTTSConfig
TextToSpeech = TextToSpeechPiper

__all__ = [
    # Piper TTS (recommended for Pi)
    'PiperTTSConfig',
    'TextToSpeechPiper',
    
    # CosyVoice2 TTS (high quality, slow on Pi)
    'CosyTTSConfig', 
    'TextToSpeechCosy',
    
    # Basic TTS (fallback)
    'BasicTTSConfig',
    'TextToSpeechBasic',
    
    # Default aliases
    'TTSConfig',
    'TextToSpeech',
]
