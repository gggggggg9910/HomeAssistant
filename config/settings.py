"""
Configuration settings for Home Assistant.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class AudioSettings(BaseSettings):
    """Audio configuration settings."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    input_device: Optional[int] = None
    output_device: Optional[int] = None


class KWSSettings(BaseSettings):
    """Keyword spotting configuration."""
    model_path: str = "~/models/kws-onnx"
    keyword: str = "你好小助手"
    threshold: float = 0.5
    max_wait_seconds: int = 30


class ASRSettings(BaseSettings):
    """Speech recognition configuration."""
    model_id: str = "iic/SenseVoiceSmall"  # Alibaba SenseVoice model
    model_path: str = "~/models/sensevoice"  # Local model cache path
    language: str = "zh"
    max_wait_seconds: int = 10
    use_gpu: bool = False  # Use GPU if available
    disable_update: bool = True  # Disable model update checks for faster startup


class TTSSettings(BaseSettings):
    """Text-to-speech configuration."""
    model_id: str = "iic/CosyVoice2-0.5B"  # Alibaba CosyVoice model
    model_path: str = "~/models/iic/CosyVoice2-0.5B/CosyVoice-BlankEN"  # Local model directory with config.json
    voice: str = "中文女"  # Voice style for CosyVoice
    speed: float = 1.0
    volume: float = 0.8
    use_gpu: bool = False  # Use GPU if available


class LLMSettings(BaseSettings):
    """Large language model configuration."""
    DASHSCOPE_API_KEY: Optional[str] = None  # DashScope API key for Alibaba Qwen
    api_key: Optional[str] = None  # Will be set from DASHSCOPE_API_KEY
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    model: str = "qwen-turbo"  # Alibaba Qwen model
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    use_local: bool = False  # Use local Qwen model instead of API

    def __init__(self, **data):
        super().__init__(**data)
        # Set api_key from DASHSCOPE_API_KEY for backward compatibility
        if self.api_key is None and self.DASHSCOPE_API_KEY is not None:
            self.api_key = self.DASHSCOPE_API_KEY


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    level: str = "INFO"
    file_path: str = "logs/assistant.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class Settings(BaseSettings):
    """Main configuration class."""
    # Base paths
    base_dir: Path = Path(__file__).parent.parent
    models_dir: Path = Path.home() / "models"

    # Component settings
    audio: AudioSettings = AudioSettings()
    kws: KWSSettings = KWSSettings()
    asr: ASRSettings = ASRSettings()
    tts: TTSSettings = TTSSettings()
    llm: LLMSettings = LLMSettings()
    logging: LoggingSettings = LoggingSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        extra = "allow"


# Global settings instance
settings = Settings()


def initialize_logging():
    """Initialize logging based on settings."""
    from utils.logging_utils import setup_logging

    log_file = settings.logging.file_path
    if log_file and not Path(log_file).is_absolute():
        # Make log file path relative to project root
        log_file = settings.base_dir / log_file

    setup_logging(
        level=settings.logging.level,
        log_file=str(log_file) if log_file else None,
        max_file_size=settings.logging.max_file_size,
        backup_count=settings.logging.backup_count
    )
