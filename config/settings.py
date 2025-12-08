"""
Configuration settings for Home Assistant.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

# Manually load .env file before any BaseSettings usage
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not available


class AudioSettings(BaseSettings):
    """Audio configuration settings."""
    # 统一的采样率（向后兼容）
    sample_rate: int = 16000
    
    # 分别的输入输出采样率（优先使用这些）
    input_sample_rate: int = 16000    # 麦克风采样率
    output_sample_rate: int = 48000   # 扬声器采样率
    
    channels: int = 1
    chunk_size: int = 1024
    input_device: Optional[int] = None    # 麦克风设备索引
    output_device: Optional[int] = None   # 扬声器设备索引
    enable_input: bool = True             # 启用麦克风输入
    enable_output: bool = True            # 启用扬声器输出

    model_config = {
        "extra": "allow"
    }


class KWSSettings(BaseSettings):
    """Keyword spotting configuration."""
    model_path: str = "~/models/kws-onnx"
    keyword: str = "你好小助手"
    threshold: float = 0.5
    max_wait_seconds: int = 30

    model_config = {
        "extra": "allow"
    }


class ASRSettings(BaseSettings):
    """Speech recognition configuration."""
    model_id: str = "iic/SenseVoiceSmall"  # Alibaba SenseVoice model
    model_path: str = "~/models/sensevoice"  # Local model cache path
    language: str = "zh"
    max_wait_seconds: int = 10
    use_gpu: bool = False  # Use GPU if available
    disable_update: bool = True  # Disable model update checks for faster startup

    model_config = {
        "extra": "allow"
    }


class TTSSettings(BaseSettings):
    """Text-to-speech configuration."""
    model_id: str = "iic/CosyVoice2-0.5B"  # Alibaba CosyVoice model
    model_path: str = "~/models/iic/CosyVoice2-0.5B/CosyVoice-BlankEN"  # Local model directory with config.json
    voice: str = "中文女"  # Voice style for CosyVoice
    speed: float = 1.0
    volume: float = 0.8
    use_gpu: bool = False  # Use GPU if available

    model_config = {
        "extra": "allow"
    }


class LLMSettings(BaseSettings):
    """Large language model configuration."""
    api_key: Optional[str] = None  # DashScope API key for Alibaba Qwen
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    model: str = "qwen-turbo"  # Alibaba Qwen model
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    use_local: bool = False  # Use local Qwen model instead of API

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow"
    }

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        """Custom source for environment variables."""
        # First try dotenv, then env
        sources = [dotenv_settings, env_settings]

        # Add direct os.environ access as fallback
        def custom_env_source():
            d = {}
            for field_name, field_info in cls.model_fields.items():
                if field_name == 'api_key':
                    env_value = os.environ.get('DASHSCOPE_API_KEY')
                    if env_value:
                        d[field_name] = env_value
            return d

        sources.append(custom_env_source)
        return (init_settings, *sources, file_secret_settings)


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    level: str = "INFO"
    file_path: str = "logs/assistant.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

    model_config = {
        "extra": "allow"
    }


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
