"""
Logging utilities for the voice assistant.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    use_colors: bool = True
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
        use_colors: Whether to use colored console output

    Returns:
        Root logger instance
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    if use_colors and COLORLOG_AVAILABLE:
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Reduce noise from some libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


class AudioLogger:
    """Specialized logger for audio-related events."""

    def __init__(self, name: str = "audio"):
        self.logger = logging.getLogger(name)

    def log_audio_chunk(self, chunk_size: int, sample_rate: int):
        """Log audio chunk processing."""
        self.logger.debug(f"Processed audio chunk: {chunk_size} samples @ {sample_rate}Hz")

    def log_keyword_detected(self, keyword: str, confidence: float):
        """Log keyword detection."""
        self.logger.info(f"Keyword detected: '{keyword}' (confidence: {confidence:.3f})")

    def log_speech_recognized(self, text: str, duration: Optional[float] = None):
        """Log speech recognition result."""
        if duration:
            self.logger.info(f"Speech recognized ({duration:.2f}s): '{text}'")
        else:
            self.logger.info(f"Speech recognized: '{text}'")

    def log_tts_synthesized(self, text_length: int, audio_duration: Optional[float] = None):
        """Log TTS synthesis."""
        if audio_duration:
            self.logger.info(f"TTS synthesized: {text_length} chars -> {audio_duration:.2f}s audio")
        else:
            self.logger.info(f"TTS synthesized: {text_length} chars")

    def log_llm_request(self, user_input: str, response_length: int):
        """Log LLM API request."""
        self.logger.info(f"LLM request: '{user_input}' -> {response_length} chars response")
