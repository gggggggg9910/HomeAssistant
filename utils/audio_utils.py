"""
Audio processing utilities.
"""
import numpy as np
from typing import Optional, Tuple


def normalize_audio(audio: np.ndarray, target_dBFS: float = -20.0) -> np.ndarray:
    """Normalize audio to target dBFS level.

    Args:
        audio: Input audio array
        target_dBFS: Target dBFS level

    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio

    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))

    if rms == 0:
        return audio

    # Calculate current dBFS
    current_dBFS = 20 * np.log10(rms)

    # Calculate gain needed
    gain = 10 ** ((target_dBFS - current_dBFS) / 20)

    # Apply gain
    normalized = audio * gain

    # Prevent clipping
    max_val = np.max(np.abs(normalized))
    if max_val > 1.0:
        normalized = normalized / max_val

    return normalized


def detect_silence(
    audio: np.ndarray,
    threshold_dBFS: float = -40.0,
    min_silence_duration: float = 0.5,
    sample_rate: int = 16000
) -> Tuple[bool, float]:
    """Detect if audio contains silence.

    Args:
        audio: Audio array
        threshold_dBFS: Silence threshold in dBFS
        min_silence_duration: Minimum silence duration in seconds
        sample_rate: Sample rate

    Returns:
        Tuple of (is_silent, silence_duration)
    """
    if len(audio) == 0:
        return True, 0.0

    # Calculate RMS
    rms = np.sqrt(np.mean(audio ** 2))

    if rms == 0:
        return True, len(audio) / sample_rate

    # Calculate dBFS
    dBFS = 20 * np.log10(rms)

    # Check if below threshold
    is_silent = dBFS < threshold_dBFS

    if is_silent:
        duration = len(audio) / sample_rate
        return True, duration
    else:
        return False, 0.0


def trim_silence(
    audio: np.ndarray,
    threshold_dBFS: float = -40.0,
    min_silence_duration: float = 0.1,
    sample_rate: int = 16000
) -> np.ndarray:
    """Trim silence from beginning and end of audio.

    Args:
        audio: Input audio array
        threshold_dBFS: Silence threshold in dBFS
        min_silence_duration: Minimum silence duration to trim
        sample_rate: Sample rate

    Returns:
        Trimmed audio array
    """
    if len(audio) == 0:
        return audio

    # Convert silence duration to samples
    min_silence_samples = int(min_silence_duration * sample_rate)

    # Find non-silent regions
    window_size = min(1024, len(audio) // 4)
    if window_size < 1:
        window_size = 1

    # Calculate RMS in windows
    rms_values = []
    for i in range(0, len(audio) - window_size + 1, window_size // 2):
        window = audio[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append((i, rms))

    # Find regions above threshold
    threshold_linear = 10 ** (threshold_dBFS / 20)
    active_regions = []

    start_idx = None
    for i, (pos, rms) in enumerate(rms_values):
        if rms > threshold_linear:
            if start_idx is None:
                start_idx = pos
        else:
            if start_idx is not None:
                # Check if silence region is long enough
                silence_start = rms_values[start_idx // (window_size // 2)][0]
                silence_end = pos
                if silence_end - silence_start >= min_silence_samples:
                    active_regions.append((start_idx, pos))
                start_idx = None

    # Handle case where audio ends with active region
    if start_idx is not None:
        active_regions.append((start_idx, len(audio)))

    if not active_regions:
        return audio

    # Find the main active region (longest or first significant one)
    main_region = max(active_regions, key=lambda x: x[1] - x[0])

    # Extract the region with some padding
    padding = min(1024, (main_region[1] - main_region[0]) // 10)
    start = max(0, main_region[0] - padding)
    end = min(len(audio), main_region[1] + padding)

    return audio[start:end]


def resample_audio(
    audio: np.ndarray,
    from_rate: int,
    to_rate: int,
    quality: str = "linear"
) -> np.ndarray:
    """Resample audio to different sample rate.

    Args:
        audio: Input audio array
        from_rate: Original sample rate
        to_rate: Target sample rate
        quality: Resampling quality ("linear" or "cubic")

    Returns:
        Resampled audio array
    """
    if from_rate == to_rate:
        return audio

    try:
        import scipy.signal
        # Calculate resampling ratio
        ratio = to_rate / from_rate

        # Resample
        if quality == "cubic":
            resampled = scipy.signal.resample_poly(audio, to_rate, from_rate)
        else:
            # Linear interpolation
            resampled = scipy.signal.resample(audio, int(len(audio) * ratio))

        return resampled.astype(audio.dtype)

    except ImportError:
        # Fallback: simple linear interpolation
        import numpy as np
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)


def calculate_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Calculate audio duration in seconds.

    Args:
        audio: Audio array
        sample_rate: Sample rate

    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate


def audio_to_bytes(audio: np.ndarray, sample_width: int = 2) -> bytes:
    """Convert audio array to bytes.

    Args:
        audio: Audio array (float32 normalized to [-1, 1])
        sample_width: Sample width in bytes (1, 2, or 4)

    Returns:
        Audio data as bytes
    """
    # Convert to appropriate integer type
    if sample_width == 1:
        # 8-bit unsigned
        audio_int = ((audio + 1.0) * 127.5).astype(np.uint8)
    elif sample_width == 2:
        # 16-bit signed
        audio_int = (audio * 32767).astype(np.int16)
    elif sample_width == 4:
        # 32-bit signed
        audio_int = (audio * 2147483647).astype(np.int32)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    return audio_int.tobytes()


def bytes_to_audio(audio_bytes: bytes, sample_width: int = 2) -> np.ndarray:
    """Convert bytes to audio array.

    Args:
        audio_bytes: Audio data as bytes
        sample_width: Sample width in bytes

    Returns:
        Audio array (float32 normalized to [-1, 1])
    """
    # Convert from appropriate integer type
    if sample_width == 1:
        # 8-bit unsigned
        audio_int = np.frombuffer(audio_bytes, dtype=np.uint8)
        audio_float = (audio_int.astype(np.float32) / 127.5) - 1.0
    elif sample_width == 2:
        # 16-bit signed
        audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int.astype(np.float32) / 32768.0
    elif sample_width == 4:
        # 32-bit signed
        audio_int = np.frombuffer(audio_bytes, dtype=np.int32)
        audio_float = audio_int.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    return audio_float
