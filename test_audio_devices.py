#!/usr/bin/env python3
"""
Audio device diagnostic script for HomeAssistant.
Helps identify available audio input/output devices.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    logger.error("PyAudio not available. Install with: pip install pyaudio")
    PYAUDIO_AVAILABLE = False
    sys.exit(1)

def list_audio_devices():
    """List all available audio devices."""
    if not PYAUDIO_AVAILABLE:
        return

    audio = pyaudio.PyAudio()

    print("=" * 60)
    print("AUDIO DEVICES DIAGNOSTIC")
    print("=" * 60)

    print(f"PyAudio version: {pyaudio.__version__}")
    print(f"PortAudio version: {pyaudio.get_portaudio_version_text()}")
    print()

    # Get device count
    device_count = audio.get_device_count()
    print(f"Total audio devices found: {device_count}")
    print()

    # List all devices
    print("ALL DEVICES:")
    print("-" * 40)
    for i in range(device_count):
        try:
            device_info = audio.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']}")
            print(f"  Max Input Channels: {device_info['maxInputChannels']}")
            print(f"  Max Output Channels: {device_info['maxOutputChannels']}")
            print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
            print()
        except Exception as e:
            print(f"Device {i}: Error getting info - {e}")
            print()

    # Get default devices
    print("DEFAULT DEVICES:")
    print("-" * 40)
    try:
        default_input = audio.get_default_input_device_info()
        print(f"Default Input Device: {default_input['name']} (Index: {default_input['index']})")
        print(f"  Max Input Channels: {default_input['maxInputChannels']}")
        print(f"  Default Sample Rate: {default_input['defaultSampleRate']}")
    except Exception as e:
        print(f"Default Input Device: ERROR - {e}")

    print()

    try:
        default_output = audio.get_default_output_device_info()
        print(f"Default Output Device: {default_output['name']} (Index: {default_output['index']})")
        print(f"  Max Output Channels: {default_output['maxOutputChannels']}")
        print(f"  Default Sample Rate: {default_output['defaultSampleRate']}")
    except Exception as e:
        print(f"Default Output Device: ERROR - {e}")

    print()
    print("=" * 60)

    # Test basic functionality
    print("FUNCTIONALITY TEST:")
    print("-" * 40)

    # Test input device
    try:
        if device_count > 0:
            # Try to open a stream with default settings
            test_stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            test_stream.close()
            print("✓ Input stream test: PASSED")
        else:
            print("✗ Input stream test: NO DEVICES")
    except Exception as e:
        print(f"✗ Input stream test: FAILED - {e}")

    # Test output device
    try:
        if device_count > 0:
            # Try to open an output stream
            test_stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True,
                frames_per_buffer=1024
            )
            test_stream.close()
            print("✓ Output stream test: PASSED")
        else:
            print("✗ Output stream test: NO DEVICES")
    except Exception as e:
        print(f"✗ Output stream test: FAILED - {e}")

    audio.terminate()

    print()
    print("RECOMMENDATIONS:")
    print("-" * 40)
    if device_count == 0:
        print("• No audio devices detected. Check your audio hardware and drivers.")
        print("• On Windows: Check Device Manager for audio devices")
        print("• On Linux: Check 'arecord -l' and 'aplay -l'")
        print("• On macOS: Check System Preferences > Sound")
    else:
        print("• Audio devices are available")
        print("• If still getting errors, try specifying device indices manually")
        print("• Check that audio devices are not being used by other applications")

if __name__ == "__main__":
    list_audio_devices()
