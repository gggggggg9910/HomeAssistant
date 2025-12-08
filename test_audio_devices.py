#!/usr/bin/env python3
"""
Enhanced audio device diagnostic script for HomeAssistant.
Helps identify available audio input/output devices and test functionality.
"""

import logging
import sys
import time
import numpy as np

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
    """List all available audio devices with detailed information."""
    if not PYAUDIO_AVAILABLE:
        return

    audio = pyaudio.PyAudio()

    print("=" * 70)
    print("ENHANCED AUDIO DEVICES DIAGNOSTIC")
    print("=" * 70)

    print(f"PyAudio version: {pyaudio.__version__}")
    print(f"PortAudio version: {pyaudio.get_portaudio_version_text()}")
    print()

    # Get device count
    device_count = audio.get_device_count()
    print(f"Total audio devices found: {device_count}")
    print()

    # Categorize devices
    input_devices = []
    output_devices = []
    duplex_devices = []
    duplex_input_devices = []  # 双工设备也算作输入设备
    duplex_output_devices = [] # 双工设备也算作输出设备

    # List all devices with detailed info
    print("ALL DEVICES:")
    print("-" * 50)
    for i in range(device_count):
        try:
            device_info = audio.get_device_info_by_index(i)
            name = device_info['name']
            input_ch = device_info['maxInputChannels']
            output_ch = device_info['maxOutputChannels']
            sample_rate = device_info['defaultSampleRate']

            print(f"Device {i}: {name}")
            print(f"  Input Channels: {input_ch}, Output Channels: {output_ch}")
            print(f"  Default Sample Rate: {sample_rate} Hz")
            print(f"  Host API: {device_info.get('hostApi', 'Unknown')}")

            # Special handling for Hikvision devices - they may report 0 output channels but actually support output
            is_hikvision = 'hikvision' in name.lower() or '2k usb camera' in name.lower() or 'hw:2,0' in name.lower()

            # Categorize
            if (input_ch > 0 and output_ch > 0) or (is_hikvision and input_ch > 0):
                duplex_devices.append(i)
                duplex_input_devices.append(i)  # 双工设备也算输入设备
                duplex_output_devices.append(i) # 双工设备也算输出设备
                print(f"  Type: DUPLEX (输入输出)")
                if is_hikvision:
                    print(f"  Recommended: Input 16000Hz, Output 48000Hz (海康威视设备)")
                    print(f"  Note: Device reports {output_ch} output channels but actually supports output")
                else:
                    print(f"  Recommended: Input 16000Hz, Output 48000Hz")
            elif input_ch > 0:
                input_devices.append(i)
                print(f"  Type: INPUT ONLY (仅输入 - 麦克风)")
            elif output_ch > 0:
                output_devices.append(i)
                print(f"  Type: OUTPUT ONLY (仅输出 - 扬声器)")
            else:
                print("  Type: UNKNOWN (未知类型)")
            print()
        except Exception as e:
            print(f"Device {i}: Error getting info - {e}")
            print()

    # Get default devices
    print("DEFAULT DEVICES:")
    print("-" * 50)
    try:
        default_input = audio.get_default_input_device_info()
        print(f"Default Input Device: {default_input['name']} (Index: {default_input['index']})")
        print(f"  Max Input Channels: {default_input['maxInputChannels']}")
        print(f"  Default Sample Rate: {default_input['defaultSampleRate']} Hz")
    except Exception as e:
        print(f"Default Input Device: ERROR - {e}")

    print()

    try:
        default_output = audio.get_default_output_device_info()
        print(f"Default Output Device: {default_output['name']} (Index: {default_output['index']})")
        print(f"  Max Output Channels: {default_output['maxOutputChannels']}")
        print(f"  Default Sample Rate: {default_output['defaultSampleRate']} Hz")
    except Exception as e:
        print(f"Default Output Device: ERROR - {e}")

    print()
    print("=" * 70)
    print("DEVICE SUMMARY:")
    print("-" * 50)
    # Combine regular devices with duplex devices for totals
    all_input_devices = input_devices + duplex_input_devices
    all_output_devices = output_devices + duplex_output_devices

    print(f"Input devices (microphones): {len(all_input_devices)} - {[f'Device {i}' for i in all_input_devices]}")
    print(f"Output devices (speakers): {len(all_output_devices)} - {[f'Device {i}' for i in all_output_devices]}")
    print(f"Duplex devices: {len(duplex_devices)} - {[f'Device {i}' for i in duplex_devices]}")
    print()

    # Store device info before terminating audio
    device_infos = {}
    for i in range(device_count):
        try:
            device_infos[i] = audio.get_device_info_by_index(i)
        except:
            device_infos[i] = {'name': f'Unknown Device {i}', 'maxInputChannels': 0, 'maxOutputChannels': 0}

    # Test devices individually
    print("INDIVIDUAL DEVICE TESTS:")
    print("-" * 50)

    for device_idx in range(device_count):
        device_info = device_infos[device_idx]
        name = device_info['name']
        input_ch = device_info['maxInputChannels']
        output_ch = device_info['maxOutputChannels']
        sample_rate = device_info['defaultSampleRate']

        print(f"Testing Device {device_idx}: {name}")

        # Test input
        if input_ch > 0:
            try:
                # Try different sample rates
                test_rates = [16000, 44100, 48000, int(sample_rate)]
                input_works = False
                working_rate = None

                for rate in test_rates:
                    try:
                        test_stream = audio.open(
                            format=pyaudio.paInt16,
                            channels=min(1, input_ch),
                            rate=rate,
                            input=True,
                            input_device_index=device_idx,
                            frames_per_buffer=1024
                        )
                        test_stream.close()
                        input_works = True
                        working_rate = rate
                        break
                    except:
                        continue

                if input_works:
                    print(f"  ✓ Input: WORKS at {working_rate} Hz")
                else:
                    print(f"  ✗ Input: FAILED - tried rates {test_rates}")
            except Exception as e:
                print(f"  ✗ Input: FAILED - {e}")
        else:
            print("  - Input: Not supported")

        # Test output
        # Special handling: test output even if device reports 0 output channels (for Hikvision devices)
        device_name_lower = name.lower()
        should_test_output = (output_ch > 0) or ('hikvision' in device_name_lower or
                                               '2k usb camera' in device_name_lower or
                                               'hw:2,0' in device_name_lower)

        if should_test_output:
            try:
                # Try different sample rates
                test_rates = [16000, 44100, 48000, int(sample_rate)]
                output_works = False
                working_rate = None

                for rate in test_rates:
                    try:
                        test_stream = audio.open(
                            format=pyaudio.paInt16,
                            channels=min(1, output_ch),
                            rate=rate,
                            output=True,
                            output_device_index=device_idx,
                            frames_per_buffer=1024
                        )
                        test_stream.close()
                        output_works = True
                        working_rate = rate
                        break
                    except:
                        continue

                if output_works:
                    print(f"  ✓ Output: WORKS at {working_rate} Hz")
                else:
                    print(f"  ✗ Output: FAILED - tried rates {test_rates}")
            except Exception as e:
                print(f"  ✗ Output: FAILED - {e}")
        else:
            print("  - Output: Not supported")

        print()

    audio.terminate()

    print("RECOMMENDATIONS:")
    print("-" * 50)

    if len(all_input_devices) == 0:
        print("❌ 没有找到麦克风设备！")
        print("   • 检查麦克风是否正确连接")
        print("   • Windows: 设备管理器 → 音频输入和输出")
        print("   • Linux: 运行 'arecord -l' 检查")
        print("   • macOS: 系统偏好设置 → 声音")
        print("   • 尝试重新插拔USB麦克风或使用蓝牙麦克风")
    else:
        print(f"✓ 找到 {len(all_input_devices)} 个麦克风设备")
        print("  建议使用的设备索引:")
        for idx in all_input_devices:
            device_info = device_infos[idx]
            print(f"    input_device: {idx}  # {device_info['name']}")

    print()

    if len(all_output_devices) == 0:
        print("❌ 没有找到扬声器设备！")
        print("   • 检查扬声器是否正确连接")
        print("   • 检查音频输出设置")
    else:
        print(f"✓ 找到 {len(all_output_devices)} 个扬声器设备")
        print("  建议使用的设备索引:")
        for idx in all_output_devices:
            device_info = device_infos[idx]
            print(f"    output_device: {idx}  # {device_info['name']}")

    print()

    if len(all_input_devices) > 0 and len(all_output_devices) > 0:
        print("✓ 麦克风和扬声器都可用！")
        print("  更新你的配置文件 config/settings.py 或 .env 文件:")
        print("  ")
        print("  # 示例配置")
        if all_input_devices:
            print(f"  audio__input_device={all_input_devices[0]}")
        if all_output_devices:
            print(f"  audio__output_device={all_output_devices[0]}")
        print("  ")
        print("  或者直接在 settings.py 中设置:")
        print("  input_device: int = 你的麦克风设备索引")
        print("  output_device: int = 你的扬声器设备索引")
    elif len(all_input_devices) == 0 or len(all_output_devices) == 0:
        print("❌ 音频设置不完整，无法正常使用语音助手")
    else:
        print("✓ 音频设置完整，可以正常使用语音助手！")

def test_audio_recording(device_index=None, duration=3):
    """Test actual audio recording and playback."""
    if not PYAUDIO_AVAILABLE:
        return

    print(f"\n{'='*50}")
    print(f"AUDIO RECORDING TEST (Device {device_index})")
    print(f"{'='*50}")

    audio = pyaudio.PyAudio()

    try:
        # Test recording
        print("Recording 3 seconds of audio...")
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )

        frames = []
        for _ in range(0, int(16000 / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)

        stream.close()
        print("✓ Recording completed")

        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0

        print(f"Recorded {len(audio_data)} samples")
        print(f"Audio duration: {len(audio_data)/16000:.2f} seconds")
        print(f"Peak amplitude: {np.max(np.abs(audio_data)):.4f}")

        # Test playback
        if device_index is not None:
            print("\nPlaying back recorded audio...")
            output_stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True,
                output_device_index=device_index
            )

            # Convert back to int16
            playback_data = (audio_data * 32767).astype(np.int16)

            # Write in chunks
            chunk_size = 1024
            for i in range(0, len(playback_data), chunk_size):
                chunk = playback_data[i:i+chunk_size]
                output_stream.write(chunk.tobytes())

            output_stream.close()
            print("✓ Playback completed")

    except Exception as e:
        print(f"✗ Audio test failed: {e}")
    finally:
        audio.terminate()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test specific device
        device_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
        test_audio_recording(device_idx)
    else:
        # Default: list all devices
        list_audio_devices()
