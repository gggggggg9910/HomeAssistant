#!/usr/bin/env python3
"""
Test script for CosyVoice2-0.5B TTS loading
Based on modelscope.cn/models/iic/CosyVoice2-0.5B usage examples
"""

import logging
logging.basicConfig(level=logging.INFO)

def test_cosyvoice_loading():
    """Test CosyVoice2-0.5B loading following modelscope examples"""
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        print("Testing CosyVoice2-0.5B loading (based on modelscope examples)...")

        # Create pipeline - following modelscope pattern
        pipe = pipeline(
            task=Tasks.text_to_speech,
            model='iic/CosyVoice2-0.5B'
        )

        print("✓ CosyVoice2-0.5B pipeline created successfully")

        # Test inference - using the format from modelscope examples
        result = pipe({'text': '你好，这是一个测试'})
        print("✓ CosyVoice inference successful")
        print(f"Result type: {type(result)}")

        if isinstance(result, dict):
            print(f"Result keys: {result.keys()}")
            # Check for common output keys
            if 'output' in result:
                audio = result['output']
                print(f"Audio output type: {type(audio)}")
                if hasattr(audio, 'shape'):
                    print(f"Audio shape: {audio.shape}")
        elif hasattr(result, 'numpy'):
            print(f"Audio shape: {result.numpy().shape}")
        elif isinstance(result, list):
            print(f"Result length: {len(result)}")

        return True

    except Exception as e:
        print(f"✗ CosyVoice loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cosyvoice_loading()
