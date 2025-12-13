#!/usr/bin/env python3
"""
Script to test if CosyVoice2-0.5B model can be used
"""

def main():
    print("Testing CosyVoice2-0.5B model availability...")
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download('iic/CosyVoice2-0.5B', local_dir='/home/wudixin/models/iic/CosyVoice2-0.5B')
        print(f"✓ Model downloaded to: {model_dir}")
        # Test basic imports
        print("Testing imports...")
        import sys
        sys.path.append('/home/wudixin/models/CosyVoice/third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav
        import torchaudio

        print("✓ CosyVoice2 import successful")

        # Test model loading
        print("Testing model loading...")
        cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
        print("✓ Model loaded successfully")

        # Test basic functionality
        print("Testing basic functionality...")
        print(f"✓ Sample rate: {cosyvoice.sample_rate}")
        print("✓ CosyVoice2 is ready to use!")

        print("\nModel test completed successfully!")
        print("You can now use CosyVoice2 for text-to-speech synthesis.")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check if CosyVoice2 is properly installed.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your model path and installation.")

if __name__ == "__main__":
    main()
