#!/usr/bin/env python3
"""
Test script to verify sherpa-onnx import and model loading.
"""
import sys
from pathlib import Path

def test_sherpa_import():
    """Test sherpa-onnx import."""
    print("Testing sherpa-onnx import...")

    # Try standard import first
    try:
        import sherpa_onnx
        print("✓ sherpa-onnx imported from standard path")
        return True, sherpa_onnx
    except ImportError:
        print("✗ sherpa-onnx not found in standard path, trying custom path...")

    # Try custom path
    try:
        sherpa_path = Path.home() / "models" / "sherpa-onnx"
        print(f"Trying path: {sherpa_path}")

        if sherpa_path.exists():
            sys.path.insert(0, str(sherpa_path))
            sys.path.insert(0, str(sherpa_path / "lib"))
            sys.path.insert(0, str(sherpa_path / "python"))

            import sherpa_onnx
            print("✓ sherpa-onnx imported from custom path")
            return True, sherpa_onnx
        else:
            print(f"✗ Custom path does not exist: {sherpa_path}")
            return False, None

    except ImportError as e:
        print(f"✗ Failed to import sherpa-onnx: {e}")
        return False, None

def test_model_path():
    """Test model path."""
    print("\nTesting model path...")

    model_path = Path.home() / "models" / "kws-onnx"
    print(f"Model path: {model_path}")

    if model_path.exists():
        print("✓ Model directory exists")

        # Check for model files
        encoder_files = list(model_path.glob("*encoder*.onnx"))
        decoder_files = list(model_path.glob("*decoder*.onnx"))
        joiner_files = list(model_path.glob("*joiner*.onnx"))

        print(f"Encoder files: {len(encoder_files)}")
        print(f"Decoder files: {len(decoder_files)}")
        print(f"Joiner files: {len(joiner_files)}")

        if encoder_files:
            print(f"First encoder: {encoder_files[0]}")
        if decoder_files:
            print(f"First decoder: {decoder_files[0]}")
        if joiner_files:
            print(f"First joiner: {joiner_files[0]}")

        return len(encoder_files) > 0 and len(decoder_files) > 0 and len(joiner_files) > 0
    else:
        print("✗ Model directory does not exist")
        return False

if __name__ == "__main__":
    print("Sherpa-onnx Test")
    print("=" * 40)

    sherpa_ok, sherpa_module = test_sherpa_import()
    model_ok = test_model_path()

    print("\n" + "=" * 40)
    print("Results:")
    print(f"Sherpa-onnx: {'✓' if sherpa_ok else '✗'}")
    print(f"Model files: {'✓' if model_ok else '✗'}")

    if sherpa_ok and model_ok:
        print("\n✓ All tests passed! Ready to run main.py")
    else:
        print("\n✗ Some tests failed. Please check installation.")
