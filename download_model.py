#!/usr/bin/env python3
"""
Script to download CosyVoice2-0.5B model from modelscope
"""

from modelscope import snapshot_download

def main():
    print("Starting download of CosyVoice2-0.5B model...")
    try:
        snapshot_download('iic/CosyVoice2-0.5B', local_dir='~/.models/iic/CosyVoice2-0.5')
        print("Download completed successfully!")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    main()
