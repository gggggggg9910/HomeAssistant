#!/usr/bin/env python3
"""
Script to download CosyVoice2-0.5B model from modelscope
"""

from modelscope import snapshot_download

def main():
    print("Starting download of CosyVoice2-0.5B model...")
    try:
        #snapshot_download('iic/CosyVoice2-0.5B', local_dir='/home/wudixin/models/CosyVoice2-0.5B')
        print("Download completed successfully!")
        print("Testing the downloaded model...")

        from modelscope.pipelines import pipeline
        tts = pipeline('text-to-speech', model='/home/wudixin/models/CosyVoice2-0.5B')
        audio = tts("你好，这是通过 ModelScope 下载的 CosyVoice2！")
        # 保存音频
        import soundfile as sf
        sf.write('output.wav', audio['audio'], audio['sampling_rate'])
        print("Test audio saved as output.wav")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
