#!/usr/bin/env python3
"""
Script to test if CosyVoice2-0.5B model can be used
"""
import traceback

def load_wav_safe(wav, target_sr):
    """使用 soundfile 直接加载音频，避免 TorchCodec 依赖"""
    import soundfile as sf
    import torch
    
    # 使用 soundfile 直接加载
    speech, sample_rate = sf.read(wav, dtype='float32')
    
    # 转换为 torch tensor
    if len(speech.shape) == 1:
        speech = speech.reshape(1, -1)
    else:
        speech = speech.T  # (channels, samples)
    
    speech = torch.from_numpy(speech)
    
    # 转换为单声道
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    
    # 重采样
    if sample_rate != target_sr:
        assert sample_rate > target_sr, f'wav sample rate {sample_rate} must be greater than {target_sr}'
        import torchaudio
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    
    return speech

def main():
    print("Testing CosyVoice2-0.5B model availability...")
    try:
        model_dir = '/home/wudixin/models/iic/CosyVoice2-0.5B'
        # Test basic imports
        print("Testing imports...")
        import sys
        sys.path.insert(0, '/home/wudixin/models/CosyVoice')
        sys.path.append('/home/wudixin/models/CosyVoice/third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
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

        # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
        # zero_shot usage
        import os
        prompt_file = '/home/wudixin/models/CosyVoice/asset/zero_shot_prompt.wav'
        if not os.path.exists(prompt_file):
            print(f"⚠️  Warning: Prompt file '{prompt_file}' not found. Skipping inference tests.")
            print("   You can download a sample prompt file from the CosyVoice2 repository.")
            return
        
        print("\nTesting zero_shot inference...")
        try:
            # 使用自定义的 load_wav_safe 函数
            prompt_speech_16k = load_wav_safe(prompt_file, 16000)
            for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
                torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
            print("✓ Zero-shot inference test successful")
        except Exception as e:
            print(f"❌ Zero-shot inference failed: {e}")
            traceback.print_exc()

        # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
        print("Testing cross_lingual inference...")
        try:
            for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
                torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
            print("✓ Cross-lingual inference test successful")
        except Exception as e:
            print(f"❌ Cross-lingual inference failed: {e}")
            traceback.print_exc()

        # instruct usage
        print("Testing instruct inference...")
        try:
            for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
                torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
            print("✓ Instruct inference test successful")
        except Exception as e:
            print(f"❌ Instruct inference failed: {e}")
            traceback.print_exc()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check if CosyVoice2 is properly installed.")
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your model path and installation.")
        traceback.print_exc()

if __name__ == "__main__":
    main()
