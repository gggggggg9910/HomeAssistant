#!/usr/bin/env python3
"""
Script to test if CosyVoice2-0.5B model can be used
"""

def main():
    print("Testing CosyVoice2-0.5B model availability...")
    try:
        #from modelscope import snapshot_download
        #model_dir = snapshot_download('iic/CosyVoice2-0.5B', local_dir='/home/wudixin/models/iic/CosyVoice2-0.5B')
        #print(f"✓ Model downloaded to: {model_dir}")
        model_dir = '/home/wudixin/models/iic/CosyVoice2-0.5B'
        # Test basic imports
        print("Testing imports...")
        import sys
        sys.path.insert(0, '/home/wudixin/models/CosyVoice')
        sys.path.append('/home/wudixin/models/CosyVoice/third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
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

        # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
        # zero_shot usage
        prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
        for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
            torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

        # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
        for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
            torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

        # instruct usage
        for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
            torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check if CosyVoice2 is properly installed.")
        traceback.print_exc()  # 打印完整堆栈
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your model path and installation.")
        traceback.print_exc()  # 打印完整堆栈
if __name__ == "__main__":
    main()
