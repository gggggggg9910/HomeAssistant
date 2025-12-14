#!/usr/bin/env python3
"""
Test script for Piper TTS on Raspberry Pi 5.

Usage:
    python test_piper_tts.py [--text "你好"] [--voice zh_CN_huayan_medium]
"""

import asyncio
import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def main():
    parser = argparse.ArgumentParser(description='Test Piper TTS')
    parser.add_argument('--text', '-t', type=str, default="你好，我是你的智能助手，有什么可以帮助你的？",
                        help='Text to synthesize')
    parser.add_argument('--voice', '-v', type=str, default="zh_CN_huayan_medium",
                        help='Voice ID to use')
    parser.add_argument('--model-dir', '-m', type=str, default="~/models/piper",
                        help='Model directory path')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='Speech speed (1.0 = normal)')
    parser.add_argument('--volume', type=float, default=0.9,
                        help='Output volume (0.0 - 1.0)')
    parser.add_argument('--benchmark', '-b', action='store_true',
                        help='Run benchmark tests')
    args = parser.parse_args()

    print("=" * 60)
    print("🎤 Piper TTS 测试工具 (树莓派5优化版)")
    print("=" * 60)
    
    # Import TTS module
    try:
        from core.tts import PiperTTSConfig, TextToSpeechPiper
        print("✅ 成功导入 Piper TTS 模块")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("\n请确保已安装 piper-tts:")
        print("  pip install piper-tts")
        return 1
    
    # Create config
    config = PiperTTSConfig(
        model_dir=args.model_dir,
        voice=args.voice,
        length_scale=args.speed,
        volume=args.volume,
    )
    
    print(f"\n📁 模型目录: {os.path.expanduser(config.model_dir)}")
    print(f"🗣️  声音: {config.voice}")
    print(f"⚡ 语速: {config.length_scale}")
    print(f"🔊 音量: {config.volume}")
    
    # Initialize TTS
    print("\n" + "-" * 40)
    print("正在初始化 Piper TTS...")
    
    tts = TextToSpeechPiper(config)
    
    start_time = time.time()
    if not await tts.initialize():
        print("\n❌ 初始化失败！")
        print("\n请按照以下步骤下载模型:")
        print(f"  mkdir -p {os.path.expanduser(config.model_dir)}")
        print(f"  cd {os.path.expanduser(config.model_dir)}")
        print("  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx")
        print("  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json")
        return 1
    
    init_time = time.time() - start_time
    print(f"✅ 初始化成功！耗时: {init_time:.2f}秒")
    
    # Run benchmark if requested
    if args.benchmark:
        await run_benchmark(tts)
    else:
        # Single synthesis test
        print("\n" + "-" * 40)
        print(f"📝 合成文本: '{args.text}'")
        print("-" * 40)
        
        start_time = time.time()
        success = await tts.speak_text(args.text)
        synthesis_time = time.time() - start_time
        
        if success:
            print(f"\n✅ 合成成功！耗时: {synthesis_time:.2f}秒")
            
            # Calculate RTF
            text_len = len(args.text)
            estimated_duration = text_len * 0.15  # ~0.15秒/字
            rtf = synthesis_time / estimated_duration if estimated_duration > 0 else 0
            print(f"📊 估计RTF: {rtf:.2f} (< 1.0 表示实时)")
        else:
            print(f"\n❌ 合成失败！")
            return 1
    
    # Cleanup
    print("\n" + "-" * 40)
    print("清理资源...")
    await tts.cleanup()
    print("✅ 完成！")
    
    return 0


async def run_benchmark(tts):
    """Run benchmark tests."""
    print("\n" + "=" * 60)
    print("📊 运行性能基准测试")
    print("=" * 60)
    
    test_cases = [
        ("短文本 (10字)", "你好，欢迎回家。"),
        ("中等文本 (30字)", "今天天气真不错，阳光明媚，温度适宜，非常适合出去散步。"),
        ("长文本 (60字)", "我是你的智能家居助手，我可以帮你控制家里的灯光、空调、窗帘等设备。你可以说打开客厅的灯，或者说把空调温度调到26度。"),
        ("超长文本 (100字)", "人工智能正在改变我们的生活方式。从智能家居到自动驾驶，从医疗诊断到金融分析，AI技术已经渗透到各个领域。语音合成技术作为人机交互的重要组成部分，让机器能够用自然的声音与人类交流，极大地提升了用户体验。"),
    ]
    
    results = []
    
    for name, text in test_cases:
        print(f"\n🔄 测试: {name}")
        print(f"   文本: '{text[:30]}...' ({len(text)}字)")
        
        # Warm up
        await tts.synthesize_speech(text[:5])
        
        # Benchmark
        times = []
        for i in range(3):
            start_time = time.time()
            audio = await tts.synthesize_speech(text)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if audio is not None:
                audio_duration = len(audio) / tts.config.sample_rate
            else:
                audio_duration = 0
        
        avg_time = sum(times) / len(times)
        rtf = avg_time / audio_duration if audio_duration > 0 else 0
        
        results.append({
            'name': name,
            'chars': len(text),
            'time': avg_time,
            'audio_duration': audio_duration,
            'rtf': rtf,
        })
        
        print(f"   ⏱️  平均耗时: {avg_time:.2f}秒")
        print(f"   🎵 音频时长: {audio_duration:.2f}秒")
        print(f"   📈 RTF: {rtf:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 基准测试结果汇总")
    print("=" * 60)
    print(f"{'测试项':<20} {'字数':<8} {'耗时(秒)':<10} {'音频(秒)':<10} {'RTF':<8}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['name']:<20} {r['chars']:<8} {r['time']:<10.2f} {r['audio_duration']:<10.2f} {r['rtf']:<8.2f}")
    
    avg_rtf = sum(r['rtf'] for r in results) / len(results)
    print("-" * 60)
    print(f"{'平均RTF':<48} {avg_rtf:<8.2f}")
    
    if avg_rtf < 0.5:
        print("\n🚀 性能评级: 优秀 (实时性很好)")
    elif avg_rtf < 1.0:
        print("\n✅ 性能评级: 良好 (可以实时)")
    else:
        print("\n⚠️  性能评级: 一般 (有延迟)")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
