#!/usr/bin/env python3
"""
Home Assistant - Voice-controlled AI assistant for Raspberry Pi.
"""
import asyncio
import logging
import os  # 添加 os 模块导入
import signal
import sys
from pathlib import Path

# 设置ALSA环境变量强制使用海康威视摄像头设备 (card 2, device 0)
os.environ['AUDIODEV'] = 'hw:2,0'

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings, initialize_logging
from core.controller import VoiceAssistantController, AssistantState


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, shutting down...")
    if 'controller' in globals():
        asyncio.create_task(controller.stop())


async def main():
    """Main application entry point."""
    global controller

    logger = logging.getLogger(__name__)
    logger.info("Starting Home Assistant...")

    try:
        # Initialize controller
        controller = VoiceAssistantController(settings)

        # Add state change callback
        def on_state_change(state: AssistantState):
            logger.info(f"Assistant state: {state.value}")

        controller.add_state_callback(on_state_change)

        # Initialize all components
        if not await controller.initialize():
            logger.error("Failed to initialize controller")
            return 1

        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Home Assistant started successfully!")
        logger.info(f"Wake word: '{settings.kws.keyword}'")
        logger.info("Press Ctrl+C to exit")

        # Start the assistant
        await controller.start()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        if 'controller' in globals():
            await controller.cleanup()

    return 0


if __name__ == "__main__":
    initialize_logging()
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
