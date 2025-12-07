"""
Main controller for voice assistant coordinating all components.
"""
import asyncio
import logging
import signal
import sys
from enum import Enum
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class AssistantState(Enum):
    """Voice assistant states."""
    IDLE = "idle"
    LISTENING_FOR_KEYWORD = "listening_for_keyword"
    RECOGNIZING_SPEECH = "recognizing_speech"
    PROCESSING_REQUEST = "processing_request"
    SPEAKING_RESPONSE = "speaking_response"
    ERROR = "error"


class VoiceAssistantController:
    """Main controller for voice assistant."""

    def __init__(self, config):
        self.config = config
        self.state = AssistantState.IDLE
        self.state_callbacks: list[Callable[[AssistantState], None]] = []

        # Component instances
        self.audio_manager = None
        self.keyword_spotter = None
        self.speech_recognizer = None
        self.tts_engine = None
        self.llm_client = None

        # Control flags
        self.running = False
        self._shutdown_event = asyncio.Event()

        # Current audio buffer for speech recognition
        self._speech_audio_buffer = []

    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("Initializing voice assistant controller...")

            # Initialize audio manager
            from .audio import AudioManager, AudioConfig
            audio_config = AudioConfig(
                sample_rate=self.config.audio.sample_rate,
                channels=self.config.audio.channels,
                chunk_size=self.config.audio.chunk_size,
                input_device=self.config.audio.input_device,
                output_device=self.config.audio.output_device
            )
            self.audio_manager = AudioManager(audio_config)
            if not await self.audio_manager.initialize():
                logger.error("Failed to initialize audio manager")
                return False

            # Initialize keyword spotter
            from .kws import KeywordSpotter, KWSConfig
            kws_config = KWSConfig(
                model_path=str(self.config.kws.model_path),
                keyword=self.config.kws.keyword,
                threshold=self.config.kws.threshold,
                max_wait_seconds=self.config.kws.max_wait_seconds,
                sample_rate=self.config.audio.sample_rate
            )
            self.keyword_spotter = KeywordSpotter(kws_config)
            if not await self.keyword_spotter.initialize():
                logger.error("Failed to initialize keyword spotter")
                return False

            # Initialize speech recognizer
            from .asr import SpeechRecognizer, ASRConfig
            asr_config = ASRConfig(
                model_path=str(self.config.asr.model_path),
                language=self.config.asr.language,
                max_wait_seconds=self.config.asr.max_wait_seconds,
                sample_rate=self.config.audio.sample_rate,
                disable_update=self.config.asr.disable_update
            )
            self.speech_recognizer = SpeechRecognizer(asr_config)
            if not await self.speech_recognizer.initialize():
                logger.warning("Failed to initialize speech recognizer - speech recognition will be disabled")
                logger.warning("This may be due to insufficient memory. Consider using a smaller model or disabling ASR.")
                self.speech_recognizer = None  # Allow system to continue without ASR

            # Initialize TTS engine
            from .tts import TextToSpeech, TTSConfig
            tts_config = TTSConfig(
                model_id=self.config.tts.model_id,
                model_path=self.config.tts.model_path if self.config.tts.model_path else None,
                voice=self.config.tts.voice,
                speed=self.config.tts.speed,
                volume=self.config.tts.volume
            )
            self.tts_engine = TextToSpeech(tts_config)
            if not await self.tts_engine.initialize():
                logger.error("Failed to initialize TTS engine")
                return False

            # Initialize LLM client
            from .llm import LLMClient, LLMConfig
            llm_config = LLMConfig(
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout=self.config.llm.timeout,
                use_local=self.config.llm.use_local,
                system_prompt="你是一个智能家居助手，可以帮助用户处理各种任务。请用中文回答用户的问题，保持简洁友好的语气。"
            )
            self.llm_client = LLMClient(llm_config)
            if not await self.llm_client.initialize():
                logger.warning("Failed to initialize LLM client - LLM features will be disabled")
                logger.warning("To enable LLM, configure DASHSCOPE_API_KEY or set LLM_USE_LOCAL=true")
                self.llm_client = None  # Disable LLM functionality
            else:
                # Test LLM connection
                if not await self.llm_client.test_connection():
                    logger.warning("LLM API connection test failed, but continuing...")

            logger.info("Voice assistant controller initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize controller: {e}")
            return False

    async def cleanup(self):
        """Clean up all components."""
        logger.info("Cleaning up voice assistant controller...")

        # Clean up in reverse order
        if self.llm_client:
            await self.llm_client.cleanup()
        if self.tts_engine:
            await self.tts_engine.cleanup()
        if self.speech_recognizer:
            await self.speech_recognizer.cleanup()
        if self.keyword_spotter:
            await self.keyword_spotter.cleanup()
        if self.audio_manager:
            await self.audio_manager.cleanup()

        self.running = False
        self._shutdown_event.set()
        logger.info("Voice assistant controller cleaned up")

    def add_state_callback(self, callback: Callable[[AssistantState], None]):
        """Add callback for state changes."""
        self.state_callbacks.append(callback)

    def _set_state(self, new_state: AssistantState):
        """Set new state and notify callbacks."""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.info(f"State changed: {old_state.value} -> {new_state.value}")

            # Notify callbacks
            for callback in self.state_callbacks:
                try:
                    callback(new_state)
                except Exception as e:
                    logger.error(f"Error in state callback: {e}")

    async def start(self):
        """Start the voice assistant main loop."""
        if not self.audio_manager or not self.audio_manager.is_initialized():
            logger.error("Controller not properly initialized")
            return

        self.running = True
        self._shutdown_event.clear()
        logger.info("Voice assistant started")

        try:
            while self.running and not self._shutdown_event.is_set():
                try:
                    # Main assistant loop
                    await self._run_assistant_loop()
                except Exception as e:
                    logger.error(f"Error in assistant loop: {e}")
                    self._set_state(AssistantState.ERROR)
                    await asyncio.sleep(1)  # Brief pause before retry

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            await self.cleanup()

    async def stop(self):
        """Stop the voice assistant."""
        logger.info("Stopping voice assistant...")
        self.running = False
        self._shutdown_event.set()

    async def _run_assistant_loop(self):
        """Main assistant conversation loop."""
        while self.running:
            try:
                # Phase 1: Listen for keyword
                self._set_state(AssistantState.LISTENING_FOR_KEYWORD)
                keyword_detected = await self._listen_for_keyword()

                if not keyword_detected:
                    continue  # Continue listening

                # Phase 2: Recognize speech
                self._set_state(AssistantState.RECOGNIZING_SPEECH)
                user_text = await self._recognize_speech()

                if not user_text:
                    # Play error sound or message
                    await self._play_error_message("没有听到您的语音，请重试")
                    continue

                logger.info(f"Recognized speech: {user_text}")

                # Phase 3: Process with LLM
                self._set_state(AssistantState.PROCESSING_REQUEST)
                ai_response = await self._process_with_llm(user_text)

                if not ai_response:
                    await self._play_error_message("处理请求时出现错误")
                    continue

                # Phase 4: Speak response
                self._set_state(AssistantState.SPEAKING_RESPONSE)
                await self._speak_response(ai_response)

                # Brief pause before listening again
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in assistant loop iteration: {e}")
                self._set_state(AssistantState.ERROR)
                await asyncio.sleep(1)

    async def _listen_for_keyword(self) -> bool:
        """Listen for keyword activation."""
        try:
            logger.info(f"Listening for keyword: '{self.config.kws.keyword}'")

            # Create callbacks
            def audio_callback(audio_chunk: np.ndarray):
                # Pass audio to keyword spotter
                self.keyword_spotter.add_audio_chunk(audio_chunk)

            def keyword_callback(keyword: str):
                logger.info(f"Keyword detected: {keyword}")

            # Start audio listening
            audio_task = asyncio.create_task(
                self.audio_manager.start_listening(audio_callback)
            )

            # Listen for keyword with timeout
            keyword_task = asyncio.create_task(
                self.keyword_spotter.listen_for_keyword(
                    audio_callback=lambda chunk: None,  # Already handled above
                    keyword_callback=keyword_callback,
                    timeout_seconds=self.config.kws.max_wait_seconds
                )
            )

            # Wait for either keyword detection or timeout
            done, pending = await asyncio.wait(
                [keyword_task],
                timeout=self.config.kws.max_wait_seconds,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Stop audio listening
            await self.audio_manager.stop_listening()

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check if keyword was detected
            if keyword_task in done:
                result = keyword_task.result()
                return result is True
            else:
                logger.info("Keyword listening timeout")
                return False

        except Exception as e:
            logger.error(f"Error listening for keyword: {e}")
            return False

    async def _recognize_speech(self) -> Optional[str]:
        """Recognize speech after keyword detection."""
        try:
            logger.info("Starting speech recognition...")

            # Clear previous buffer
            self._speech_audio_buffer.clear()

            # Create audio callback
            def audio_callback(audio_chunk: np.ndarray):
                self._speech_audio_buffer.append(audio_chunk.copy())
                self.speech_recognizer.add_audio_chunk(audio_chunk)

            # Start audio listening
            audio_task = asyncio.create_task(
                self.audio_manager.start_listening(audio_callback)
            )

            # Start speech recognition
            recognition_task = asyncio.create_task(
                self.speech_recognizer.start_continuous_recognition(
                    audio_callback=lambda chunk: None,  # Already handled
                    timeout_seconds=self.config.asr.max_wait_seconds
                )
            )

            # Wait for recognition to complete or timeout
            done, pending = await asyncio.wait(
                [recognition_task],
                timeout=self.config.asr.max_wait_seconds,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Stop audio listening
            await self.audio_manager.stop_listening()

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Get recognition result
            if recognition_task in done:
                text = recognition_task.result()
                return text
            else:
                logger.info("Speech recognition timeout")
                return None

        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            return None

    async def _process_with_llm(self, user_text: str) -> Optional[str]:
        """Process user input with LLM."""
        if not self.llm_client:
            logger.warning("LLM client not available")
            return "抱歉，AI对话功能暂时不可用。请检查API配置。"

        try:
            logger.info(f"Processing with LLM: {user_text}")
            response = await self.llm_client.generate_response(user_text)

            if response:
                logger.info(f"LLM response: {response}")
                return response
            else:
                logger.error("No response from LLM")
                return "抱歉，我现在无法回答您的问题。"

        except Exception as e:
            logger.error(f"Error processing with LLM: {e}")
            return "处理请求时出现错误，请稍后再试。"

    async def _speak_response(self, text: str):
        """Speak the AI response."""
        try:
            logger.info(f"Speaking response: {text}")
            success = await self.tts_engine.speak_text(text)

            if not success:
                logger.error("Failed to speak response")
                # Could play a beep or error sound here

        except Exception as e:
            logger.error(f"Error speaking response: {e}")

    async def _play_error_message(self, message: str):
        """Play an error message."""
        try:
            await self._speak_response(message)
        except Exception as e:
            logger.error(f"Error playing error message: {e}")

    def get_state(self) -> AssistantState:
        """Get current assistant state."""
        return self.state

    async def manual_input(self, text: str) -> Optional[str]:
        """Process manual text input (for testing/debugging)."""
        try:
            self._set_state(AssistantState.PROCESSING_REQUEST)
            response = await self._process_with_llm(text)

            if response:
                self._set_state(AssistantState.SPEAKING_RESPONSE)
                await self._speak_response(response)
                self._set_state(AssistantState.IDLE)

            return response
        except Exception as e:
            logger.error(f"Error processing manual input: {e}")
            self._set_state(AssistantState.ERROR)
            return None
