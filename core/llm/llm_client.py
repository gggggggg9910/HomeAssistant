"""
LLM integration for AI dialogue using Alibaba Qwen models.
"""
import asyncio
import logging
from typing import Optional, List, Dict, Any
import json

try:
    from dashscope import Generation
    import dashscope
    DASHScope_AVAILABLE = True
except ImportError:
    DASHScope_AVAILABLE = False
    Generation = None
    dashscope = None

try:
    import torch
    from modelscope import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MODELScope_AVAILABLE = True
except ImportError:
    MODELScope_AVAILABLE = False
    torch = None
    snapshot_download = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

logger = logging.getLogger(__name__)


class LLMConfig:
    """Configuration for Alibaba Qwen LLM integration."""
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        model: str = "qwen-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30,
        use_local: bool = False,
        system_prompt: str = "你是一个智能家居助手，可以帮助用户处理各种任务。请用中文回答用户的问题，保持简洁友好的语气。"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.use_local = use_local
        self.system_prompt = system_prompt


class Message:
    """Represents a chat message."""
    def __init__(self, role: str, content: str):
        self.role = role  # "system", "user", or "assistant"
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class Conversation:
    """Manages conversation history."""
    def __init__(self, system_prompt: str = "", max_history: int = 20):
        self.messages: List[Message] = []
        self.max_history = max_history

        if system_prompt:
            self.add_message("system", system_prompt)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append(Message(role, content))

        # Keep conversation history manageable
        if len(self.messages) > self.max_history:
            # Keep system message if it exists
            system_msg = None
            if self.messages[0].role == "system":
                system_msg = self.messages[0]

            # Remove old messages but keep recent ones
            self.messages = self.messages[-self.max_history+1:]

            # Re-add system message if it was removed
            if system_msg and (not self.messages or self.messages[0].role != "system"):
                self.messages.insert(0, system_msg)

    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages in OpenAI API format."""
        return [msg.to_dict() for msg in self.messages]

    def clear_history(self):
        """Clear conversation history except system message."""
        system_msg = None
        if self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]

        self.messages.clear()
        if system_msg:
            self.messages.append(system_msg)


class LLMClient:
    """LLM client for AI dialogue using Alibaba Qwen models."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None  # For DashScope API
        self.local_model = None  # For local Qwen model
        self.local_tokenizer = None  # For local Qwen model
        self.session = None
        self._is_initialized = False
        self.conversation = Conversation(config.system_prompt)

    async def initialize(self) -> bool:
        """Initialize the Qwen LLM client."""
        try:
            if self.config.use_local:
                # Use local Qwen model
                return await self._initialize_local_model()
            else:
                # Use DashScope API
                return await self._initialize_dashscope_api()

        except Exception as e:
            logger.error(f"Failed to initialize Qwen LLM client: {e}")
            return False

    async def _initialize_dashscope_api(self) -> bool:
        """Initialize DashScope API client."""
        if not DASHScope_AVAILABLE:
            logger.error("dashscope not available. Please install with: pip install dashscope")
            return False

        # Debug: print API key status and try direct env read
        import os
        direct_env_key = os.environ.get('DASHSCOPE_API_KEY')
        logger.info(f"Config API key exists: {self.config.api_key is not None}")
        logger.info(f"Direct env DASHSCOPE_API_KEY exists: {direct_env_key is not None}")
        logger.info(f"Config API key value: {'*' * len(self.config.api_key) if self.config.api_key else 'None'}")

        # Use direct env var if config doesn't have it
        api_key = self.config.api_key or direct_env_key

        if not api_key:
            logger.error("DashScope API key not provided")
            return False

        try:
            # Set API key
            dashscope.api_key = api_key
            self._is_initialized = True
            logger.info(f"DashScope API client initialized with model: {self.config.model}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize DashScope API: {e}")
            return False

    async def _initialize_local_model(self) -> bool:
        """Initialize local Qwen model."""
        if not MODELScope_AVAILABLE:
            logger.error("modelscope not available. Please install with: pip install modelscope transformers torch")
            return False

        try:
            # Download model if needed
            model_path = snapshot_download(self.config.model)
            logger.info(f"Using local Qwen model: {model_path}")

            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer and model
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto"
            )

            self._is_initialized = True
            logger.info(f"Local Qwen model initialized on device: {device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize local Qwen model: {e}")
            return False

    async def cleanup(self):
        """Clean up LLM client resources."""
        if self.session:
            await self.session.close()
        if self.local_model:
            del self.local_model
        if self.local_tokenizer:
            del self.local_tokenizer

        self.client = None
        self.local_model = None
        self.local_tokenizer = None
        self.session = None
        self._is_initialized = False
        logger.info("Qwen LLM client cleaned up")

    def is_initialized(self) -> bool:
        """Check if LLM client is initialized."""
        return self._is_initialized

    async def generate_response(self, user_input: str) -> Optional[str]:
        """Generate AI response to user input.

        Args:
            user_input: User's text input

        Returns:
            AI response text or None if failed
        """
        if not self._is_initialized:
            logger.error("Qwen LLM client not initialized")
            return None

        try:
            # Add user message to conversation
            self.conversation.add_message("user", user_input)

            # Generate response
            if self.config.use_local and self.local_model:
                # Use local Qwen model
                response = await self._generate_local_response()
            else:
                # Use DashScope API
                response = await self._generate_dashscope_response()

            if response:
                # Add assistant response to conversation
                self.conversation.add_message("assistant", response)
                logger.info(f"Qwen generated response: {response[:100]}...")
                return response
            else:
                logger.error("Failed to generate Qwen response")
                return None

        except Exception as e:
            logger.error(f"Error generating Qwen response: {e}")
            return None

    async def _generate_dashscope_response(self) -> Optional[str]:
        """Generate response using DashScope API."""
        try:
            # Get API key (use config or direct env)
            import os
            api_key = self.config.api_key or os.environ.get('DASHSCOPE_API_KEY')

            # Prepare messages (DashScope format)
            messages = []
            for msg in self.conversation.messages:
                if msg.role == "system":
                    messages.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    messages.append({"role": "assistant", "content": msg.content})

            # Call DashScope API
            response = Generation.call(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=api_key
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                return content.strip()
            else:
                logger.error(f"DashScope API error: {response.status_code} - {response.message}")
                return None

        except Exception as e:
            logger.error(f"DashScope API request error: {e}")
            return None

    async def _generate_local_response(self) -> Optional[str]:
        """Generate response using local Qwen model."""
        try:
            # Prepare input text
            messages = self.conversation.get_messages()

            # Format for Qwen model
            input_text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    input_text += f"System: {content}\n"
                elif role == "user":
                    input_text += f"User: {content}\n"
                elif role == "assistant":
                    input_text += f"Assistant: {content}\n"

            input_text += "Assistant: "

            # Tokenize input
            inputs = self.local_tokenizer(input_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # Generate response
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )

            # Decode response
            response_text = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response (after the last "Assistant: ")
            assistant_responses = response_text.split("Assistant: ")
            if len(assistant_responses) > 1:
                response = assistant_responses[-1].strip()
                return response
            else:
                return response_text.strip()

        except Exception as e:
            logger.error(f"Local Qwen model inference error: {e}")
            return None

    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation.clear_history()
        logger.info("Conversation history cleared")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation.get_messages()

    async def test_connection(self) -> bool:
        """Test connection to Qwen LLM."""
        if not self._is_initialized:
            return False

        try:
            # Simple test query
            test_response = await self.generate_response("你好，请简单介绍一下你自己")
            if test_response:
                # Remove the test message from history
                if len(self.conversation.messages) >= 2:
                    self.conversation.messages = self.conversation.messages[:-2]
                logger.info("Qwen LLM connection test successful")
                return True
            else:
                logger.error("Qwen LLM connection test failed")
                return False
        except Exception as e:
            logger.error(f"Qwen LLM connection test error: {e}")
            return False
