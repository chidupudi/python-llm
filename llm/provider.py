import ollama
from typing import Iterator, Optional
import logging

logger = logging.getLogger(__name__)


class LocalLLMProvider:
    """
    Free local LLM provider using Ollama.
    Supports: llama3.1, codellama, mistral, deepseek-coder
    """

    def __init__(self, model: str = "llama3.1:latest"):
        self.model = model
        self.client = ollama.Client()
        self._verify_model()

    def _verify_model(self):
        """Check if model is available, suggest download if not."""
        try:
            models = self.client.list()
            available = [m['name'] for m in models.get('models', [])]

            if self.model not in available:
                logger.warning(f"Model {self.model} not found.")
                logger.info(f"Available models: {available}")
                logger.info(f"Download with: ollama pull {self.model}")

                # Try to pull automatically
                logger.info(f"Attempting to pull {self.model}...")
                self.client.pull(self.model)
                logger.info(f"✓ Model {self.model} downloaded successfully")
        except Exception as e:
            logger.error(f"Error verifying model: {e}")
            logger.info("Please install Ollama from: https://ollama.ai")
            raise

    def generate(self, prompt: str, stream: bool = True) -> Iterator[str]:
        """
        Generate response from local LLM.

        Args:
            prompt: The complete prompt with context
            stream: Whether to stream response chunks

        Yields:
            Response chunks if streaming, full response if not
        """
        try:
            if stream:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=True
                )

                for chunk in response:
                    if 'response' in chunk:
                        yield chunk['response']
            else:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=False
                )
                yield response['response']

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            yield f"Error: {str(e)}\n\nPlease ensure Ollama is running: `ollama serve`"

    def chat(self, messages: list, stream: bool = True) -> Iterator[str]:
        """
        Chat-style interface for conversational context.

        Args:
            messages: List of {role: "user"/"assistant", content: "..."}
            stream: Whether to stream response

        Yields:
            Response chunks
        """
        try:
            if stream:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    stream=True
                )

                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        yield chunk['message']['content']
            else:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    stream=False
                )
                yield response['message']['content']

        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            yield f"Error: {str(e)}"


class ModelRegistry:
    """Registry of free, recommended models for debugging tasks."""

    MODELS = {
        "codellama": {
            "name": "codellama:13b",
            "description": "Best for code understanding and debugging",
            "size": "7.4GB"
        },
        "llama3.1": {
            "name": "llama3.1:latest",
            "description": "General purpose, good reasoning",
            "size": "4.7GB"
        },
        "mistral": {
            "name": "mistral:latest",
            "description": "Fast, lightweight, good for quick suggestions",
            "size": "4.1GB"
        },
        "deepseek-coder": {
            "name": "deepseek-coder:6.7b",
            "description": "Specialized for code, excellent for debugging",
            "size": "3.8GB"
        }
    }

    @classmethod
    def get_recommended(cls) -> str:
        """Get recommended model for debugging."""
        return cls.MODELS["deepseek-coder"]["name"]

    @classmethod
    def list_models(cls) -> dict:
        """List all available free models."""
        return cls.MODELS
