# src/providers/openai_provider.py
from langchain_openai import ChatOpenAI
from src.providers.base import BaseLLMProvider
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (GPT-4o, GPT-4-turbo, etc.)."""

    def get_llm(self, task_type: str = "analysis") -> ChatOpenAI:
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or set LLM_PROVIDER=local"
            )

        temperature = self._get_temperature(task_type)

        logger.info(f"[OpenAIProvider] model={settings.OPENAI_MODEL} task={task_type} temp={temperature}")

        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
            max_retries=3,
        )