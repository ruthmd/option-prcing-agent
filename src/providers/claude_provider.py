# src/providers/claude_provider.py
from langchain_anthropic import ChatAnthropic
from src.providers.base import BaseLLMProvider
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def get_llm(self, task_type: str = "analysis") -> ChatAnthropic:
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or set LLM_PROVIDER=local"
            )

        temperature = self._get_temperature(task_type)

        logger.info(f"[ClaudeProvider] model={settings.ANTHROPIC_MODEL} task={task_type} temp={temperature}")

        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_retries=3,
        )