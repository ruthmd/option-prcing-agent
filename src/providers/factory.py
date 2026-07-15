# src/providers/factory.py
from src.providers.base import BaseLLMProvider
from src.config import settings
import logging

logger = logging.getLogger(__name__)

def get_provider() -> BaseLLMProvider:
    """
    Factory function — returns the correct provider based on LLM_PROVIDER.

    Usage:
        llm = get_provider().get_llm("pricing")

    Switch provider by changing LLM_PROVIDER in .env:
        LLM_PROVIDER=local    → Ollama (Llama 3.2)
        LLM_PROVIDER=openai   → GPT-4o
        LLM_PROVIDER=claude   → Claude Sonnet
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "local":
        from src.providers.local import LocalProvider
        return LocalProvider()

    elif provider == "openai":
        from src.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()

    elif provider == "claude":
        from src.providers.claude_provider import ClaudeProvider
        return ClaudeProvider()

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            f"Choose from: 'local', 'openai', 'claude'"
        )


def get_llm(task_type: str = "analysis"):
    """
    Convenience shortcut so callers don't need to know about providers.

    Usage:
        from src.providers.factory import get_llm
        llm = get_llm("pricing")
    """
    return get_provider().get_llm(task_type)