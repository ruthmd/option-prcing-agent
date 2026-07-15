# src/providers/factory.py
from typing import Optional
from src.providers.base import BaseLLMProvider
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

def get_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
    """
    Factory function — returns a provider by name, defaulting to LLM_PROVIDER.

    Usage:
        llm = get_provider().get_llm("pricing")            # uses settings.LLM_PROVIDER
        llm = get_provider("openai").get_llm("validation")  # explicit override

    Switch the default provider by changing LLM_PROVIDER in .env:
        LLM_PROVIDER=local    → Ollama (Llama 3.2)
        LLM_PROVIDER=openai   → GPT-4o
        LLM_PROVIDER=claude   → Claude Sonnet
    """
    provider = (provider_name or settings.LLM_PROVIDER).lower()

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
            f"Unknown provider='{provider}'. "
            f"Choose from: 'local', 'openai', 'claude'"
        )


def get_llm(task_type: str = "analysis", provider_name: Optional[str] = None):
    """
    Convenience shortcut so callers don't need to know about providers.

    Usage:
        from src.providers.factory import get_llm
        llm = get_llm("pricing")                          # uses settings.LLM_PROVIDER
        llm = get_llm("validation", provider_name="openai") # explicit override
    """
    return get_provider(provider_name).get_llm(task_type)