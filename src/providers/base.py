# src/providers/base.py
from abc import ABC, abstractmethod
from langchain_core.language_models import BaseLLM

class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Every provider must implement get_llm() so the
    rest of the codebase stays provider-agnostic.
    """

    @abstractmethod
    def get_llm(self, task_type: str = "analysis") -> BaseLLM:
        """
        Return a configured LLM for a given task type.

        Args:
            task_type: One of 'pricing', 'analysis', 'strategy', 'validation'

        Returns:
            A LangChain-compatible LLM instance
        """
        raise NotImplementedError

    def _get_temperature(self, task_type: str) -> float:
        """Map task type to the correct temperature from settings."""
        from src.config import settings
        return {
            "pricing":    settings.MODEL_TEMP_PRICING,
            "analysis":   settings.MODEL_TEMP_ANALYSIS,
            "strategy":   settings.MODEL_TEMP_STRATEGY,
            "validation": settings.MODEL_TEMP_VALIDATION,
        }.get(task_type, settings.MODEL_TEMP_ANALYSIS)