# src/providers/local.py
from langchain_community.llms import Ollama
from src.providers.base import BaseLLMProvider
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class LocalProvider(BaseLLMProvider):
    """Ollama local inference — no API key needed."""

    # System prompts per role
    _SYSTEM_PROMPTS = {
        "pricing":    "You are a precise quantitative analyst specializing in options pricing.",
        "analysis":   "You are an expert financial analyst specializing in derivatives.",
        "strategy":   "You are a professional options trader specializing in complex strategies.",
        "validation": "You are a financial engineering expert focused on model validation.",
    }

    def get_llm(self, task_type: str = "analysis") -> Ollama:
        temperature = self._get_temperature(task_type)
        system = self._SYSTEM_PROMPTS.get(task_type, self._SYSTEM_PROMPTS["analysis"])

        logger.info(f"[LocalProvider] model={settings.OLLAMA_MODEL} task={task_type} temp={temperature}")

        return Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            system=system,
        )