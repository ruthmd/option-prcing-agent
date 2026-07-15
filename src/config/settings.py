# src/config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):

    # ── Provider Selection ────────────────────────────────────────────
    # Switch between: "local", "openai", "claude"
    LLM_PROVIDER: str = "local"

    # ── LLM-as-Judge (independent grounding check) ─────────────────────
    # Provider used to fact-check responses, kept independent from LLM_PROVIDER
    # so the same model isn't grading its own output. "none" disables the check.
    # Switch between: "local", "openai", "claude", "none"
    LLM_JUDGE: str = "openai"

    # ── Local (Ollama) ────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # ── OpenAI ────────────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"

    # ── Claude (Anthropic) ────────────────────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    # ── Temperature per task (applies to all providers) ───────────────
    MODEL_TEMP_PRICING: float = 0.01
    MODEL_TEMP_ANALYSIS: float = 0.05
    MODEL_TEMP_STRATEGY: float = 0.10
    MODEL_TEMP_VALIDATION: float = 0.00

    # ── Embedding Model ───────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Vector Store ──────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "data/knowledge_base/faiss_index"
    KNOWLEDGE_BASE_PATH: str = "data/knowledge_base"

    # ── Validation Thresholds ─────────────────────────────────────────
    DOMAIN_RELEVANCE_THRESHOLD: float = 0.7
    CONFIDENCE_THRESHOLD: float = 0.8
    MAX_STRIKE_DEVIATION: float = 0.5
    MAX_DAYS_TO_EXPIRY: int = 1095

    # ── Market Data ───────────────────────────────────────────────────
    YFINANCE_TIMEOUT: int = 30
    CACHE_DURATION_MINUTES: int = 5
    RISK_FREE_RATE_TICKER: str = "^TNX"
    DEFAULT_RISK_FREE_RATE: float = 0.05

    # ── Options Liquidity Filter ───────────────────────────────────────
    # An option quote is considered liquid enough to trust for implied
    # volatility calculations if it meets EITHER threshold (not both) —
    # a freshly-opened contract may have high volume but low open interest,
    # while an established position may have the reverse.
    MIN_OPTION_VOLUME: int = 1
    MIN_OPTION_OPEN_INTEREST: int = 10

    # Target horizon for volatility-analysis queries specifically — the
    # options chain nearest to this many days out is used, rather than
    # literally the next available expiration (which can be 1-2 DTE and
    # produce a numerically unstable smile). 30 days matches the standard
    # industry convention (e.g. VIX is a 30-day constant-maturity IV).
    VOLATILITY_TARGET_DTE: int = 30

    # ── Risk Management ───────────────────────────────────────────────
    MAX_VOLATILITY: float = 3.0
    MIN_VOLATILITY: float = 0.01
    MAX_INTEREST_RATE: float = 0.20
    MIN_STOCK_PRICE: float = 0.01

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()