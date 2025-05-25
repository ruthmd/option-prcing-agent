from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import os

class Settings(BaseSettings):
    # Model Configuration
    HF_MODEL_NAME: str = "microsoft/DialoGPT-medium"  # Lightweight for local testing
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Store
    FAISS_INDEX_PATH: str = "data/knowledge_base/faiss_index"
    KNOWLEDGE_BASE_PATH: str = "data/knowledge_base"
    
    # Validation Thresholds
    DOMAIN_RELEVANCE_THRESHOLD: float = 0.7
    CONFIDENCE_THRESHOLD: float = 0.8
    MAX_STRIKE_DEVIATION: float = 0.5  # 50% from current price
    MAX_DAYS_TO_EXPIRY: int = 1095  # 3 years
    
    # Market Data
    YFINANCE_TIMEOUT: int = 30
    CACHE_DURATION_MINUTES: int = 5
    
    # Risk Management
    MAX_VOLATILITY: float = 3.0  # 300%
    MIN_VOLATILITY: float = 0.01  # 1%
    MAX_INTEREST_RATE: float = 0.2  # 20%
    
    class Config:
        env_file = ".env"

settings = Settings()