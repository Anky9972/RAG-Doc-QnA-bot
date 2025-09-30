# core/config.py
import os
from dotenv import load_dotenv
from typing import Dict, List
import json

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Database Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "pdf_qa_system")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Weaviate Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "PdfChunk")
WEAVIATE_CHAT_MESSAGE_CLASS_NAME = "ChatMessage"
WEAVIATE_CONVERSATION_CLASS_NAME = "Conversation"

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_DIM = 384  # For sentence-transformers/all-MiniLM-L6-v2

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", "llama2,mistral,phi3").split(",")

# LlamaCloud configuration
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')
LLAMA_CLOUD_TIMEOUT = int(os.getenv('LLAMA_CLOUD_TIMEOUT', '120'))  # 2 minutes
LLAMA_CLOUD_MAX_WORKERS = int(os.getenv('LLAMA_CLOUD_MAX_WORKERS', '4'))

# Processing preferences
USE_LLAMACLOUD_BY_DEFAULT = os.getenv('USE_LLAMACLOUD_BY_DEFAULT', 'true').lower() == 'true'
LLAMACLOUD_MIN_FILE_SIZE = int(os.getenv('LLAMACLOUD_MIN_FILE_SIZE', '1048576'))  # 1MB
LLAMACLOUD_MAX_FILE_SIZE = int(os.getenv('LLAMACLOUD_MAX_FILE_SIZE', '52428800'))  # 50MB
# LLM Provider Configuration
LLM_PROVIDERS = {
    "ollama": {
        "enabled": True,
        "models": OLLAMA_MODELS,
        "base_url": OLLAMA_BASE_URL,
        "use_for": ["fast", "light"]
    },
    "gemini": {
        "enabled": bool(GOOGLE_API_KEY),
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
        "use_for": ["complex", "reasoning"]
    },
    "openai": {
        "enabled": bool(OPENAI_API_KEY),
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "use_for": ["complex", "reasoning"]
    },
    "anthropic": {
        "enabled": bool(ANTHROPIC_API_KEY),
        "models": ["claude-3-sonnet", "claude-3-haiku"],
        "use_for": ["complex", "reasoning"]
    }
}

# Search Configuration
HYBRID_SEARCH_CONFIG = {
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    "rerank_top_k": 20,
    "final_top_k": 5
}

# Memory Configuration
MEMORY_CONFIG = {
    "short_term_turns": 10,
    "long_term_summary_threshold": 20,
    "conversation_timeout_hours": 24
}

# Chunking Configuration
CHUNKING_CONFIG = {
    "adaptive_chunking": True,
    "base_chunk_size": 1000,
    "max_chunk_size": 2000,
    "min_chunk_size": 200,
    "overlap_percentage": 0.1,
    "semantic_threshold": 0.8
}

# Rate Limiting
RATE_LIMITS = {
    "queries_per_minute": 60,
    "uploads_per_hour": 100
}

def get_active_llm_providers() -> List[str]:
    """Get list of active LLM providers"""
    return [provider for provider, config in LLM_PROVIDERS.items() if config["enabled"]]

def get_provider_for_query_type(query_type: str) -> str:
    """Get best provider for specific query type"""
    for provider, config in LLM_PROVIDERS.items():
        if config["enabled"] and query_type in config.get("use_for", []):
            return provider
    
    # Fallback to first available provider
    active_providers = get_active_llm_providers()
    return active_providers[0] if active_providers else "gemini"