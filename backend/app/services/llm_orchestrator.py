# services/llm_orchestrator.py
import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
import openai
import anthropic
import ollama
import requests
from app.core import config

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"  
    COMPLEX = "complex"

@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    tokens_used: int
    latency_ms: int
    cost_cents: float
    metadata: Dict[str, Any] = None

class BaseLLMProvider(ABC):
    def __init__(self, provider_name: str, config_dict: Dict[str, Any]):
        self.provider_name = provider_name
        self.config = config_dict
        self.models = config_dict.get("models", [])
        self.enabled = config_dict.get("enabled", False)
        # Default models for each provider
        self.default_models = self.get_default_models()

    @abstractmethod
    async def generate_response(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @abstractmethod
    def get_default_models(self) -> List[str]:
        """Return list of default models for this provider"""
        pass
    
    def validate_model(self, model: str) -> str:
        """Validate and return appropriate model for this provider"""
        if not model:
            return self.default_models[0] if self.default_models else None
        
        # If the model is in our configured models, use it
        if model in self.models:
            return model
        
        # If it's a valid model for this provider, use it
        if model in self.get_all_supported_models():
            return model
        
        # Otherwise, return default
        logger.warning(f"Model '{model}' not supported by {self.provider_name}, using default")
        return self.default_models[0] if self.default_models else None
    
    @abstractmethod
    def get_all_supported_models(self) -> List[str]:
        """Return all supported models for this provider"""
        pass

class OllamaProvider(BaseLLMProvider):
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__("ollama", config_dict)
        self.base_url = config_dict.get("base_url", "http://localhost:11434")
        self.available_models = []
        # Initialize Ollama client
        self.client = ollama.Client(host=self.base_url)

    def get_default_models(self) -> List[str]:
        return ["llama3.2:3b", "llama2", "mistral"]
    
    def get_all_supported_models(self) -> List[str]:
        """Return all models currently available in Ollama"""
        return self.available_models

    def is_available(self) -> bool:
        """Check if Ollama is running and has models available"""
        try:
            # Use the client to list models - Ollama 0.5.3+ returns a structured response
            response = self.client.list()
            
            # In Ollama 0.5.3, response is likely a Pydantic model with a 'models' attribute
            models_list = []
            
            if hasattr(response, 'models'):
                # This is the most likely format for v0.5.3
                models_list = response.models
            elif hasattr(response, 'model_list'):  # Alternative attribute name
                models_list = response.model_list
            elif isinstance(response, dict):
                models_list = response.get('models', [])
            elif isinstance(response, list):
                models_list = response
            else:
                # Try to convert to dict if it's a Pydantic model
                try:
                    response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
                    models_list = response_dict.get('models', [])
                except:
                    logger.error(f"Unknown response format: {type(response)}")
                    return False
            
            # Extract model names - handle different model object formats
            self.available_models = []
            for model in models_list:
                model_name = None
                
                if isinstance(model, str):
                    model_name = model
                elif isinstance(model, dict):
                    model_name = model.get('name') or model.get('model')
                elif hasattr(model, 'name'):
                    model_name = model.name
                elif hasattr(model, 'model'):
                    model_name = model.model
                else:
                    # Try to convert to dict if it's a Pydantic model
                    try:
                        model_dict = model.model_dump() if hasattr(model, 'model_dump') else model.dict()
                        model_name = model_dict.get('name') or model_dict.get('model')
                    except:
                        # Last resort - convert to string
                        model_name = str(model)
                
                if model_name:
                    self.available_models.append(model_name)
            
            logger.info(f"Ollama available with models: {self.available_models}")
            return len(self.available_models) > 0
            
        except Exception as e:
            logger.warning(f"Cannot connect to Ollama at {self.base_url}: {e}")
            logger.debug(f"Full error: {type(e).__name__}: {str(e)}")
            return False

    def validate_model(self, model: str) -> str:
        """Override to check against available models"""
        if not model:
            return self.available_models[0] if self.available_models else "llama3.2:3b"
        
        # Try exact match first
        if model in self.available_models:
            return model
        
        # Try partial match (useful for models with tags like :latest)
        for available_model in self.available_models:
            if model in available_model or available_model.startswith(model.split(':')[0]):
                logger.info(f"Using model '{available_model}' for requested '{model}'")
                return available_model
        
        # If requested model not available, use first available
        if self.available_models:
            logger.warning(f"Model '{model}' not available in Ollama, using '{self.available_models[0]}'")
            return self.available_models[0]
        
        return "llama3.2:3b"  # Final fallback

    async def generate_response(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        model = self.validate_model(model)
        
        start_time = time.time()
        try:
            # Use the client instance with proper async handling
            response = await asyncio.to_thread(
                self.client.generate,
                model=model,
                prompt=prompt,
                options={
                    'temperature': kwargs.get('temperature', 0.1),
                    'top_p': kwargs.get('top_p', 0.9),
                    'num_predict': kwargs.get('max_tokens', 2048)
                }
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Handle response - it might be a Pydantic model in v0.5.3
            response_text = ""
            tokens_used = 0
            metadata = {}
            
            if isinstance(response, dict):
                response_text = response.get('response', '')
                tokens_used = response.get('eval_count', 0)
                metadata = response
            elif hasattr(response, 'response'):
                response_text = response.response
                tokens_used = getattr(response, 'eval_count', 0)
                # Convert to dict for metadata
                try:
                    metadata = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
                except:
                    metadata = {'raw_response': str(response)}
            else:
                response_text = str(response)
                metadata = {'raw_response': str(response)}
            
            return LLMResponse(
                content=response_text,
                provider=self.provider_name,
                model=model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_cents=0.0,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

class GeminiProvider(BaseLLMProvider):
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__("gemini", config_dict)
        if self.enabled and hasattr(config, 'GOOGLE_API_KEY') and config.GOOGLE_API_KEY:
            genai.configure(api_key=config.GOOGLE_API_KEY)

    def get_default_models(self) -> List[str]:
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    def get_all_supported_models(self) -> List[str]:
        return [
            "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro",
            "gemini-1.5-flash-001", "gemini-1.5-pro-001"
        ]

    def is_available(self) -> bool:
        return hasattr(config, 'GOOGLE_API_KEY') and bool(config.GOOGLE_API_KEY)

    async def generate_response(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        model = self.validate_model(model)
        
        start_time = time.time()
        try:
            gemini_model = genai.GenerativeModel(model)
            
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0.1),
                top_p=kwargs.get('top_p', 0.8),
                max_output_tokens=kwargs.get('max_tokens', 2048)
            )
            
            response = await asyncio.to_thread(
                gemini_model.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Get actual usage if available - FIXED: Better handling of usage_metadata
            tokens_used = 0
            usage_dict = {}
            
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                try:
                    # Try to get token counts from usage_metadata
                    usage_metadata = response.usage_metadata
                    prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                    output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
                    tokens_used = prompt_tokens + output_tokens
                    
                    # Safely extract usage data without using _asdict
                    usage_dict = {
                        'prompt_token_count': prompt_tokens,
                        'candidates_token_count': output_tokens,
                        'total_token_count': getattr(usage_metadata, 'total_token_count', tokens_used)
                    }
                except Exception as usage_error:
                    logger.warning(f"Could not extract usage metadata: {usage_error}")
                    # Fallback to text-based estimation
                    tokens_used = len(response.text.split()) if response.text else 0
            else:
                # Fallback estimation
                tokens_used = len(response.text.split()) if response.text else 0
            
            cost_cents = tokens_used * 0.001
            
            return LLMResponse(
                content=response.text or '',
                provider=self.provider_name,
                model=model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_cents=cost_cents,
                metadata={'usage': usage_dict}
            )
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__("openai", config_dict)
        if self.enabled and hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY:
            self.client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    def get_default_models(self) -> List[str]:
        return ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    
    def get_all_supported_models(self) -> List[str]:
        return [
            "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo-16k", "gpt-4-32k"
        ]

    def is_available(self) -> bool:
        return hasattr(config, 'OPENAI_API_KEY') and bool(config.OPENAI_API_KEY)

    async def generate_response(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        model = self.validate_model(model)
        
        start_time = time.time()
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_tokens', 2048)
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            usage = response.usage
            
            # Calculate cost based on model
            cost_per_1k_tokens = {
                'gpt-4o': 0.005,
                'gpt-4o-mini': 0.0003,
                'gpt-4-turbo': 0.01,
                'gpt-4': 0.03,
                'gpt-3.5-turbo': 0.002
            }
            cost_rate = cost_per_1k_tokens.get(model, 0.002)
            cost_cents = (usage.total_tokens / 1000) * cost_rate * 100
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=self.provider_name,
                model=model,
                tokens_used=usage.total_tokens,
                latency_ms=latency_ms,
                cost_cents=cost_cents,
                metadata={'usage': {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens
                }}
            )
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__("anthropic", config_dict)
        if self.enabled and hasattr(config, 'ANTHROPIC_API_KEY') and config.ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def get_default_models(self) -> List[str]:
        return ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620"]
    
    def get_all_supported_models(self) -> List[str]:
        return [
            "claude-3-haiku-20240307", "claude-3-sonnet-20240229", 
            "claude-3-opus-20240229", "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022", "claude-2.1", "claude-2.0"
        ]

    def is_available(self) -> bool:
        return hasattr(config, 'ANTHROPIC_API_KEY') and bool(config.ANTHROPIC_API_KEY)

    async def generate_response(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        model = self.validate_model(model)
        
        start_time = time.time()
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model,
                max_tokens=kwargs.get('max_tokens', 2048),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.1)
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Updated cost estimates for Claude 3.5
            cost_per_1k_tokens = {
                'claude-3-5-sonnet-20241022': 0.003,
                'claude-3-5-sonnet-20240620': 0.003,
                'claude-3-sonnet-20240229': 0.003,
                'claude-3-haiku-20240307': 0.00025,
                'claude-3-opus-20240229': 0.015,
            }
            cost_rate = cost_per_1k_tokens.get(model, 0.003)
            cost_cents = (tokens_used / 1000) * cost_rate * 100
            
            return LLMResponse(
                content=response.content[0].text,
                provider=self.provider_name,
                model=model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost_cents=cost_cents,
                metadata={'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }}
            )
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

class LLMOrchestrator:
    def __init__(self):
        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available providers"""
        provider_classes = {
            'ollama': OllamaProvider,
            'gemini': GeminiProvider,
            'openai': OpenAIProvider,
            'anthropic': AnthropicProvider
        }
        
        for provider_name, provider_class in provider_classes.items():
            if hasattr(config, 'LLM_PROVIDERS') and provider_name in config.LLM_PROVIDERS:
                provider_config = config.LLM_PROVIDERS[provider_name]
                try:
                    logger.info(f"Initializing {provider_name} provider...")
                    provider = provider_class(provider_config)
                    
                    if provider.enabled and provider.is_available():
                        self.providers[provider_name] = provider
                        logger.info(f"✅ {provider_name} provider initialized and available")
                    elif provider.enabled:
                        logger.warning(f"⚠️  {provider_name} provider enabled but not available")
                    else:
                        logger.info(f"ℹ️  {provider_name} provider disabled in config")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to initialize {provider_name} provider: {e}")

    def analyze_query_complexity(self, query: str, context_length: int = 0) -> QueryComplexity:
        """Analyze query to determine complexity"""
        query_words = len(query.split())
        
        complex_keywords = [
            'analyze', 'compare', 'summarize', 'explain', 'reasoning', 
            'why', 'how', 'what if', 'implications', 'consequences',
            'structure', 'detailed', 'comprehensive', 'citations'
        ]
        
        has_complex_keywords = any(keyword in query.lower() for keyword in complex_keywords)
        
        if query_words > 20 or has_complex_keywords or context_length > 5000:
            return QueryComplexity.COMPLEX
        elif query_words > 10 or context_length > 2000:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE

    def select_provider_and_model(
        self, 
        complexity: QueryComplexity, 
        prefer_fast: bool = False, 
        preferred_provider: str = None,
        preferred_model: str = None
    ) -> Tuple[str, str]:
        """Select best provider and appropriate model with updated priority: Ollama -> Gemini -> OpenAI -> Anthropic"""
        if not self.providers:
            raise Exception("No LLM providers available")
        
        provider_name = None
        model = None
        
        # If specific provider requested and available
        if preferred_provider and preferred_provider in self.providers:
            provider_name = preferred_provider
        else:
            # Updated priority order: Ollama first for all cases (cost-effective, privacy-friendly)
            if complexity == QueryComplexity.SIMPLE or prefer_fast:
                # For simple queries, prioritize speed and local processing
                priority = ['ollama', 'gemini', 'openai', 'anthropic']
            elif complexity == QueryComplexity.MEDIUM:
                # For medium queries, still prefer local but allow cloud fallback
                priority = ['ollama', 'gemini', 'openai', 'anthropic']
            else:  # Complex queries
                # For complex queries, start with Ollama but have stronger cloud models as backup
                priority = ['ollama', 'gemini', 'openai', 'anthropic']
            
            # Find first available provider
            for p in priority:
                if p in self.providers:
                    provider_name = p
                    break
            
            if not provider_name:
                provider_name = list(self.providers.keys())[0]
        
        # Get appropriate model for the selected provider
        provider = self.providers[provider_name]
        
        # If a specific model is requested, validate it for this provider
        if preferred_model:
            model = provider.validate_model(preferred_model)
        else:
            # Select default model based on complexity and provider
            if provider_name == 'ollama':
                # For Ollama, prefer larger models for complex queries if available
                if complexity == QueryComplexity.COMPLEX and len(provider.available_models) > 1:
                    # Try to find a larger model (look for models with higher parameter counts)
                    larger_models = [m for m in provider.available_models if any(size in m.lower() for size in ['13b', '70b', '7b'])]
                    if larger_models:
                        # Sort by model size (rough heuristic)
                        larger_models.sort(key=lambda x: (
                            70 if '70b' in x.lower() else
                            13 if '13b' in x.lower() else
                            7 if '7b' in x.lower() else
                            3 if '3b' in x.lower() else 1
                        ), reverse=True)
                        model = larger_models[0]
                    else:
                        model = provider.default_models[-1] if provider.default_models else None
                else:
                    # For simple/medium or when only one model available
                    model = provider.default_models[0] if provider.default_models else None
            else:
                # For cloud providers, use existing logic
                if complexity == QueryComplexity.SIMPLE or prefer_fast:
                    # Use fastest/smallest model
                    model = provider.default_models[0] if provider.default_models else None
                elif complexity == QueryComplexity.COMPLEX:
                    # Use most capable model (last in list is typically most capable)
                    model = provider.default_models[-1] if provider.default_models else None
                else:
                    # Use middle-ground model
                    models = provider.default_models
                    model = models[len(models)//2] if models else None
        
        logger.info(f"Selected {provider_name} with model {model} for {complexity.value} query")
        return provider_name, model

    async def generate_response(
        self, 
        prompt: str, 
        context_chunks: List[Dict[str, Any]] = None,
        prefer_fast: bool = False,
        provider: str = None,
        model: str = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using the best available provider and model"""
        
        if context_chunks is None:
            context_chunks = []
        
        # Analyze query complexity
        context_length = sum(len(chunk.get('text', '')) for chunk in context_chunks)
        complexity = self.analyze_query_complexity(prompt, context_length)
        
        # Select provider and model
        provider_name, model_name = self.select_provider_and_model(
            complexity, prefer_fast, provider, model
        )
        
        if provider_name not in self.providers:
            raise Exception(f"Provider {provider_name} not available. Available: {list(self.providers.keys())}")
        
        # Generate response
        try:
            logger.info(f"Generating response using {provider_name} with model {model_name}")
            return await self.providers[provider_name].generate_response(
                prompt, model_name, **kwargs
            )
        except Exception as e:
            logger.error(f"Primary provider {provider_name} failed: {e}")
            
            # Fallback with updated priority order
            fallback_priority = ['ollama', 'gemini', 'openai', 'anthropic']
            available_providers = [p for p in fallback_priority if p in self.providers.keys() and p != provider_name]
            
            if available_providers:
                fallback_provider = available_providers[0]
                logger.info(f"Falling back to {fallback_provider}")
                fallback_model = self.providers[fallback_provider].default_models[0]
                return await self.providers[fallback_provider].generate_response(
                    prompt, fallback_model, **kwargs
                )
            else:
                raise Exception("All providers failed")

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers"""
        stats = {}
        
        if hasattr(config, 'LLM_PROVIDERS'):
            for provider_name, config_dict in config.LLM_PROVIDERS.items():
                if provider_name in self.providers:
                    provider = self.providers[provider_name]
                    stats[provider_name] = {
                        'enabled': provider.enabled,
                        'models': provider.models,
                        'default_models': provider.default_models,
                        'available': provider.is_available(),
                        'status': 'active'
                    }
                    
                    # Add available models for Ollama
                    if provider_name == 'ollama':
                        stats[provider_name]['available_models'] = getattr(provider, 'available_models', [])
                else:
                    stats[provider_name] = {
                        'enabled': config_dict.get('enabled', False),
                        'models': config_dict.get('models', []),
                        'available': False,
                        'status': 'inactive'
                    }
        
        return stats

    def refresh_providers(self):
        """Refresh provider availability"""
        logger.info("Refreshing provider connections...")
        self.providers.clear()
        self._initialize_providers()

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())

    def get_provider_models(self, provider_name: str) -> List[str]:
        """Get available models for a specific provider"""
        if provider_name in self.providers:
            return self.providers[provider_name].get_all_supported_models()
        return []

# Initialize global orchestrator
orchestrator = LLMOrchestrator()