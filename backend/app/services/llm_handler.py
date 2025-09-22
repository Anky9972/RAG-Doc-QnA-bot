# llm_handler.py
import google.generativeai as genai
from app.core import config
from typing import List, Dict, Optional, Tuple
import logging
import time
import json
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Global variables for model management
gemini_model = None
current_model_name_used = None
model_initialization_time = None
model_stats = {"queries_processed": 0, "total_tokens_used": 0, "errors": 0}

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass

def get_available_models() -> List[str]:
    """Get list of available Gemini models"""
    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                models.append(model.name)
        return models
    except Exception as e:
        logger.error(f"Error fetching available models: {e}")
        return []

def get_gemini_model():
    """Initialize Gemini model with improved error handling and fallback"""
    global gemini_model, current_model_name_used, model_initialization_time
    
    if gemini_model is not None:
        return gemini_model
    
    # Primary model candidates in order of preference
    model_candidates = [
        'models/gemini-1.5-pro-latest',
        'models/gemini-1.5-flash-latest', 
        'models/gemini-1.0-pro',
        'gemini-1.5-pro',
        'gemini-1.5-flash',
        'gemini-pro'
    ]
    
    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        
        # Get available models first
        available_models = get_available_models()
        if available_models:
            logger.info(f"Available models: {available_models}")
            # Prioritize available models
            model_candidates = [m for m in model_candidates if m in available_models] + \
                             [m for m in available_models if m not in model_candidates]
        
        for model_name in model_candidates:
            logger.info(f"Attempting to initialize Gemini model: {model_name}")
            try:
                # Test model initialization with a simple generation
                test_model = genai.GenerativeModel(model_name)
                
                # Configure generation settings for better performance
                generation_config = genai.types.GenerationConfig(
                    temperature=0.1,  # Lower temperature for more consistent responses
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                    stop_sequences=[]
                )
                
                # Test with a simple prompt
                test_response = test_model.generate_content(
                    "Say 'Model initialized successfully'",
                    generation_config=generation_config
                )
                
                if test_response.text:
                    gemini_model = test_model  # Store the configured model
                    current_model_name_used = model_name
                    model_initialization_time = time.time()
                    logger.info(f"Successfully initialized Gemini model: {current_model_name_used}")
                    return gemini_model
                    
            except Exception as e:
                logger.warning(f"Failed to initialize model '{model_name}': {e}")
                continue
        
        # If no model worked
        logger.error("Could not initialize any Gemini model")
        raise LLMError("Failed to initialize any available Gemini model")
        
    except Exception as e:
        logger.error(f"Error during Gemini model initialization: {e}", exc_info=True)
        raise LLMError(f"Gemini model initialization failed: {str(e)}")

@lru_cache(maxsize=100)
def construct_optimized_prompt(query: str, context_hash: str, context_str: str) -> str:
    """Construct optimized prompt with caching"""
    
    if not context_str or context_str.strip() == "No relevant context was found in the document.":
        prompt = f"""You are a helpful AI assistant. The user has asked a question about a document, but no relevant context could be found.

Question: {query}

Please respond that you could not find information related to this question in the provided document. Suggest that the user:
1. Try rephrasing their question
2. Check if the information might be in a different document
3. Verify that the document was uploaded correctly

Answer:"""
    else:
        prompt = f"""You are an AI assistant specialized in answering questions based on document content. 

Instructions:
- Answer the question using ONLY the information provided in the context
- If the answer is not in the context, clearly state that you cannot find the information
- Be precise and cite specific details when possible
- If the context is unclear or incomplete, mention this limitation

Context from document:
{context_str}

Question: {query}

Answer:"""
    
    return prompt

def construct_prompt(query: str, context_chunks: List[Dict[str, any]]) -> str:
    """Construct prompt for LLM with improved context handling"""
    
    if not context_chunks:
        context_str = "No relevant context was found in the document."
    else:
        # Sort chunks by relevance if distance is available
        sorted_chunks = sorted(context_chunks, 
                             key=lambda x: x.get('_additional', {}).get('distance', 1.0))
        
        # Create context with page references
        context_parts = []
        for i, chunk in enumerate(sorted_chunks[:5]):  # Limit to top 5 chunks
            text = chunk.get('text', '').strip()
            page_num = chunk.get('page_number', 'Unknown')
            
            if text:
                context_parts.append(f"[Page {page_num}] {text}")
        
        context_str = "\n\n".join(context_parts)
    
    # Create a hash of context for caching
    context_hash = str(hash(context_str))
    
    return construct_optimized_prompt(query, context_hash, context_str)

def validate_response(response_text: str, query: str) -> Tuple[bool, str]:
    """Validate LLM response quality"""
    
    if not response_text or len(response_text.strip()) < 10:
        return False, "Response too short"
    
    # Check for common error patterns
    error_patterns = [
        "I'm sorry, I can't",
        "I cannot provide",
        "I don't have access",
        "I'm not able to"
    ]
    
    response_lower = response_text.lower()
    for pattern in error_patterns:
        if pattern.lower() in response_lower:
            logger.warning(f"Response contains potential error pattern: {pattern}")
    
    # Check if response seems related to query
    query_words = set(query.lower().split())
    response_words = set(response_text.lower().split())
    
    if len(query_words) > 2:  # Only check for longer queries
        common_words = query_words.intersection(response_words)
        if len(common_words) == 0:
            logger.warning("Response may not be related to query")
    
    return True, "Valid"

async def query_llm_async(query: str, context_chunks: List[Dict[str, any]]) -> str:
    """Async version of query_llm for better performance"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, query_llm, query, context_chunks)

def query_llm(query: str, context_chunks: List[Dict[str, any]], 
              max_retries: int = 3, timeout: int = 30) -> str:
    """
    Query LLM with improved error handling, retries, and response validation
    """
    global model_stats
    
    start_time = time.time()
    
    try:
        model = get_gemini_model()
        if model is None:
            model_stats["errors"] += 1
            return "Sorry, the AI language model is currently unavailable."
            
    except Exception as e:
        logger.error(f"LLM model acquisition failed: {e}")
        model_stats["errors"] += 1
        return "Sorry, the AI language model could not be initialized."

    prompt = construct_prompt(query, context_chunks)
    
    # Configure generation parameters
    generation_config = genai.types.GenerationConfig(
        temperature=0.1,
        top_p=0.8,
        top_k=40,
        max_output_tokens=2048,
    )
    
    logger.info(f"Querying LLM ({current_model_name_used}) with {len(context_chunks)} context chunks")
    
    for attempt in range(max_retries):
        try:
            # Generate response with timeout
            api_response = model.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={'timeout': timeout}
            )
            
            # Check for safety blocks
            if hasattr(api_response, 'prompt_feedback') and api_response.prompt_feedback:
                block_reason = getattr(api_response.prompt_feedback, 'block_reason', None)
                if block_reason:
                    logger.warning(f"Content blocked by Gemini: {block_reason}")
                    model_stats["errors"] += 1
                    return f"My response was blocked by safety settings. Please try rephrasing your query."
            
            # Extract response text
            response_text = ""
            if hasattr(api_response, 'text') and api_response.text:
                response_text = api_response.text
            elif hasattr(api_response, 'parts') and api_response.parts:
                text_parts = [part.text for part in api_response.parts if hasattr(part, 'text')]
                response_text = "".join(text_parts)
            
            if not response_text:
                logger.warning("No text content in LLM response")
                if attempt < max_retries - 1:
                    continue
                else:
                    model_stats["errors"] += 1
                    return "I could not generate a proper response. Please try again."
            
            # Validate response
            is_valid, validation_msg = validate_response(response_text, query)
            if not is_valid and attempt < max_retries - 1:
                logger.warning(f"Response validation failed: {validation_msg}, retrying...")
                continue
            
            # Update statistics
            model_stats["queries_processed"] += 1
            processing_time = time.time() - start_time
            logger.info(f"LLM query completed in {processing_time:.2f}s")
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Error during LLM query (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            else:
                model_stats["errors"] += 1
                return "I encountered an error while processing your question. Please try again."
    
    model_stats["errors"] += 1
    return "Failed to get a response after multiple attempts. Please try again later."

def get_model_stats() -> Dict[str, any]:
    """Get model performance statistics"""
    global model_stats, model_initialization_time, current_model_name_used
    
    stats = model_stats.copy()
    stats.update({
        "current_model": current_model_name_used,
        "initialization_time": model_initialization_time,
        "uptime_hours": (time.time() - model_initialization_time) / 3600 if model_initialization_time else 0
    })
    
    return stats

def reset_model():
    """Reset model (useful for testing or recovery)"""
    global gemini_model, current_model_name_used, model_initialization_time
    gemini_model = None
    current_model_name_used = None  
    model_initialization_time = None
    logger.info("Model reset completed")