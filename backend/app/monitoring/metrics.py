# monitoring/metrics.py
import time
import asyncio
import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps
from app.core import config

logger = logging.getLogger(__name__)

# Metrics definitions
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
QUERY_COUNT = Counter('queries_total', 'Total queries processed', ['provider', 'model'])
QUERY_DURATION = Histogram('query_duration_seconds', 'Query processing duration', ['provider'])
EMBEDDING_COUNT = Counter('embeddings_total', 'Total embeddings generated')
DOCUMENT_COUNT = Gauge('documents_total', 'Total documents stored')
VECTOR_SEARCH_DURATION = Histogram('vector_search_duration_seconds', 'Vector search duration')
LLM_TOKEN_COUNT = Counter('llm_tokens_total', 'Total LLM tokens used', ['provider', 'model'])
LLM_COST = Counter('llm_cost_cents_total', 'Total LLM cost in cents', ['provider'])
ERROR_COUNT = Counter('errors_total', 'Total errors', ['component', 'error_type'])

# Additional metrics for better monitoring
LLM_LATENCY = Histogram('llm_latency_seconds', 'LLM response latency', ['provider', 'model'])
DOCUMENT_PROCESSING_DURATION = Histogram('document_processing_duration_seconds', 'Document processing duration')
CHUNK_COUNT = Counter('chunks_processed_total', 'Total chunks processed')
PROVIDER_AVAILABILITY = Gauge('provider_availability', 'Provider availability status', ['provider'])

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self._active_connections = 0
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
    
    def record_query(self, provider: str, model: str, duration: float, tokens: int, cost: float):
        """Record query metrics"""
        QUERY_COUNT.labels(provider=provider, model=model).inc()
        QUERY_DURATION.labels(provider=provider).observe(duration)
        LLM_TOKEN_COUNT.labels(provider=provider, model=model).inc(tokens)
        LLM_COST.labels(provider=provider).inc(cost)
        LLM_LATENCY.labels(provider=provider, model=model).observe(duration)
    
    def record_embedding(self):
        """Record embedding generation"""
        EMBEDDING_COUNT.inc()
    
    def record_vector_search(self, duration: float):
        """Record vector search metrics"""
        VECTOR_SEARCH_DURATION.observe(duration)
    
    def update_document_count(self, count: int):
        """Update total document count"""
        DOCUMENT_COUNT.set(count)
    
    def record_error(self, component: str, error_type: str):
        """Record error metrics"""
        ERROR_COUNT.labels(component=component, error_type=error_type).inc()
    
    def record_document_processing(self, duration: float):
        """Record document processing time"""
        DOCUMENT_PROCESSING_DURATION.observe(duration)
    
    def record_chunks_processed(self, count: int = 1):
        """Record number of chunks processed"""
        CHUNK_COUNT.inc(count)
    
    def update_provider_availability(self, provider: str, available: bool):
        """Update provider availability status"""
        PROVIDER_AVAILABILITY.labels(provider=provider).set(1 if available else 0)
    
    def increment_active_connections(self):
        """Increment active connections"""
        self._active_connections += 1
        ACTIVE_CONNECTIONS.set(self._active_connections)
    
    def decrement_active_connections(self):
        """Decrement active connections"""
        self._active_connections = max(0, self._active_connections - 1)
        ACTIVE_CONNECTIONS.set(self._active_connections)

# Global metrics collector
metrics = MetricsCollector()

def track_time(metric_name: str, **labels):
    """Decorator to track execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics based on metric name
                if metric_name == 'query':
                    provider = labels.get('provider') or kwargs.get('provider', 'unknown')
                    QUERY_DURATION.labels(provider=provider).observe(duration)
                elif metric_name == 'vector_search':
                    VECTOR_SEARCH_DURATION.observe(duration)
                elif metric_name == 'document_processing':
                    DOCUMENT_PROCESSING_DURATION.observe(duration)
                elif metric_name == 'llm_request':
                    provider = labels.get('provider') or kwargs.get('provider', 'unknown')
                    model = labels.get('model') or kwargs.get('model', 'unknown')
                    LLM_LATENCY.labels(provider=provider, model=model).observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                # Record error
                component = labels.get('component', func.__name__)
                error_type = type(e).__name__
                metrics.record_error(component, error_type)
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics based on metric name
                if metric_name == 'embedding':
                    metrics.record_embedding()
                elif metric_name == 'document_processing':
                    DOCUMENT_PROCESSING_DURATION.observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                # Record error
                component = labels.get('component', func.__name__)
                error_type = type(e).__name__
                metrics.record_error(component, error_type)
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def track_llm_response(response_data: Dict[str, Any]):
    """Track LLM response metrics"""
    provider = response_data.get('provider', 'unknown')
    model = response_data.get('model', 'unknown')
    duration = response_data.get('latency_ms', 0) / 1000.0  # Convert to seconds
    tokens = response_data.get('tokens_used', 0)
    cost = response_data.get('cost_cents', 0.0)
    
    metrics.record_query(provider, model, duration, tokens, cost)

def track_provider_status(providers_status: Dict[str, bool]):
    """Track provider availability status"""
    for provider, available in providers_status.items():
        metrics.update_provider_availability(provider, available)

class ConnectionTracker:
    """Context manager to track active connections"""
    def __enter__(self):
        metrics.increment_active_connections()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        metrics.decrement_active_connections()

def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics server"""
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
        logger.info(f"Metrics available at http://localhost:{port}/metrics")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

# Middleware for FastAPI to track requests
async def metrics_middleware(request, call_next):
    """FastAPI middleware to track request metrics"""
    start_time = time.time()
    
    with ConnectionTracker():
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Extract endpoint from path
            endpoint = request.url.path
            method = request.method
            status = response.status_code
            
            metrics.record_request(method, endpoint, status, duration)
            
            return response
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_request(request.method, request.url.path, 500, duration)
            metrics.record_error('api', type(e).__name__)
            raise
    