# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class QueryComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class SearchType(str, Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class ResponseStyle(str, Enum):
    CONCISE = "concise"
    BALANCED = "balanced"
    DETAILED = "detailed"

class ExpertiseLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"

# Request Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    pdf_id: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = Field(5, ge=1, le=20)
    use_reranking: bool = True
    prefer_fast_response: bool = False
    preferred_provider: Optional[str] = None
    preferred_model: Optional[str] = None
    search_type: SearchType = SearchType.HYBRID

class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Conversation"
    document_id: Optional[str] = None

class UpdateUserProfileRequest(BaseModel):
    profile_updates: Dict[str, Any]

# Response Models
class SearchResult(BaseModel):
    text: str
    page_number: int
    relevance_score: float
    search_type: str

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SearchResult]] = []
    conversation_id: str
    provider_used: str
    model_used: str
    processing_time: float
    tokens_used: int
    cost_cents: float
    search_metadata: Dict[str, Any]

class UploadResponse(BaseModel):
    message: str
    pdf_id: str
    filename: str
    processing_time: float
    chunk_count: int
    metadata: Dict[str, Any]

class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = {}

class ConversationSummary(BaseModel):
    id: str
    title: str
    document_id: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime] = None
    message_count: int

class ConversationResponse(BaseModel):
    id: str
    title: str
    document_id: Optional[str]
    created_at: datetime

class DocumentSummary(BaseModel):
    id: str
    filename: str
    file_size: int
    page_count: int
    created_at: datetime
    metadata: Dict[str, Any]

class ProviderInfo(BaseModel):
    enabled: bool
    models: List[str]
    available: bool

class ProvidersStatus(BaseModel):
    providers: Dict[str, ProviderInfo]
    active_count: int
    default_provider: str

class UserProfileResponse(BaseModel):
    user_id: str
    profile: Dict[str, Any]
    updated: bool

class AnalyticsData(BaseModel):
    total_queries: int
    total_documents: int
    avg_response_time: float
    total_cost: float
    provider_usage: Dict[str, int]
    popular_queries: List[str]

class AnalyticsDashboard(BaseModel):
    user_analytics: AnalyticsData
    time_period_days: int
    last_updated: datetime

class QueryClassification(BaseModel):
    complexity: QueryComplexityLevel
    requires_table: bool
    table_names: Optional[List[str]] = []
    expertise_level: ExpertiseLevel
    response_style: ResponseStyle

class StructuredQueryResult(BaseModel):
    original_query: str
    classification: QueryClassification
    structured_query: str
