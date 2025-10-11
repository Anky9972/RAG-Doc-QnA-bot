# app/schemas.py
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import re
# Enum definitions
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

class DocumentType(Enum):
    """Document type classification"""
    ACADEMIC = "academic"
    FINANCIAL = "financial" 
    LEGAL = "legal"
    TECHNICAL = "technical"
    REPORT = "report"
    EDUCATIONAL = "educational"
    GENERAL = "general"

class QueryType(Enum):
    """Query classification types"""
    STRUCTURED = "structured"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    ANALYTICAL = "analytical"

# DataClass definitions for PDF processing
@dataclass
class TextChunk:
    """Represents a chunk of text from a document"""
    id: int
    text: str
    page_number: int
    chunk_type: str = 'text'
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ExtractedTable:
    """Represents a table extracted from a document"""
    page_number: int
    table_index: int
    headers: List[str]
    data: List[List[str]]
    table_type: str = 'general'
    confidence_score: float = 0.0
    extraction_method: str = 'unknown'

@dataclass
class ProcessedDocument:
    """Result of document processing"""
    text_chunks: List[TextChunk]
    tables: List[ExtractedTable]
    document_metadata: Dict[str, Any]
    document_type: DocumentType
    processing_method: str = 'unknown'

@dataclass
class QueryClassification:
    """Result of query classification"""
    query_type: QueryType
    confidence: float
    intent: str
    entities: List[Dict[str, Any]]
    suggested_response_format: str

@dataclass
class StructuredQueryResult:
    """Result from structured query processing"""
    data: List[Dict[str, Any]]
    total_matches: int
    metadata: Dict[str, Any]

# Request Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    pdf_id: Optional[str] = None  # Made optional to support general queries
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
    conversation_id: Optional[str] = None
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
    document_metadata: Optional[dict] = None  # ✅ Fix: allow missing

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



class ForgotPasswordRequest(BaseModel):
    """Request to initiate password reset"""
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    """Request to reset password with token"""
    token: str
    new_password: str
    
    @validator('new_password')
    def validate_password_strength(cls, v):
        """Validate password meets security requirements"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v

class PasswordResetResponse(BaseModel):
    """Response for password reset operations"""
    message: str
    success: bool

class ChangePasswordRequest(BaseModel):
    """Request to change password (for authenticated users)"""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_password_strength(cls, v):
        """Validate password meets security requirements"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
    
# Additional models for LlamaCloud integration
class ProcessingStatus(BaseModel):
    """Status of document processing"""
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress_percent: int
    message: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_details: Optional[str] = None

class ProcessingJobResponse(BaseModel):
    """Response for processing job creation"""
    job_id: str
    document_id: str
    status: str
    estimated_completion: Optional[datetime] = None
    processing_method: str

class DocumentImageInfo(BaseModel):
    """Information about extracted images"""
    id: str
    page_number: int
    image_type: str
    description: str
    confidence_score: float
    extracted_text: Optional[str] = None

class EnhancedDocumentSummary(DocumentSummary):
    """Extended document summary with processing info"""
    processing_method: str
    processing_quality: str
    extraction_metadata: Dict[str, Any]
    table_count: int
    image_count: int

# Query Classification Models (corrected)
class QueryClassificationRequest(BaseModel):
    """Request for query classification"""
    query: str
    document_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None

class QueryClassificationResponse(BaseModel):
    """Response from query classification"""
    query: str
    query_type: QueryType
    confidence: float
    intent: str
    entities: List[Dict[str, Any]]
    suggested_response_format: str
    processing_recommendation: str  # 'llamacloud', 'fallback', 'structured'