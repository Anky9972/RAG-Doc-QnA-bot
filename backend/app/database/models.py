# database/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, timezone
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    profile = Column(JSON, default={})  # Store user preferences, expertise level, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    documents = relationship("Document", back_populates="user")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False)
    file_size = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=False)
    document_metadata = Column(JSON, default={})
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # NEW: Processing method tracking
    processing_method = Column(String(50), default='fallback')  # 'llamacloud', 'fallback', 'hybrid'
    processing_quality = Column(String(20), default='standard')  # 'high', 'standard', 'low'
    processing_time_ms = Column(Integer)  # Time taken to process
    extraction_metadata = Column(JSON, default={})  # LlamaCloud-specific metadata
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships (unchanged)
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")
    conversations = relationship("Conversation", back_populates="document")
    tables = relationship("DocumentTable", back_populates="document", cascade="all, delete-orphan")
    images = relationship("DocumentImage", back_populates="document", cascade="all, delete-orphan")  # NEW

class DocumentImage(Base):
    """Store extracted image data and descriptions from PDFs"""
    __tablename__ = "document_images"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    image_index = Column(Integer, nullable=False)  # Multiple images per page
    
    # Image metadata
    image_name = Column(String(255))
    image_type = Column(String(50))  # 'diagram', 'chart', 'photo', 'screenshot'
    description = Column(Text)  # AI-generated description
    confidence_score = Column(Float, default=0.0)
    
    # Image properties
    width = Column(Integer)
    height = Column(Integer)
    file_size = Column(Integer)
    format = Column(String(10))  # 'png', 'jpg', etc.
    
    # Processing metadata
    extraction_method = Column(String(50), default='llamacloud')
    ai_description = Column(Text)  # Detailed AI description
    extracted_text = Column(Text)  # Any OCR text from image
    
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    
    # Relationships
    document = relationship("Document", back_populates="images")

class PasswordResetToken(Base):
    """Store password reset tokens"""
    __tablename__ = "password_reset_tokens"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    token = Column(String(255), unique=True, nullable=False, index=True)
    
    # Token metadata
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)
    used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Security tracking
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    
    def is_valid(self) -> bool:
        """Check if token is still valid"""
        if self.used:
            return False
        if datetime.now(timezone.utc) > self.expires_at:
            return False
        return True
    
    
class DocumentTable(Base):
    """Store extracted table data from PDFs"""
    __tablename__ = "document_tables"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    table_index = Column(Integer, nullable=False)
    
    # Raw table data
    table_data = Column(JSON, nullable=False)
    headers = Column(JSON)
    column_count = Column(Integer)
    row_count = Column(Integer)
    
    # Table metadata
    table_type = Column(String)
    confidence_score = Column(Float, default=0.0)
    extraction_method = Column(String)  # 'tabula', 'camelot', 'pdfplumber', 'llamacloud'
    
    # NEW: LlamaCloud enhancements
    markdown_representation = Column(Text)  # Raw markdown table
    structured_data = Column(JSON)  # Enhanced structured representation
    table_context = Column(Text)  # Surrounding text context
    processing_quality = Column(String(20), default='standard')  # 'high' for LlamaCloud
    
    # Processing metadata
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    processed = Column(Boolean, default=False)
    
    # Relationships
    document = relationship("Document", back_populates="tables")
    entities = relationship("TableEntity", back_populates="table")

class ProcessingJob(Base):
    """Track document processing jobs (especially async LlamaCloud jobs)"""
    __tablename__ = "processing_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Job details
    job_type = Column(String(50), nullable=False)  # 'llamacloud_parse', 'entity_extraction'
    status = Column(String(20), default='pending')  # 'pending', 'processing', 'completed', 'failed'
    
    # External job tracking
    external_job_id = Column(String(255))  # LlamaCloud job ID
    provider = Column(String(50))  # 'llamacloud', 'internal'
    
    # Progress tracking
    progress_percent = Column(Integer, default=0)
    current_step = Column(String(100))
    total_steps = Column(Integer)
    
    # Results and errors
    result_data = Column(JSON)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    estimated_completion = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    # Relationships
    document = relationship("Document")
    user = relationship("User")
    
class ProcessingConfig(Base):
    """Store processing configuration and preferences"""
    __tablename__ = "processing_configs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # NULL for global config
    
    # Configuration
    config_name = Column(String(100), nullable=False)
    config_type = Column(String(50), nullable=False)  # 'user_preference', 'system_setting'
    
    # Processing preferences
    prefer_llamacloud = Column(Boolean, default=True)
    min_file_size_for_llamacloud = Column(Integer, default=1048576)  # 1MB
    max_processing_time = Column(Integer, default=300)  # 5 minutes
    
    # Quality thresholds
    min_confidence_score = Column(Float, default=0.7)
    require_table_validation = Column(Boolean, default=False)
    require_image_descriptions = Column(Boolean, default=True)
    
    # Fallback behavior
    auto_fallback_on_failure = Column(Boolean, default=True)
    fallback_timeout_seconds = Column(Integer, default=60)
    
    # Configuration data
    settings = Column(JSON, default={})
    
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User")


class TableEntity(Base):
    """Store extracted entities from table data (subjects, teachers, etc.)"""
    __tablename__ = "table_entities"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    table_id = Column(String, ForeignKey("document_tables.id"), nullable=False)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    
    # Entity information
    entity_type = Column(String, nullable=False)  # 'subject', 'teacher', 'semester', 'credits', 'hours'
    entity_value = Column(String, nullable=False)  # The actual value
    
    # Table position
    row_index = Column(Integer)
    column_index = Column(Integer)
    cell_content = Column(Text)  # Original cell content
    
    # Relationships with other entities
    related_entities = Column(JSON)  # Links to other entities in same row/context
    
    # Metadata
    confidence_score = Column(Float, default=0.0)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    table = relationship("DocumentTable", back_populates="entities")
    document = relationship("Document")


class QueryPattern(Base):
    """Store common query patterns for structured data"""
    __tablename__ = "query_patterns"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Pattern information
    pattern_name = Column(String, nullable=False)
    pattern_regex = Column(String, nullable=False)
    query_type = Column(String, nullable=False)  # 'structured', 'semantic', 'analytical'
    
    # Pattern metadata
    description = Column(Text)
    example_queries = Column(JSON)  # List of example queries that match this pattern
    response_template = Column(JSON)  # Template for formatting responses
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=False)
    char_count = Column(Integer, nullable=False)
    chunk_metadata = Column(JSON, default={})
    
    # NEW: Enhanced chunk classification
    chunk_type = Column(String(50))  # 'text', 'table', 'header', 'list', 'figure_description'
    content_format = Column(String(20), default='text')  # 'text', 'markdown', 'structured'
    extraction_method = Column(String(50))  # 'llamacloud', 'fallback'
    quality_score = Column(Float)  # Processing quality assessment
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    title = Column(String(500), nullable=True)
    summary = Column(Text, nullable=True)
    document_metadata  = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    document = relationship("Document", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant'
    content = Column(Text, nullable=False)
    document_metadata  = Column(JSON, default={})  # Store provider, model, tokens, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    provider = Column(String(50), nullable=False)
    model = Column(String(100), nullable=False)
    latency_ms = Column(Integer, nullable=False)
    token_count = Column(Integer, default=0)
    cost_cents = Column(Float, default=0.0)
    retrieval_score = Column(Float, nullable=True)
    chunks_retrieved = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    document_metadata  = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

