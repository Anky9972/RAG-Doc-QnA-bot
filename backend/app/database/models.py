# database/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
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
    document_metadata  = Column(JSON, default={})
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")
    conversations = relationship("Conversation", back_populates="document")
    tables = relationship("DocumentTable", back_populates="document", cascade="all, delete-orphan")
    
class DocumentTable(Base):
    """Store extracted table data from PDFs"""
    __tablename__ = "document_tables"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    table_index = Column(Integer, nullable=False)  # Multiple tables per page
    
    # Raw table data
    table_data = Column(JSON, nullable=False)  # List of lists representing table rows
    headers = Column(JSON)  # Detected column headers
    column_count = Column(Integer)
    row_count = Column(Integer)
    
    # Table metadata
    table_type = Column(String)  # 'course_catalog', 'schedule', 'grades', etc.
    confidence_score = Column(Float, default=0.0)
    extraction_method = Column(String)  # 'tabula', 'camelot', 'pdfplumber'
    
    # Processing metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    
    # Relationships
    document = relationship("Document", back_populates="tables")
    entities = relationship("TableEntity", back_populates="table")

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
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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

