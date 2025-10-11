# main.py
import asyncio
import logging
import time
import uuid
import secrets
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
from app.database.models import User, Document, QueryLog, Conversation, Message, UserSession, DocumentTable, TableEntity, DocumentChunk
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks, APIRouter, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
import psutil
import re
import os
from app.services.email_service import email_service
from app.database.models import PasswordResetToken
from app.schemas import (
    ForgotPasswordRequest, 
    ResetPasswordRequest, 
    PasswordResetResponse,
    ChangePasswordRequest
)
from app.services.llamacloud_processor import (
    get_llamacloud_processor,  # Changed from llamacloud_processor 
    is_llamacloud_available, 
    should_use_llamacloud,
    LlamaCloudError
)

from app.core import config
from app.database.connection import init_db, get_db, get_db_session
from app.database.models import User, Document, QueryLog, Conversation, Message, UserSession, DocumentTable, TableEntity
from app.schemas import *
from app.schemas import QueryClassification, StructuredQueryResult
from app.services import pdf_processor, vector_store
from app.services.llm_orchestrator import orchestrator
from app.services.hybrid_search import hybrid_search_engine
from app.services.memory_manager import memory_manager
from app.utils.analytics import analytics_logger
from app.utils.rate_limiter import rate_limiter
from app.services.pdf_processor import enhanced_pdf_processor
from app.services.query_classifier import query_classifier, structured_query_processor
from app.database.models import DocumentTable, TableEntity, QueryPattern
from app.services.email_service import email_service
from app.database.models import PasswordResetToken
from app.schemas import (
    ForgotPasswordRequest, 
    ResetPasswordRequest, 
    PasswordResetResponse,
    ChangePasswordRequest
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security setup
SECRET_KEY = config.JWT_SECRET_KEY if hasattr(config, 'JWT_SECRET_KEY') else "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

# Auth Pydantic models
class UserSignup(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: dict

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    profile: dict
    created_at: datetime

# Password utilities
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# JWT utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("Starting PDF Q&A System...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized")
        
        # Initialize Weaviate client
        client = vector_store.get_weaviate_client()
        if client:
            logger.info("Weaviate client initialized")
        
        # Initialize embedding models
        vector_store.get_embedding_model()
        logger.info("Embedding model loaded")
        
        # Initialize LLM orchestrator
        provider_stats = orchestrator.get_provider_stats()
        logger.info(f"LLM providers initialized: {list(provider_stats.keys())}")
        
        # Initialize hybrid search
        logger.info("Hybrid search engine initialized")
        
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Cleanup
    logger.info("Shutting down...")
    try:
        # Close Weaviate connection
        global_client = getattr(vector_store, 'weaviate_client', None)
        if global_client and hasattr(global_client, 'close'):
            global_client.close()
            logger.info("Weaviate client closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

app = FastAPI(
    title="Enhanced PDF Q&A System with Authentication",
    description="Multi-model RAG system with hybrid search, conversation memory, and user authentication",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    user_id = verify_token(token)
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Optional authentication for backward compatibility
async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None"""
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None

def get_user_id_from_user(user: Optional[User]) -> str:
    """Extract user ID from user object or return anonymous"""
    return user.id if user else "anonymous"

def get_weaviate_client():
    """Get Weaviate client dependency"""
    try:
        client = vector_store.get_weaviate_client()
        if not client or not client.is_connected():
            raise HTTPException(status_code=503, detail="Vector database unavailable")
        return client
    except Exception as e:
        logger.error(f"Weaviate client error: {e}")
        raise HTTPException(status_code=503, detail="Vector database service error")

# Authentication endpoints
@app.post("/auth/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """Register a new user"""
    
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.email == user_data.email) | (User.username == user_data.username)
    ).first()
    
    if existing_user:
        if existing_user.email == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Create new user
    hashed_password = hash_password(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        profile={"password_hash": hashed_password}
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create access token
    access_token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(
        data={"sub": new_user.id}, expires_delta=access_token_expires
    )
    
    # Create user session
    session_token = secrets.token_urlsafe(32)
    user_session = UserSession(
        user_id=new_user.id,
        session_token=session_token,
        expires_at=datetime.now(timezone.utc) + access_token_expires
    )
    db.add(user_session)
    db.commit()
    
    logger.info(f"New user registered: {new_user.username} ({new_user.email})")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        user={
            "id": new_user.id,
            "username": new_user.username,
            "email": new_user.email,
            "profile": {k: v for k, v in new_user.profile.items() if k != "password_hash"},
            "created_at": new_user.created_at.isoformat()
        }
    )

@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return access token"""
    
    # Find user by email
    user = db.query(User).filter(User.email == user_credentials.email).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    stored_password_hash = user.profile.get("password_hash")
    if not stored_password_hash or not verify_password(user_credentials.password, stored_password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create access token
    access_token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    
    # Create or update user session
    existing_session = db.query(UserSession).filter(UserSession.user_id == user.id).first()
    if existing_session:
        existing_session.session_token = secrets.token_urlsafe(32)
        existing_session.expires_at = datetime.now(timezone.utc) + access_token_expires
    else:
        user_session = UserSession(
            user_id=user.id,
            session_token=secrets.token_urlsafe(32),
            expires_at=datetime.now(timezone.utc) + access_token_expires
        )
        db.add(user_session)
    
    db.commit()
    
    logger.info(f"User logged in: {user.username} ({user.email})")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        user={
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "profile": {k: v for k, v in user.profile.items() if k != "password_hash"},
            "created_at": user.created_at.isoformat()
        }
    )

@app.post("/auth/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout user and invalidate session"""
    
    # Remove user session
    user_session = db.query(UserSession).filter(UserSession.user_id == current_user.id).first()
    if user_session:
        db.delete(user_session)
        db.commit()
    
    logger.info(f"User logged out: {current_user.username}")
    
    return {"message": "Successfully logged out"}

@app.post("/auth/forgot-password", response_model=PasswordResetResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Initiate password reset process
    
    - Validates email exists
    - Generates reset token
    - Sends reset email
    - Rate limited to prevent abuse
    """
    try:
        # Find user by email
        user = db.query(User).filter(User.email == request.email).first()
        
        # Always return success to prevent email enumeration
        # But only send email if user exists
        if user:
            # Check for existing valid tokens
            existing_token = db.query(PasswordResetToken).filter(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.used == False,
                PasswordResetToken.expires_at > datetime.now(timezone.utc)
            ).first()
            
            if existing_token:
                # Token already exists and is valid
                logger.info(f"Valid reset token already exists for user: {user.email}")
            else:
                # Generate new reset token
                reset_token = secrets.token_urlsafe(32)
                expires_at = datetime.now(timezone.utc) + timedelta(hours=1)  # 1 hour expiry
                
                # Store token in database
                password_reset_token = PasswordResetToken(
                    user_id=user.id,
                    token=reset_token,
                    expires_at=expires_at
                )
                
                db.add(password_reset_token)
                db.commit()
                
                # Send reset email in background
                if email_service.is_enabled():
                    background_tasks.add_task(
                        email_service.send_password_reset_email,
                        to_email=user.email,
                        username=user.username,
                        reset_token=reset_token,
                        expires_in_minutes=60
                    )
                    logger.info(f"Password reset email queued for: {user.email}")
                else:
                    logger.warning(f"Email service disabled. Reset token for {user.email}: {reset_token}")
        else:
            logger.info(f"Password reset requested for non-existent email: {request.email}")
        
        # Always return success message (prevent email enumeration)
        return PasswordResetResponse(
            message="If an account with that email exists, a password reset link has been sent.",
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error in forgot password: {e}", exc_info=True)
        # Still return success to prevent information leakage
        return PasswordResetResponse(
            message="If an account with that email exists, a password reset link has been sent.",
            success=True
        )


@app.post("/auth/reset-password", response_model=PasswordResetResponse)
async def reset_password(
    request: ResetPasswordRequest,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Reset password using token
    
    - Validates reset token
    - Updates user password
    - Invalidates token
    - Sends confirmation email
    """
    try:
        # Find and validate token
        reset_token = db.query(PasswordResetToken).filter(
            PasswordResetToken.token == request.token,
            PasswordResetToken.used == False
        ).first()
        
        if not reset_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Check if token is expired
        if not reset_token.is_valid():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token has expired. Please request a new one."
            )
        
        # Get user
        user = db.query(User).filter(User.id == reset_token.user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Hash new password
        hashed_password = hash_password(request.new_password)
        
        # ✅ FIX: Update user password correctly
        user.password_hash = hashed_password  # Changed from user.profile["password_hash"]
        
        # Optional: Update password_changed_at timestamp if you have this field
        # user.password_changed_at = datetime.now(timezone.utc)
        
        # Mark token as used
        reset_token.used = True
        reset_token.used_at = datetime.now(timezone.utc)
        
        # Invalidate all user sessions (force re-login)
        db.query(UserSession).filter(UserSession.user_id == user.id).delete()
        
        # Commit changes
        db.commit()
        
        logger.info(f"Password reset successful for user: {user.username}")
        
        # Send confirmation email in background
        if email_service.is_enabled():
            background_tasks.add_task(
                email_service.send_password_changed_confirmation,
                to_email=user.email,
                username=user.username
            )
        
        return PasswordResetResponse(
            message="Password reset successfully. Please login with your new password.",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in reset password: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password. Please try again."
        )

@app.post("/auth/change-password", response_model=PasswordResetResponse)
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Change password for authenticated user
    
    - Requires authentication
    - Validates current password
    - Updates to new password
    - Sends confirmation email
    """
    try:
        # Verify current password
        stored_password_hash = current_user.profile.get("password_hash")
        
        if not stored_password_hash or not verify_password(request.current_password, stored_password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        
        # Check if new password is same as current
        if verify_password(request.new_password, stored_password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password"
            )
        
        # Hash new password
        hashed_password = hash_password(request.new_password)
        
        # Update user password
        current_user.profile["password_hash"] = hashed_password
        
        # Invalidate other sessions (keep current session active)
        db.query(UserSession).filter(
            UserSession.user_id == current_user.id
        ).delete()
        
        # Create new session for current user
        access_token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
        session_token = secrets.token_urlsafe(32)
        user_session = UserSession(
            user_id=current_user.id,
            session_token=session_token,
            expires_at=datetime.now(timezone.utc) + access_token_expires
        )
        db.add(user_session)
        
        db.commit()
        
        logger.info(f"Password changed successfully for user: {current_user.username}")
        
        # Send confirmation email in background
        if email_service.is_enabled():
            background_tasks.add_task(
                email_service.send_password_changed_confirmation,
                to_email=current_user.email,
                username=current_user.username
            )
        
        return PasswordResetResponse(
            message="Password changed successfully",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in change password: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password. Please try again."
        )


@app.get("/auth/validate-reset-token/{token}")
async def validate_reset_token(
    token: str,
    db: Session = Depends(get_db)
):
    """
    Validate if a reset token is still valid
    
    - Useful for frontend to check token before showing reset form
    """
    try:
        reset_token = db.query(PasswordResetToken).filter(
            PasswordResetToken.token == token,
            PasswordResetToken.used == False
        ).first()
        
        if not reset_token:
            return {
                "valid": False,
                "message": "Invalid token"
            }
        
        if not reset_token.is_valid():
            return {
                "valid": False,
                "message": "Token has expired"
            }
        
        # Get user to return username
        user = db.query(User).filter(User.id == reset_token.user_id).first()
        
        return {
            "valid": True,
            "message": "Token is valid",
            "username": user.username if user else None,
            "expires_at": reset_token.expires_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error validating reset token: {e}")
        return {
            "valid": False,
            "message": "Error validating token"
        }
    
@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        profile={k: v for k, v in current_user.profile.items() if k != "password_hash"},
        created_at=current_user.created_at
    )

@app.get("/auth/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Refresh access token"""
    
    # Create new access token
    access_token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(
        data={"sub": current_user.id}, expires_delta=access_token_expires
    )
    
    # Update user session
    user_session = db.query(UserSession).filter(UserSession.user_id == current_user.id).first()
    if user_session:
        user_session.expires_at = datetime.now(timezone.utc) + access_token_expires
        db.commit()
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        user={
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "profile": {k: v for k, v in current_user.profile.items() if k != "password_hash"},
            "created_at": current_user.created_at.isoformat()
        }
    )

# Health check router
health_router = APIRouter()

@health_router.get("/health")
async def health_check():
    """Comprehensive health check including LlamaCloud status"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    # Database connectivity
    try:
        db = get_db_session()
        db.execute(text("SELECT 1"))
        db.close()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Weaviate connectivity
    try:
        client = vector_store.get_weaviate_client()
        if client and client.is_ready():
            health_status["checks"]["weaviate"] = "healthy"
        else:
            health_status["checks"]["weaviate"] = "unhealthy: not ready"
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["weaviate"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # LLM providers
    try:
        provider_stats = orchestrator.get_provider_stats()
        available_providers = [p for p, stats in provider_stats.items() if stats.get('available')]
        
        if available_providers:
            health_status["checks"]["llm_providers"] = f"healthy: {len(available_providers)} available"
        else:
            health_status["checks"]["llm_providers"] = "unhealthy: no providers available"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["llm_providers"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # NEW: LlamaCloud processing
    try:
        if is_llamacloud_available():
            health_status["checks"]["llamacloud"] = "healthy: API accessible"
        else:
            health_status["checks"]["llamacloud"] = "degraded: fallback available"
            if health_status["status"] == "healthy":
                health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["llamacloud"] = f"error: {str(e)}"
    
    # System resources
    try:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        health_status["checks"]["system_resources"] = {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory_usage,
            "disk_usage_percent": disk_usage,
            "status": "healthy" if cpu_usage < 90 and memory_usage < 90 and disk_usage < 90 else "degraded"
        }
        
        if cpu_usage > 95 or memory_usage > 95 or disk_usage > 95:
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["system_resources"] = f"error: {str(e)}"
    
    return health_status


@health_router.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")

@health_router.get("/stats")
async def get_app_stats():
    """Get detailed application statistics"""
    try:
        db = get_db_session()
        
        # Basic counts
        total_documents = db.query(Document).count()
        total_queries = db.query(QueryLog).count()
        total_conversations = db.query(Conversation).count()
        total_users = db.query(User).count()
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        
        recent_queries = db.query(QueryLog).filter(
            QueryLog.created_at >= recent_cutoff
        ).count()
        
        recent_documents = db.query(Document).filter(
            Document.created_at >= recent_cutoff
        ).count()
        
        recent_users = db.query(User).filter(
            User.created_at >= recent_cutoff
        ).count()
        
        # Provider usage
        provider_usage = db.query(
            QueryLog.provider,
            func.count(QueryLog.id).label('count')
        ).group_by(QueryLog.provider).all()
        
        # Average metrics
        avg_latency = db.query(func.avg(QueryLog.latency_ms)).scalar() or 0
        total_cost = db.query(func.sum(QueryLog.cost_cents)).scalar() or 0
        
        db.close()
        
        return {
            "totals": {
                "users": total_users,
                "documents": total_documents,
                "queries": total_queries,
                "conversations": total_conversations
            },
            "recent_24h": {
                "users": recent_users,
                "queries": recent_queries,
                "documents": recent_documents
            },
            "performance": {
                "avg_query_latency_ms": float(avg_latency),
                "total_cost_cents": float(total_cost)
            },
            "provider_usage": {p.provider: p.count for p in provider_usage},
            "system_info": {
                "uptime_seconds": time.time() - getattr(get_app_stats, 'start_time', time.time()),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting app stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

# Register the health router
app.include_router(health_router, prefix="", tags=["health"])


@app.post("/upload_pdf/", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    client = Depends(get_weaviate_client),
    force_fallback: bool = Query(False, description="Force use of fallback processing")
):
    """Enhanced PDF upload with intelligent chunking"""
    
    user_id = current_user.id
    
    # Rate limiting
    if not rate_limiter.check_upload_limit(user_id):
        raise HTTPException(status_code=429, detail="Upload rate limit exceeded")
    
    # Validate file
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    try:
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        logger.info(f"Processing PDF: {file.filename} for user: {current_user.username}")
        
        # Read file content
        contents = await file.read()
        
        processed_document = None
        processing_method = "fallback"
        
        # Try LlamaCloud processing first
        if not force_fallback and should_use_llamacloud(contents, file.filename):
            try:
                logger.info(f"Using LlamaCloud for {file.filename}")
                
                llamacloud_processor = get_llamacloud_processor()
                
                if llamacloud_processor and llamacloud_processor.is_available():
                    processed_document = llamacloud_processor.process_pdf_comprehensive(
                        contents, file.filename
                    )
                    processing_method = "llamacloud"
                    logger.info(f"✓ LlamaCloud processed: {len(processed_document.text_chunks)} chunks")
                else:
                    logger.warning("LlamaCloud processor not available")
                    
            except LlamaCloudError as e:
                logger.warning(f"LlamaCloud failed: {e}, using fallback")
            except Exception as e:
                logger.error(f"LlamaCloud error: {e}, using fallback")
        
        # Fallback processing
        if processed_document is None:
            logger.info(f"Using fallback processing for {file.filename}")
            try:
                from app.services.pdf_processor import enhanced_pdf_processor
                processed_document = enhanced_pdf_processor.process_pdf_comprehensive(contents)
                processing_method = "fallback"
            except Exception as e:
                logger.error(f"Fallback processing failed: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"PDF processing failed: {str(e)}"
                )
        
        # Check for duplicate
        pdf_hash = processed_document.document_metadata.get('pdf_hash', '')
        existing_doc = db.query(Document).filter(
            Document.file_hash == pdf_hash,
            Document.user_id == user_id
        ).first()
        
        if existing_doc:
            raise HTTPException(status_code=400, detail="Document already exists")
        
        # Generate embeddings - CRITICAL: Use the intelligent chunks
        logger.info(f"Generating embeddings for {len(processed_document.text_chunks)} chunks...")
        embeddings = vector_store.embed_chunks(processed_document.text_chunks)
        
        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
        
        logger.info(f"✓ Generated {len(embeddings)} embeddings")
        
        # Store in Weaviate - using the SAME intelligent chunks
        logger.info(f"Storing in Weaviate...")
        vector_store.store_embeddings(client, document_id, processed_document.text_chunks, embeddings)
        logger.info(f"✓ Stored in Weaviate")
        
        # Save to PostgreSQL
        document_metadata = processed_document.document_metadata.copy()
        document_metadata['processing_method'] = processing_method
        document_metadata['chunk_strategy'] = 'intelligent' if processing_method == 'llamacloud' else 'standard'
        
        document = Document(
            id=document_id,
            filename=file.filename,
            file_hash=pdf_hash,
            file_size=len(contents),
            page_count=document_metadata.get('page_count', 0),
            document_metadata=document_metadata,
            user_id=user_id,
            processing_method=processing_method,
            processing_quality='high' if processing_method == 'llamacloud' else 'standard',
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
        db.add(document)
        db.flush()
        
        # Store chunks in PostgreSQL - SAME intelligent chunks as Weaviate
        logger.info(f"Storing {len(processed_document.text_chunks)} chunks in PostgreSQL...")
        chunk_count = 0
        
        for chunk in processed_document.text_chunks:
            db_chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk.id,
                page_number=chunk.page_number,
                content=chunk.text,
                word_count=len(chunk.text.split()),
                char_count=len(chunk.text),
                chunk_metadata=chunk.metadata or {},
                chunk_type=chunk.chunk_type,
                content_format='markdown' if processing_method == 'llamacloud' else 'text',
                extraction_method=processing_method,
                quality_score=chunk.metadata.get('quality_score', 0.8)
            )
            db.add(db_chunk)
            chunk_count += 1
        
        logger.info(f"✓ Stored {chunk_count} chunks in PostgreSQL")
        
        # Store extracted tables
        table_count = 0
        for table in processed_document.tables:
            db_table = DocumentTable(
                document_id=document_id,
                page_number=table.page_number,
                table_index=table.table_index,
                table_data=table.data,
                headers=table.headers,
                column_count=len(table.data[0]) if table.data else 0,
                row_count=len(table.data),
                table_type=table.table_type,
                confidence_score=table.confidence_score,
                extraction_method=table.extraction_method,
                markdown_representation='|'.join(table.headers) if table.headers else '',
                processing_quality='high' if processing_method == 'llamacloud' else 'standard',
                processed=False
            )
            db.add(db_table)
            table_count += 1
        
        logger.info(f"✓ Stored {table_count} tables")
        
        # Commit everything
        db.commit()
        
        processing_time = time.time() - start_time
        
        # Background task for entity extraction
        if table_count > 0:
            background_tasks.add_task(
                process_table_entities,
                document_id, user_id
            )
        
        # Log analytics
        background_tasks.add_task(
            analytics_logger.log_document_upload,
            user_id, document_id, file.filename, processing_time
        )
        
        logger.info(
            f"✓✓✓ Successfully processed {file.filename} in {processing_time:.2f}s\n"
            f"    Method: {processing_method}\n"
            f"    Chunks: {chunk_count} (intelligent, not page-based)\n"
            f"    Tables: {table_count}\n"
            f"    Vector DB: ✓  PostgreSQL: ✓"
        )
        
        return UploadResponse(
            message=f"PDF processed successfully using {processing_method}",
            pdf_id=document_id,
            filename=file.filename,
            processing_time=processing_time,
            chunk_count=chunk_count,
            metadata={
                **document_metadata,
                'tables_extracted': table_count,
                'document_type': processed_document.document_type.value,
                'processing_method': processing_method,
                'chunk_strategy': 'intelligent' if processing_method == 'llamacloud' else 'standard',
                'llamacloud_available': is_llamacloud_available(),
                'same_chunks_everywhere': True  # Confirmation that chunks match
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
@app.get("/processing/status", response_model=Dict[str, Any])
async def get_processing_status():
    """Get status of available PDF processing methods"""
    try:
        # Get the processor instance and check its status
        llamacloud_processor = get_llamacloud_processor()
        
        if llamacloud_processor:
            llamacloud_stats = llamacloud_processor.get_processing_stats()
        else:
            llamacloud_stats = {
                "service": "llamacloud",
                "status": "unavailable",
                "error": "Processor not initialized"
            }
        
        return {
            "llamacloud": llamacloud_stats,
            "fallback": {
                "service": "enhanced_pdf_processor",
                "status": "healthy",
                "capabilities": [
                    "basic_tables",
                    "text_extraction",
                    "metadata_extraction"
                ],
                "supported_formats": ["pdf"]
            },
            "recommendations": {
                "use_llamacloud_for": [
                    "complex_layouts",
                    "image_heavy_documents", 
                    "scientific_papers",
                    "financial_reports",
                    "technical_documentation"
                ],
                "use_fallback_for": [
                    "simple_text_documents",
                    "when_llamacloud_unavailable"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get processing status")

# NEW: Background task for processing table entities
async def process_table_entities(document_id: str, user_id: str):
    """Background task to extract entities from tables"""
    try:
        logger.info(f"Processing table entities for document {document_id}")
        
        db = get_db_session()
        
        # Get all unprocessed tables for this document
        tables = db.query(DocumentTable).filter(
            DocumentTable.document_id == document_id,
            DocumentTable.processed == False
        ).all()
        
        entity_count = 0
        for table in tables:
            # Extract entities from table data
            entities = extract_entities_from_table(table)
            
            for entity_data in entities:
                entity = TableEntity(
                    table_id=table.id,
                    document_id=document_id,
                    entity_type=entity_data['type'],
                    entity_value=entity_data['value'],
                    row_index=entity_data.get('row_index'),
                    column_index=entity_data.get('column_index'),
                    cell_content=entity_data.get('cell_content', ''),
                    confidence_score=entity_data.get('confidence', 0.0),
                    related_entities=entity_data.get('related_entities', {})
                )
                db.add(entity)
                entity_count += 1
            
            # Mark table as processed
            table.processed = True
        
        db.commit()
        db.close()
        
        logger.info(f"Processed {entity_count} entities from {len(tables)} tables for document {document_id}")
        
    except Exception as e:
        logger.error(f"Error processing table entities: {e}", exc_info=True)


def extract_entities_from_table(table: DocumentTable) -> List[Dict[str, Any]]:
    """Extract entities from a single table"""
    entities = []
    
    if not table.table_data:
        return entities
    
    # Define entity extraction patterns
    entity_patterns = {
        'subject': [
            r'[A-Z]{2,4}\s*\d{3,4}',  # Course codes
            r'(mathematics|math|physics|chemistry|biology|english|history|computer\s+science)',
        ],
        'teacher': [
            r'(dr\.|prof\.|mr\.|ms\.|mrs\.)\s*[a-zA-Z\s]+',
            r'[a-zA-Z]+\s*,\s*[a-zA-Z]+',  # Last, First format
        ],
        'credits': [
            r'(\d+)\s*(credits?|cr)',
        ],
        'hours': [
            r'(\d+)\s*(hours?|hrs?)',
        ],
        'semester': [
            r'(semester\s*\d+|sem\s*\d+)',
        ]
    }
    
    for row_idx, row_data in enumerate(table.table_data):
        for col_idx, cell in enumerate(row_data):
            cell_text = str(cell).strip()
            
            if not cell_text:
                continue
            
            # Try to match against all patterns
            for entity_type, patterns in entity_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, cell_text, re.IGNORECASE)
                    if match:
                        entities.append({
                            'type': entity_type,
                            'value': match.group().strip(),
                            'row_index': row_idx,
                            'column_index': col_idx,
                            'cell_content': cell_text,
                            'confidence': 0.8,
                            'related_entities': {}  # Will be populated later
                        })
    
    return entities


# REPLACE your existing query endpoint in main.py with this enhanced version
@app.get("/documents/{document_id}/tables", response_model=List[Dict[str, Any]])
async def get_document_tables(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get extracted tables for a document"""
    try:
        # Verify document ownership
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        tables = db.query(DocumentTable).filter(
            DocumentTable.document_id == document_id
        ).all()
        
        return [
            {
                "id": table.id,
                "page_number": table.page_number,
                "table_index": table.table_index,
                "table_type": table.table_type,
                "confidence_score": table.confidence_score,
                "extraction_method": table.extraction_method,
                "column_count": table.column_count,
                "row_count": table.row_count,
                "headers": table.headers,
                "sample_data": table.table_data[:3] if table.table_data else [],  # First 3 rows
                "processed": table.processed
            }
            for table in tables
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document tables: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tables")

@app.get("/documents/{document_id}/entities", response_model=List[Dict[str, Any]])
async def get_document_entities(
    document_id: str,
    entity_type: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get extracted entities for a document"""
    try:
        # Verify document ownership
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        query = db.query(TableEntity).filter(
            TableEntity.document_id == document_id
        )
        
        if entity_type:
            query = query.filter(TableEntity.entity_type == entity_type)
        
        entities = query.all()
        
        return [
            {
                "id": entity.id,
                "entity_type": entity.entity_type,
                "entity_value": entity.entity_value,
                "confidence_score": entity.confidence_score,
                "table_id": entity.table_id,
                "row_index": entity.row_index,
                "column_index": entity.column_index,
                "related_entities": entity.related_entities
            }
            for entity in entities
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document entities: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch entities")
@app.post("/query/structured", response_model=Dict[str, Any])
async def structured_query_only(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Endpoint specifically for structured queries - returns raw structured data"""
    try:
        # Verify document ownership
        if request.pdf_id:
            document = db.query(Document).filter(
                Document.id == request.pdf_id,
                Document.user_id == current_user.id
            ).first()
            
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
        
        from app.services.query_classifier import structured_query_processor
        
        result = await structured_query_processor.process_structured_query(
            request.query, request.pdf_id, db
        )
        
        return {
            "query": request.query,
            "results": result.data,
            "total_matches": result.total_matches,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"Structured query error: {e}")
        raise HTTPException(status_code=500, detail=f"Structured query failed: {str(e)}")


@app.get("/query/classify/{query_text}")
async def classify_query_endpoint(query_text: str):
    """Endpoint to classify a query and see how it would be processed"""
    from app.services.query_classifier import query_classifier
    
    classification = query_classifier.classify_query(query_text)
    
    return {
        "query": query_text,
        "query_type": classification.query_type.value,
        "confidence": classification.confidence,
        "intent": classification.intent,
        "entities": classification.entities,
        "suggested_response_format": classification.suggested_response_format
    }

@app.post("/query/", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    client = Depends(get_weaviate_client)
):
    """Enhanced query with authentication, multi-model routing, memory, and structured data support"""
    
    user_id = current_user.id
    
    # Rate limiting
    if not rate_limiter.check_query_limit(user_id):
        raise HTTPException(status_code=429, detail="Query rate limit exceeded")
    
    try:
        start_time = time.time()
        
        # Verify document ownership
        if request.pdf_id:
            document = db.query(Document).filter(
                Document.id == request.pdf_id,
                Document.user_id == user_id
            ).first()
            
            if not document:
                raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        # NEW: Classify the query to determine processing approach
        from app.services.query_classifier import query_classifier, structured_query_processor
        
        classification = query_classifier.classify_query(request.query)
        
        logger.info(f"Query classified as {classification.query_type.value} with confidence {classification.confidence:.2f}")
        
        # Handle structured queries
        if classification.query_type in ['structured', 'analytical'] and classification.confidence > 0.6:
            return await handle_structured_query(
                request, classification, current_user, db, background_tasks, start_time
            )
        
        # Handle hybrid queries (structured + semantic)
        elif classification.query_type == 'hybrid':
            return await handle_hybrid_query(
                request, classification, current_user, db, client, background_tasks, start_time
            )
        
        # Default to semantic search for everything else
        else:
            return await handle_semantic_query(
                request, current_user, db, client, background_tasks, start_time
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


async def handle_structured_query(
    request: QueryRequest,
    classification: QueryClassification,
    current_user: User,
    db: Session,
    background_tasks: BackgroundTasks,
    start_time: float
) -> QueryResponse:
    """Handle purely structured queries"""
    
    try:
        # Process structured query
        structured_result = await structured_query_processor.process_structured_query(
            request.query, request.pdf_id, db
        )
        
        if not structured_result.data:
            # Fallback to semantic search if no structured data found
            logger.info("No structured data found, falling back to semantic search")
            return await handle_semantic_query(request, current_user, db, None, background_tasks, start_time)
        
        # Format structured response into natural language
        formatted_response = format_structured_response(
            structured_result, classification, request.query
        )
        
        processing_time = time.time() - start_time
        
        # Log query analytics
        background_tasks.add_task(
            analytics_logger.log_query,
            current_user.id, request.query, formatted_response,
            "structured_processor", "table_data",
            processing_time * 1000, 0, 0.0, len(structured_result.data)
        )
        
        return QueryResponse(
            answer=formatted_response,
            sources=format_structured_sources(structured_result),
            conversation_id=None,  # Structured queries don't use conversation context yet
            provider_used="structured_processor",
            model_used="table_data",
            processing_time=processing_time,
            tokens_used=0,
            cost_cents=0.0,
            search_metadata={
                "total_chunks_found": structured_result.total_matches,
                "search_type": "structured",
                "query_classification": classification.query_type.value,
                "entities_found": len(classification.entities)
            }
        )
        
    except Exception as e:
        logger.error(f"Structured query processing failed: {e}")
        # Fallback to semantic search
        return await handle_semantic_query(request, current_user, db, None, background_tasks, start_time)


async def handle_hybrid_query(
    request: QueryRequest,
    classification: QueryClassification,
    current_user: User,
    db: Session,
    client,
    background_tasks: BackgroundTasks,
    start_time: float
) -> QueryResponse:
    """Handle hybrid queries that combine structured and semantic search"""
    
    structured_data = []
    semantic_results = []
    
    try:
        # Get structured data
        structured_result = await structured_query_processor.process_structured_query(
            request.query, request.pdf_id, db
        )
        structured_data = structured_result.data[:3]  # Top 3 structured results
        
    except Exception as e:
        logger.warning(f"Structured component failed in hybrid query: {e}")
    
    try:
        # Get semantic search results
        search_results = await hybrid_search_engine.hybrid_search(
            client=client,
            query=request.query,
            document_id=request.pdf_id,
            top_k=5,
            use_reranking=request.use_reranking
        )
        semantic_results = search_results
        
    except Exception as e:
        logger.warning(f"Semantic component failed in hybrid query: {e}")
    
    # Combine both types of context
    combined_context = build_hybrid_context(
        structured_data, semantic_results, request.query, classification
    )
    
    # Generate response using LLM with combined context
    llm_response = await orchestrator.generate_response(
        prompt=combined_context,
        context_chunks=semantic_results,
        prefer_fast=request.prefer_fast_response,
        provider=request.preferred_provider,
        model=request.preferred_model
    )
    
    processing_time = time.time() - start_time
    
    # Combine sources from both structured and semantic
    all_sources = []
    
    # Add structured sources
    for item in structured_data:
        all_sources.append({
            "text": f"Table data: {str(item.get('raw_data', item))}",
            "page_number": item.get('page_number', 0),
            "relevance_score": 0.9,  # High relevance for structured matches
            "search_type": "structured"
        })
    
    # Add semantic sources
    for result in semantic_results:
        all_sources.append({
            "text": result.get("text", "")[:200] + "...",
            "page_number": result.get("page_number"),
            "relevance_score": result.get("combined_score", 0),
            "search_type": result.get("search_type", "semantic")
        })
    
    # Log analytics
    background_tasks.add_task(
        analytics_logger.log_query,
        current_user.id, request.query, llm_response.content,
        llm_response.provider, llm_response.model,
        processing_time * 1000, llm_response.tokens_used,
        llm_response.cost_cents, len(all_sources)
    )
    
    return QueryResponse(
        answer=llm_response.content,
        sources=all_sources[:10],  # Limit to top 10 sources
        conversation_id=None,
        provider_used=llm_response.provider,
        model_used=llm_response.model,
        processing_time=processing_time,
        tokens_used=llm_response.tokens_used,
        cost_cents=llm_response.cost_cents,
        search_metadata={
            "total_chunks_found": len(semantic_results) + len(structured_data),
            "search_type": "hybrid_structured_semantic",
            "structured_results": len(structured_data),
            "semantic_results": len(semantic_results)
        }
    )


async def handle_semantic_query(
    request: QueryRequest,
    current_user: User,
    db: Session,
    client,
    background_tasks: BackgroundTasks,
    start_time: float
) -> QueryResponse:
    """Handle traditional semantic queries (your existing logic)"""
    
    # Get or create conversation
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = await memory_manager.create_conversation(
            current_user.id, request.pdf_id
        )
    
    # Get conversation context and user preferences
    conversation_context, user_prefs = await memory_manager.get_conversation_context(
        conversation_id, current_user.id
    )
    
    logger.info(f"Semantic Query: '{request.query}' | User: {current_user.username} | Doc: {request.pdf_id}")
    
    # Perform hybrid search
    search_results = await hybrid_search_engine.hybrid_search(
        client=client,
        query=request.query,
        document_id=request.pdf_id,
        top_k=request.top_k or 5,
        use_reranking=request.use_reranking
    )
    
    # Build enhanced prompt with conversation context
    enhanced_prompt = build_enhanced_prompt(
        query=request.query,
        context_chunks=search_results,
        conversation_context=conversation_context,
        user_preferences=user_prefs
    )
    
    # Generate response using orchestrator
    llm_response = await orchestrator.generate_response(
        prompt=enhanced_prompt,
        context_chunks=search_results,
        prefer_fast=request.prefer_fast_response,
        provider=request.preferred_provider,
        model=request.preferred_model
    )
    
    # Add message to conversation memory
    await memory_manager.add_message_to_conversation(
        conversation_id, "user", request.query
    )
    
    await memory_manager.add_message_to_conversation(
        conversation_id, "assistant", llm_response.content,
        metadata={
            "provider": llm_response.provider,
            "model": llm_response.model,
            "tokens_used": llm_response.tokens_used,
            "search_results_count": len(search_results)
        }
    )
    
    # Prepare sources
    sources = []
    for result in search_results:
        sources.append({
            "text": result.get("text", "")[:200] + "...",
            "page_number": result.get("page_number"),
            "relevance_score": result.get("combined_score", result.get("rerank_score", 0)),
            "search_type": result.get("search_type", "hybrid")
        })
    
    processing_time = time.time() - start_time
    
    # Log query analytics
    background_tasks.add_task(
        analytics_logger.log_query,
        current_user.id, request.query, llm_response.content,
        llm_response.provider, llm_response.model,
        processing_time * 1000, llm_response.tokens_used,
        llm_response.cost_cents, len(search_results)
    )
    
    return QueryResponse(
        answer=llm_response.content,
        sources=sources,
        conversation_id=conversation_id,
        provider_used=llm_response.provider,
        model_used=llm_response.model,
        processing_time=processing_time,
        tokens_used=llm_response.tokens_used,
        cost_cents=llm_response.cost_cents,
        search_metadata={
            "total_chunks_found": len(search_results),
            "search_type": "semantic_hybrid",
            "reranked": request.use_reranking
        }
    )


def format_structured_response(
    result: StructuredQueryResult,
    classification: QueryClassification,
    original_query: str
) -> str:
    """Format structured query results into natural language"""
    
    if not result.data:
        return "I couldn't find any structured data matching your query in this document."
    
    # Format based on query intent
    if classification.intent == 'find_teacher':
        return format_teacher_response(result.data, original_query)
    elif classification.intent == 'find_subjects':
        return format_subjects_response(result.data, original_query)
    elif classification.intent == 'get_credits':
        return format_credits_response(result.data, original_query)
    elif classification.intent == 'list_by_semester':
        return format_semester_response(result.data, original_query)
    else:
        return format_general_structured_response(result.data, original_query)


def format_teacher_response(data: List[Dict[str, Any]], query: str) -> str:
    """Format teacher query results"""
    if not data:
        return "No teacher information found for your query."
    
    response = "Here's the teacher information I found:\n\n"
    
    for i, item in enumerate(data[:5], 1):  # Limit to top 5
        teacher = item.get('teacher', 'Unknown')
        subject = item.get('subject', 'Subject not specified')
        credits = item.get('credits', '')
        hours = item.get('hours', '')
        
        response += f"{i}. **{teacher}**\n"
        response += f"   - Subject: {subject}\n"
        if credits:
            response += f"   - Credits: {credits}\n"
        if hours:
            response += f"   - Hours: {hours}\n"
        response += f"   - Page: {item.get('page_number', 'N/A')}\n\n"
    
    if len(data) > 5:
        response += f"... and {len(data) - 5} more results found."
    
    return response


def format_subjects_response(data: List[Dict[str, Any]], query: str) -> str:
    """Format subject query results"""
    if not data:
        return "No subject information found for your query."
    
    response = "Here are the subjects I found:\n\n"
    
    for i, item in enumerate(data[:5], 1):
        subject = item.get('subject', 'Unknown Subject')
        teacher = item.get('teacher', '')
        credits = item.get('credits', '')
        hours = item.get('hours', '')
        
        response += f"{i}. **{subject}**\n"
        if teacher:
            response += f"   - Teacher: {teacher}\n"
        if credits:
            response += f"   - Credits: {credits}\n"
        if hours:
            response += f"   - Hours: {hours}\n"
        response += f"   - Page: {item.get('page_number', 'N/A')}\n\n"
    
    if len(data) > 5:
        response += f"... and {len(data) - 5} more results found."
    
    return response


def format_credits_response(data: List[Dict[str, Any]], query: str) -> str:
    """Format credits query results"""
    response = "Here's the credit information I found:\n\n"
    
    for i, item in enumerate(data[:5], 1):
        credits = item.get('credits', 'Unknown')
        subject = item.get('subject', 'Subject not specified')
        
        response += f"{i}. **{credits} credits** - {subject}\n"
        response += f"   - Page: {item.get('page_number', 'N/A')}\n\n"
    
    return response


def format_semester_response(data: List[Dict[str, Any]], query: str) -> str:
    """Format semester query results"""
    response = "Here are the subjects for the requested semester:\n\n"
    
    for i, item in enumerate(data[:10], 1):  # More results for semester queries
        subject = item.get('subject', 'Subject not specified')
        teacher = item.get('teacher', '')
        
        response += f"{i}. **{subject}**"
        if teacher:
            response += f" - {teacher}"
        response += f"\n   - Page: {item.get('page_number', 'N/A')}\n\n"
    
    return response


def format_general_structured_response(data: List[Dict[str, Any]], query: str) -> str:
    """Format general structured query results"""
    response = "Here's what I found in the structured data:\n\n"
    
    for i, item in enumerate(data[:5], 1):
        response += f"{i}. "
        
        # Try to extract meaningful information
        if 'subject' in item:
            response += f"Subject: {item['subject']} "
        if 'teacher' in item:
            response += f"Teacher: {item['teacher']} "
        if 'credits' in item:
            response += f"Credits: {item['credits']} "
        
        response += f"\n   - Page: {item.get('page_number', 'N/A')}\n"
        
        # Show raw data if available
        if 'raw_data' in item:
            response += f"   - Data: {str(item['raw_data'])}\n\n"
    
    return response


def format_structured_sources(result: StructuredQueryResult) -> List[Dict[str, Any]]:
    """Format structured query results as sources"""
    sources = []
    
    for item in result.data:
        sources.append({
            "text": str(item.get('raw_data', item))[:200] + "...",
            "page_number": item.get('page_number', 0),
            "relevance_score": 0.9,  # High relevance for structured matches
            "search_type": "structured"
        })
    
    return sources


def build_hybrid_context(
    structured_data: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    query: str,
    classification: QueryClassification
) -> str:
    """Build context that combines structured and semantic data"""
    
    context = f"""You are answering a query that requires both structured data analysis and general document understanding.

Query: {query}
Query Type: {classification.query_type.value}
Intent: {classification.intent}

STRUCTURED DATA FOUND:
"""
    
    if structured_data:
        for i, item in enumerate(structured_data, 1):
            context += f"\n{i}. "
            if 'subject' in item:
                context += f"Subject: {item['subject']} "
            if 'teacher' in item:
                context += f"Teacher: {item['teacher']} "
            if 'credits' in item:
                context += f"Credits: {item['credits']} "
            if 'hours' in item:
                context += f"Hours: {item['hours']} "
            context += f"(Page {item.get('page_number', 'N/A')})"
    else:
        context += "\nNo structured data found matching the query."
    
    context += "\n\nRELEVANT DOCUMENT CONTENT:\n"
    
    if semantic_results:
        for i, result in enumerate(semantic_results[:3], 1):
            context += f"\n[Excerpt {i} - Page {result.get('page_number', 'N/A')}]\n"
            context += result.get('text', '')[:300] + "...\n"
    else:
        context += "No relevant document content found."
    
    context += f"""

Instructions:
- Use the structured data to provide specific, factual answers
- Use the document content to provide context and explanation
- If the structured data directly answers the query, prioritize that information
- Cite page numbers when referencing information
- Be concise but comprehensive

Answer:"""
    
    return context

@app.get("/debug/llamacloud")
async def debug_llamacloud_status():
    """Debug LlamaCloud configuration and status"""
    from app.services.llamacloud_processor import (
        get_llamacloud_processor, 
        LLAMAPARSE_AVAILABLE,
        LLAMAPARSE_ERROR
    )
    
    processor = get_llamacloud_processor()
    
    status = {
        "package_info": {
            "llamaparse_available": LLAMAPARSE_AVAILABLE,
            "import_error": LLAMAPARSE_ERROR
        },
        "environment": {
            "api_key_in_env": bool(os.getenv('LLAMA_CLOUD_API_KEY')),
            "api_key_prefix": os.getenv('LLAMA_CLOUD_API_KEY', '')[:10] + "..." if os.getenv('LLAMA_CLOUD_API_KEY') else None,
            "api_key_in_config": hasattr(config, 'LLAMA_CLOUD_API_KEY')
        },
        "processor_status": processor.get_status() if processor else {
            "error": "Processor instance is None"
        },
        "recommendations": []
    }
    
    # Add recommendations based on status
    if not LLAMAPARSE_AVAILABLE:
        status["recommendations"].append("Install package: pip install llama-parse")
    
    if not status["environment"]["api_key_in_env"] and not status["environment"]["api_key_in_config"]:
        status["recommendations"].append("Set API key: export LLAMA_CLOUD_API_KEY='llx-your-key'")
    
    if processor and not processor.is_available():
        status["recommendations"].append(f"Fix initialization error: {processor._init_error}")
    
    return status

@app.get("/conversations/", response_model=List[ConversationSummary])
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get user's conversations"""
    try:
        conversations = db.query(Conversation).filter(
            Conversation.user_id == current_user.id
        ).order_by(Conversation.updated_at.desc()).offset(offset).limit(limit).all()
        
        return [
            ConversationSummary(
                id=conv.id,
                title=conv.title,
                document_id=conv.document_id,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=len(conv.messages)
            )
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversations")

@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200)
):
    """Get messages from a conversation"""
    try:
        # Verify ownership
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.desc()).limit(limit).all()
        
        return [
            MessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at,
                metadata=msg.document_metadata
            )
            for msg in reversed(messages)  # Return in chronological order
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch messages")

@app.post("/conversations/", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new conversation"""
    try:
        # Verify document ownership if document_id is provided
        if request.document_id:
            document = db.query(Document).filter(
                Document.id == request.document_id,
                Document.user_id == current_user.id
            ).first()
            
            if not document:
                raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        conversation_id = await memory_manager.create_conversation(
            current_user.id, request.document_id, request.title
        )
        
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            document_id=conversation.document_id,
            created_at=conversation.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")

@app.get("/documents/", response_model=List[DocumentSummary])
async def get_user_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get user's uploaded documents"""
    try:
        documents = db.query(Document).filter(
            Document.user_id == current_user.id
        ).order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
        
        return [
            DocumentSummary(
                id=doc.id,
                filename=doc.filename,
                file_size=doc.file_size,
                page_count=doc.page_count,
                created_at=doc.created_at,
                metadata=doc.document_metadata
            )
            for doc in documents
        ]
        
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")

@app.get("/analytics/dashboard", response_model=AnalyticsDashboard)
async def get_analytics_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    days: int = Query(7, ge=1, le=365)
):
    """Get analytics dashboard data for authenticated user"""
    try:
        return await analytics_logger.get_dashboard_data(current_user.id, days, db)
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics")

@app.get("/providers/status", response_model=ProvidersStatus)
async def get_providers_status():
    """Get status of all LLM providers"""
    try:
        stats = orchestrator.get_provider_stats()
        
        return ProvidersStatus(
            providers=stats,
            active_count=len([p for p in stats.values() if p.get('available', False)]),
            default_provider=config.get_provider_for_query_type('complex')
        )
        
    except Exception as e:
        logger.error(f"Error fetching provider status: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch provider status")

@app.post("/user/profile", response_model=UserProfileResponse)
async def update_user_profile(
    request: UpdateUserProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile and preferences"""
    try:
        # Update user profile in database
        current_user.profile.update(request.profile_updates)
        db.commit()
        
        # Update memory manager profile if available
        try:
            profile_manager = memory_manager.get_user_profile(current_user.id)
            profile_manager.update_profile(request.profile_updates)
        except Exception as e:
            logger.warning(f"Failed to update memory manager profile: {e}")
        
        return UserProfileResponse(
            user_id=current_user.id,
            profile={k: v for k, v in current_user.profile.items() if k != "password_hash"},
            updated=True
        )
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    client = Depends(get_weaviate_client)
):
    """Delete a document and its associated data"""
    try:
        # Verify ownership
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from Weaviate
        try:
            collection = client.collections.get(config.WEAVIATE_CLASS_NAME)
            from app.services import vector_store
            collection.data.delete_many(
                where=vector_store.wvc.query.Filter.by_property("pdf_id").equal(document_id)
            )
        except Exception as e:
            logger.warning(f"Failed to delete from Weaviate: {e}")
        
        # Delete associated conversations and messages
        conversations = db.query(Conversation).filter(Conversation.document_id == document_id).all()
        for conv in conversations:
            db.query(Message).filter(Message.conversation_id == conv.id).delete()
            db.delete(conv)
        
        # Delete document chunks if they exist in the database
        try:
            from app.database.models import DocumentChunk
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        except Exception as e:
            logger.warning(f"Failed to delete document chunks: {e}")
        
        # Delete from PostgreSQL
        db.delete(document)
        db.commit()
        
        logger.info(f"Document deleted: {document.filename} by user {current_user.username}")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation and its messages"""
    try:
        # Verify ownership
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete messages
        db.query(Message).filter(Message.conversation_id == conversation_id).delete()
        
        # Delete conversation
        db.delete(conversation)
        db.commit()
        
        logger.info(f"Conversation deleted: {conversation_id} by user {current_user.username}")
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")

# Public endpoints (for backward compatibility or demo purposes)
@app.get("/public/query", response_model=QueryResponse)
async def public_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
    client = Depends(get_weaviate_client)
):
    """Public query endpoint for testing or demo purposes"""
    
    user_id = get_user_id_from_user(current_user)
    
    # Rate limiting for public users
    if not current_user and not rate_limiter.check_query_limit(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for anonymous users")
    
    try:
        start_time = time.time()
        
        logger.info(f"Public Query: '{request.query}' | User: {user_id} | Doc: {request.pdf_id}")
        
        # Perform hybrid search (without user restrictions for public queries)
        search_results = await hybrid_search_engine.hybrid_search(
            client=client,
            query=request.query,
            document_id=request.pdf_id,
            top_k=request.top_k or 5,
            use_reranking=request.use_reranking
        )
        
        # Build basic prompt without user preferences
        enhanced_prompt = build_enhanced_prompt(
            query=request.query,
            context_chunks=search_results,
            conversation_context="",
            user_preferences={}
        )
        
        # Generate response using orchestrator
        llm_response = await orchestrator.generate_response(
            prompt=enhanced_prompt,
            context_chunks=search_results,
            prefer_fast=request.prefer_fast_response,
            provider=request.preferred_provider,
            model=request.preferred_model
        )
        
        # Prepare sources
        sources = []
        for result in search_results:
            sources.append({
                "text": result.get("text", "")[:200] + "...",
                "page_number": result.get("page_number"),
                "relevance_score": result.get("combined_score", result.get("rerank_score", 0)),
                "search_type": result.get("search_type", "hybrid")
            })
        
        processing_time = time.time() - start_time
        
        # Log query analytics for public queries too
        background_tasks.add_task(
            analytics_logger.log_query,
            user_id, request.query, llm_response.content,
            llm_response.provider, llm_response.model,
            processing_time * 1000, llm_response.tokens_used,
            llm_response.cost_cents, len(search_results)
        )
        
        return QueryResponse(
            answer=llm_response.content,
            sources=sources,
            conversation_id=None,  # No conversation for public queries
            provider_used=llm_response.provider,
            model_used=llm_response.model,
            processing_time=processing_time,
            tokens_used=llm_response.tokens_used,
            cost_cents=llm_response.cost_cents,
            search_metadata={
                "total_chunks_found": len(search_results),
                "search_type": "hybrid",
                "reranked": request.use_reranking
            }
        )
        
    except Exception as e:
        logger.error(f"Public query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# Protected route example
@app.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    """Example protected endpoint"""
    return {
        "message": f"Hello {current_user.username}!", 
        "user_id": current_user.id,
        "access_granted": True
    }

def build_enhanced_prompt(
    query: str,
    context_chunks: List[Dict[str, Any]],
    conversation_context: str,
    user_preferences: Dict[str, Any]
) -> str:
    """Build enhanced prompt with context and user preferences"""
    
    expertise_level = user_preferences.get('expertise_level', 'beginner')
    response_style = user_preferences.get('response_style', 'balanced')
    
    # Build context from chunks
    if context_chunks:
        context_text = "\n\n".join([
            f"[Page {chunk.get('page_number', 'N/A')}] {chunk.get('text', '')}"
            for chunk in context_chunks[:5]
        ])
    else:
        context_text = "No relevant context was found in the document."
    
    # Adaptation instructions based on user profile
    adaptation_instructions = ""
    if expertise_level == 'beginner':
        adaptation_instructions = "Provide clear explanations and avoid excessive jargon. Include examples when helpful."
    elif expertise_level == 'expert':
        adaptation_instructions = "You can use technical terminology and assume domain knowledge."
    
    if response_style == 'concise':
        adaptation_instructions += " Keep responses brief and direct."
    elif response_style == 'detailed':
        adaptation_instructions += " Provide comprehensive explanations with context."
    
    prompt = f"""You are an AI assistant specialized in answering questions based on document content.

{adaptation_instructions}

{conversation_context}

Context from document:
{context_text}

Current question: {query}

Instructions:
- Answer using ONLY the information provided in the context
- If the answer is not in the context, clearly state that you cannot find the information
- Be precise and cite specific page numbers when possible
- Consider the conversation history for better context understanding
- If the context is unclear or incomplete, mention this limitation

Answer:"""
    
    return prompt

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )