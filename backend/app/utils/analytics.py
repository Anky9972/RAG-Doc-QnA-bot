# utils/analytics.py
import logging
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.database.models import QueryLog, Document, Message, Conversation
from app.schemas import AnalyticsDashboard, AnalyticsData

logger = logging.getLogger(__name__)

class AnalyticsLogger:
    def __init__(self):
        self.query_cache = {}
        self.upload_cache = {}
    
    async def log_query(
        self,
        user_id: str,
        query: str,
        response: str,
        provider: str,
        model: str,
        latency_ms: int,
        token_count: int,
        cost_cents: float,
        chunks_retrieved: int
    ):
        """Log query analytics"""
        try:
            from app.database.connection import get_db_session
            
            db = get_db_session()
            
            query_log = QueryLog(
                user_id=user_id,
                query=query,
                response=response,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                token_count=token_count,
                cost_cents=cost_cents,
                chunks_retrieved=chunks_retrieved
            )
            
            db.add(query_log)
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Error logging query analytics: {e}")
    
    async def log_document_upload(
        self,
        user_id: str,
        document_id: str,
        filename: str,
        processing_time: float
    ):
        """Log document upload analytics"""
        try:
            # Store in cache for real-time metrics
            self.upload_cache[user_id] = {
                'document_id': document_id,
                'filename': filename,
                'processing_time': processing_time,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error logging upload analytics: {e}")
    
    async def get_dashboard_data(
        self,
        user_id: str,
        days: int,
        db: Session
    ) -> AnalyticsDashboard:
        """Get analytics dashboard data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Query analytics
            query_stats = db.query(
                func.count(QueryLog.id).label('total_queries'),
                func.avg(QueryLog.latency_ms).label('avg_latency'),
                func.sum(QueryLog.cost_cents).label('total_cost')
            ).filter(
                QueryLog.user_id == user_id,
                QueryLog.created_at >= cutoff_date
            ).first()
            
            # Provider usage
            provider_usage = db.query(
                QueryLog.provider,
                func.count(QueryLog.id).label('usage_count')
            ).filter(
                QueryLog.user_id == user_id,
                QueryLog.created_at >= cutoff_date
            ).group_by(QueryLog.provider).all()
            
            # Document count
            doc_count = db.query(func.count(Document.id)).filter(
                Document.user_id == user_id,
                Document.created_at >= cutoff_date
            ).scalar()
            
            # Popular queries
            popular_queries = db.query(
                QueryLog.query
            ).filter(
                QueryLog.user_id == user_id,
                QueryLog.created_at >= cutoff_date
            ).order_by(desc(QueryLog.created_at)).limit(10).all()
            
            analytics_data = AnalyticsData(
                total_queries=query_stats.total_queries or 0,
                total_documents=doc_count or 0,
                avg_response_time=float(query_stats.avg_latency or 0) / 1000,  # Convert to seconds
                total_cost=float(query_stats.total_cost or 0) / 100,  # Convert to dollars
                provider_usage={p.provider: p.usage_count for p in provider_usage},
                popular_queries=[q.query for q in popular_queries]
            )
            
            return AnalyticsDashboard(
                user_analytics=analytics_data,
                time_period_days=days,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            raise

# Initialize global analytics logger
analytics_logger = AnalyticsLogger()