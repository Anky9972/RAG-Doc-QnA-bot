# app/utils/analytics.py
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
import uuid

from app.database.models import QueryLog, Document, User
from app.schemas import AnalyticsData, AnalyticsDashboard

logger = logging.getLogger(__name__)

class AnalyticsLogger:
    """Analytics logging and reporting system"""
    
    async def log_query(
        self,
        user_id: str,
        query: str,
        response: str,
        provider: str,
        model: str,
        latency_ms: float,
        tokens_used: int,
        cost_cents: float,
        chunks_retrieved: int,
        conversation_id: Optional[str] = None,
        retrieval_score: Optional[float] = None
    ):
        """
        Log a query to the database
        
        Args:
            user_id: User ID who made the query
            query: The query text
            response: The response text
            provider: LLM provider used (e.g., 'openai', 'anthropic')
            model: Model used (e.g., 'gpt-4', 'claude-3')
            latency_ms: Response time in milliseconds
            tokens_used: Number of tokens consumed
            cost_cents: Cost in cents
            chunks_retrieved: Number of document chunks retrieved
            conversation_id: Optional conversation ID
            retrieval_score: Optional retrieval quality score
        """
        try:
            from app.database.connection import get_db_session
            db = get_db_session()
            
            query_log = QueryLog(
                id=str(uuid.uuid4()),
                user_id=user_id,
                conversation_id=conversation_id,
                query=query[:1000],  # Limit query length
                response=response[:2000],  # Limit response length
                provider=provider,
                model=model,
                latency_ms=int(latency_ms),  # Convert to int
                token_count=tokens_used,  # ✅ Correct field name
                cost_cents=float(cost_cents),
                chunks_retrieved=chunks_retrieved,
                retrieval_score=retrieval_score,
                created_at=datetime.now(timezone.utc)
            )
            
            db.add(query_log)
            db.commit()
            
            logger.info(f"✅ Analytics logged: user={user_id}, provider={provider}, latency={latency_ms:.0f}ms")
            
        except Exception as e:
            logger.error(f"❌ Failed to log analytics: {e}", exc_info=True)
        finally:
            if 'db' in locals():
                db.close()
    
    async def log_document_upload(
        self,
        user_id: str,
        document_id: str,
        filename: str,
        processing_time: float
    ):
        """Log document upload event"""
        try:
            logger.info(
                f"📄 Document upload: user={user_id}, doc={document_id}, "
                f"file={filename}, time={processing_time:.2f}s"
            )
        except Exception as e:
            logger.error(f"Failed to log document upload: {e}")
    
    async def get_dashboard_data(
        self,
        user_id: str,
        days: int,
        db: Session
    ) -> AnalyticsDashboard:
        """
        Get analytics dashboard data for a user
        
        Args:
            user_id: User ID to get analytics for
            days: Number of days to look back
            db: Database session
            
        Returns:
            AnalyticsDashboard with user analytics
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get all query logs for this user within the time period
            query_logs = db.query(QueryLog).filter(
                QueryLog.user_id == user_id,
                QueryLog.created_at >= cutoff_date
            ).all()
            
            # Calculate metrics
            total_queries = len(query_logs)
            
            # Average response time (in seconds)
            avg_response_time = 0.0
            if total_queries > 0:
                total_latency_ms = sum(log.latency_ms for log in query_logs)
                avg_response_time = (total_latency_ms / total_queries) / 1000  # Convert to seconds
            
            # Total cost
            total_cost = sum(log.cost_cents for log in query_logs) / 100  # Convert cents to dollars
            
            # Provider usage
            provider_usage = {}
            for log in query_logs:
                provider = log.provider
                provider_usage[provider] = provider_usage.get(provider, 0) + 1
            
            # Popular queries (deduplicated and sorted by frequency)
            query_counts = {}
            for log in query_logs:
                query_text = log.query.strip()
                if query_text:  # Skip empty queries
                    query_counts[query_text] = query_counts.get(query_text, 0) + 1
            
            # Sort by count and get top 10
            popular_queries_sorted = sorted(
                query_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            popular_queries = [q[0] for q in popular_queries_sorted]
            
            # Get total documents for this user
            total_documents = db.query(func.count(Document.id)).filter(
                Document.user_id == user_id
            ).scalar() or 0
            
            logger.info(
                f"📊 Analytics generated: user={user_id}, "
                f"queries={total_queries}, docs={total_documents}, "
                f"avg_time={avg_response_time:.2f}s, cost=${total_cost:.4f}"
            )
            
            return AnalyticsDashboard(
                user_analytics=AnalyticsData(
                    total_queries=total_queries,
                    total_documents=total_documents,
                    avg_response_time=avg_response_time,
                    total_cost=total_cost,
                    provider_usage=provider_usage,
                    popular_queries=popular_queries
                ),
                time_period_days=days,
                last_updated=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting dashboard data: {e}", exc_info=True)
            # Return empty analytics instead of failing
            return AnalyticsDashboard(
                user_analytics=AnalyticsData(
                    total_queries=0,
                    total_documents=0,
                    avg_response_time=0.0,
                    total_cost=0.0,
                    provider_usage={},
                    popular_queries=[]
                ),
                time_period_days=days,
                last_updated=datetime.now(timezone.utc)
            )
    
    def get_user_query_history(
        self,
        user_id: str,
        limit: int = 50,
        db: Session = None
    ) -> list:
        """Get recent query history for a user"""
        try:
            if db is None:
                from app.database.connection import get_db_session
                db = get_db_session()
                close_db = True
            else:
                close_db = False
            
            queries = db.query(QueryLog).filter(
                QueryLog.user_id == user_id
            ).order_by(QueryLog.created_at.desc()).limit(limit).all()
            
            result = [
                {
                    "id": q.id,
                    "query": q.query,
                    "response": q.response[:200] + "..." if len(q.response) > 200 else q.response,
                    "provider": q.provider,
                    "model": q.model,
                    "latency_ms": q.latency_ms,
                    "tokens": q.token_count,
                    "cost_cents": q.cost_cents,
                    "created_at": q.created_at.isoformat()
                }
                for q in queries
            ]
            
            if close_db:
                db.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting query history: {e}")
            return []

# Global instance
analytics_logger = AnalyticsLogger()