# utils/rate_limiter.py
import time
from typing import Dict, DefaultDict
from collections import defaultdict
from app.core import config

class RateLimiter:
    def __init__(self):
        self.query_timestamps: DefaultDict[str, list] = defaultdict(list)
        self.upload_timestamps: DefaultDict[str, list] = defaultdict(list)
    
    def _cleanup_old_timestamps(self, timestamps: list, window_seconds: int):
        """Remove timestamps outside the time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Remove old timestamps
        while timestamps and timestamps[0] < cutoff_time:
            timestamps.pop(0)
    
    def check_query_limit(self, user_id: str) -> bool:
        """Check if user can make a query (rate limiting)"""
        current_time = time.time()
        user_queries = self.query_timestamps[user_id]
        
        # Clean old timestamps (1 minute window)
        self._cleanup_old_timestamps(user_queries, 60)
        
        # Check rate limit
        if len(user_queries) >= config.RATE_LIMITS['queries_per_minute']:
            return False
        
        # Add current timestamp
        user_queries.append(current_time)
        return True
    
    def check_upload_limit(self, user_id: str) -> bool:
        """Check if user can upload a file (rate limiting)"""
        current_time = time.time()
        user_uploads = self.upload_timestamps[user_id]
        
        # Clean old timestamps (1 hour window)
        self._cleanup_old_timestamps(user_uploads, 3600)
        
        # Check rate limit
        if len(user_uploads) >= config.RATE_LIMITS['uploads_per_hour']:
            return False
        
        # Add current timestamp
        user_uploads.append(current_time)
        return True

# Initialize global rate limiter
rate_limiter = RateLimiter()