# init_db.sql
-- Initialize database with extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for better performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_query_logs_user_id ON query_logs(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at);

-- Full-text search index for queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_query_logs_query_text ON query_logs USING gin(to_tsvector('english', query));
