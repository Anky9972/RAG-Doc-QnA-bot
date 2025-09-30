# services/memory_manager.py
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from app.database.models import Conversation, Message, User
from app.database.connection import get_db_session
from app.core import config
from app.services.llm_orchestrator import orchestrator
import json

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self, conversation_id: str, db_session: Session):
        self.conversation_id = conversation_id
        self.db_session = db_session
        self.short_term_memory = []
        self.long_term_summary = ""
        self._load_conversation()
    
    def _load_conversation(self):
        """Load existing conversation from database"""
        try:
            conversation = self.db_session.query(Conversation).filter(
                Conversation.id == self.conversation_id
            ).first()
            
            if conversation:
                self.long_term_summary = conversation.summary or ""
                
                # Load recent messages for short-term memory
                recent_messages = self.db_session.query(Message).filter(
                    Message.conversation_id == self.conversation_id
                ).order_by(desc(Message.created_at)).limit(
                    config.MEMORY_CONFIG['short_term_turns']
                ).all()
                
                self.short_term_memory = []
                for msg in reversed(recent_messages):  # Reverse to get chronological order
                    self.short_term_memory.append({
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.created_at.isoformat(),
                        'metadata': msg.metadata or {}
                    })
                    
        except Exception as e:
            logger.error(f"Error loading conversation {self.conversation_id}: {e}")
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to short-term memory"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        self.short_term_memory.append(message)
        
        # Keep only recent messages in short-term memory
        max_turns = config.MEMORY_CONFIG['short_term_turns']
        if len(self.short_term_memory) > max_turns:
            self.short_term_memory = self.short_term_memory[-max_turns:]
    
    async def should_update_summary(self) -> bool:
        """Check if long-term summary should be updated"""
        total_messages = self.db_session.query(Message).filter(
            Message.conversation_id == self.conversation_id
        ).count()
        
        threshold = config.MEMORY_CONFIG['long_term_summary_threshold']
        return total_messages >= threshold and total_messages % 10 == 0
    
    async def update_long_term_summary(self):
        """Update long-term conversation summary"""
        try:
            # Get all messages for summarization
            messages = self.db_session.query(Message).filter(
                Message.conversation_id == self.conversation_id
            ).order_by(Message.created_at).all()
            
            if len(messages) < 5:  # Not enough content to summarize
                return
            
            # Create conversation context
            conversation_text = ""
            for msg in messages[:-10]:  # Exclude recent messages (they're in short-term)
                conversation_text += f"{msg.role}: {msg.content}\n"
            
            # Generate summary using LLM
            summary_prompt = f"""
            Please provide a concise summary of this conversation, focusing on:
            1. Main topics discussed
            2. Key questions asked by the user
            3. Important information provided
            4. Any ongoing context that would be useful for future responses
            
            Conversation:
            {conversation_text}
            
            Summary:"""
            
            response = await orchestrator.generate_response(
                summary_prompt,
                prefer_fast=True
            )
            
            self.long_term_summary = response.content
            
            # Update database
            conversation = self.db_session.query(Conversation).filter(
                Conversation.id == self.conversation_id
            ).first()
            
            if conversation:
                conversation.summary = self.long_term_summary
                conversation.updated_at = datetime.utcnow()
                self.db_session.commit()
                
        except Exception as e:
            logger.error(f"Error updating long-term summary: {e}")
    
    def get_context_for_query(self, include_summary: bool = True) -> str:
        """Get conversation context for current query"""
        context_parts = []
        
        # Add long-term summary if available
        if include_summary and self.long_term_summary:
            context_parts.append(f"Previous conversation summary:\n{self.long_term_summary}\n")
        
        # Add recent conversation history
        if self.short_term_memory:
            context_parts.append("Recent conversation:")
            for msg in self.short_term_memory[-6:]:  # Last 3 turns (user + assistant)
                context_parts.append(f"{msg['role']}: {msg['content']}")
            context_parts.append("")
        
        return "\n".join(context_parts)

class UserProfileManager:
    def __init__(self, user_id: str, db_session: Session):
        self.user_id = user_id
        self.db_session = db_session
        self.profile = {}
        self._load_profile()
    
    def _load_profile(self):
        """Load user profile from database"""
        try:
            user = self.db_session.query(User).filter(User.id == self.user_id).first()
            if user and user.profile:
                self.profile = user.profile
        except Exception as e:
            logger.error(f"Error loading user profile: {e}")
    
    def update_profile(self, updates: Dict[str, Any]):
        """Update user profile"""
        try:
            self.profile.update(updates)
            
            user = self.db_session.query(User).filter(User.id == self.user_id).first()
            if user:
                user.profile = self.profile
                user.updated_at = datetime.utcnow()
                self.db_session.commit()
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
    
    def get_expertise_level(self) -> str:
        """Get user's expertise level"""
        return self.profile.get('expertise_level', 'beginner')
    
    def get_preferred_response_style(self) -> str:
        """Get user's preferred response style"""
        return self.profile.get('response_style', 'balanced')
    
    def adapt_response_for_user(self, base_response: str) -> str:
        """Adapt response based on user profile"""
        expertise = self.get_expertise_level()
        style = self.get_preferred_response_style()
        
        adaptations = []
        
        if expertise == 'beginner':
            adaptations.append("Please provide simplified explanations with examples.")
        elif expertise == 'expert':
            adaptations.append("You can use technical terminology and assume domain knowledge.")
        
        if style == 'concise':
            adaptations.append("Keep responses brief and to the point.")
        elif style == 'detailed':
            adaptations.append("Provide comprehensive explanations with background context.")
        
        if adaptations:
            adaptation_prompt = f"""
            Adapt the following response for a user with these preferences:
            {' '.join(adaptations)}
            
            Original response: {base_response}
            
            Adapted response:"""
            
            # This would typically use the LLM to adapt, but for now return original
            return base_response
        
        return base_response

class MemoryManager:
    def __init__(self):
        self.active_conversations = {}
        self.user_profiles = {}
    
    def get_conversation_memory(self, conversation_id: str) -> ConversationMemory:
        """Get or create conversation memory"""
        if conversation_id not in self.active_conversations:
            db_session = get_db_session()
            self.active_conversations[conversation_id] = ConversationMemory(
                conversation_id, db_session
            )
        return self.active_conversations[conversation_id]
    
    def get_user_profile(self, user_id: str) -> UserProfileManager:
        """Get or create user profile manager"""
        if user_id not in self.user_profiles:
            db_session = get_db_session()
            self.user_profiles[user_id] = UserProfileManager(user_id, db_session)
        return self.user_profiles[user_id]
    
    async def create_conversation(
        self, 
        user_id: str, 
        document_id: str = None, 
        title: str = None
    ) -> str:
        """Create a new conversation"""
        try:
            db_session = get_db_session()
            
            conversation = Conversation(
                user_id=user_id,
                document_id=document_id,
                title=title or "New Conversation",
                metadata={}
            )
            
            db_session.add(conversation)
            db_session.commit()
            
            conversation_id = conversation.id
            db_session.close()
            
            # Initialize memory
            self.get_conversation_memory(conversation_id)
            
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    async def add_message_to_conversation(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ):
        """Add message to conversation and memory"""
        try:
            db_session = get_db_session()
            
            # Save to database
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata=metadata or {}
            )
            
            db_session.add(message)
            db_session.commit()
            db_session.close()
            
            # Add to memory
            memory = self.get_conversation_memory(conversation_id)
            memory.add_message(role, content, metadata)
            
            # Check if summary needs updating
            if await memory.should_update_summary():
                await memory.update_long_term_summary()
                
        except Exception as e:
            logger.error(f"Error adding message to conversation: {e}")
    
    async def get_conversation_context(
        self, 
        conversation_id: str,
        user_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Get full conversation context and user preferences"""
        memory = self.get_conversation_memory(conversation_id)
        profile = self.get_user_profile(user_id)
        
        context = memory.get_context_for_query()
        
        user_prefs = {
            'expertise_level': profile.get_expertise_level(),
            'response_style': profile.get_preferred_response_style()
        }
        
        return context, user_prefs
    
    async def cleanup_old_conversations(self, days: int = 30):
        """Clean up old conversation data"""
        try:
            db_session = get_db_session()
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get old conversations
            old_conversations = db_session.query(Conversation).filter(
                Conversation.updated_at < cutoff_date
            ).all()
            
            for conv in old_conversations:
                # Remove from active memory
                if conv.id in self.active_conversations:
                    del self.active_conversations[conv.id]
            
            db_session.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up old conversations: {e}")

# Initialize global memory manager
memory_manager = MemoryManager()