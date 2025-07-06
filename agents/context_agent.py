"""
Project AWARENESS - Context Agent
Manages serialized context objects with sliding window and auto-summarization.
"""

import asyncio
import time
import msgpack
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import hashlib
import re

from agents.base_agent import BaseAgent, AgentMessage
from core.config import AwarenessConfig


@dataclass
class ContextItem:
    """Represents a context item."""
    id: str
    content: str
    timestamp: float
    type: str  # user, assistant, system
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0
    

@dataclass
class ContextWindow:
    """Represents a context window."""
    items: List[ContextItem] = field(default_factory=list)
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)
    summary: Optional[str] = None
    

class ContextAgent(BaseAgent):
    """
    Context Agent managing serialized context objects.
    Implements sliding window and auto-summarization for context compression.
    """
    
    def __init__(self, name: str, config: AwarenessConfig, agent_config: Dict[str, Any]):
        super().__init__(name, config, agent_config)
        
        # Context configuration
        self.max_context_size = self.config.memory.context_window_size
        self.auto_summarize_threshold = self.config.memory.auto_summarize_threshold
        
        # Current context
        self.current_context: ContextWindow = ContextWindow()
        self.context_history: deque = deque(maxlen=100)  # Keep last 100 windows
        
        # Context management
        self.token_estimator_ratio = 4  # Rough chars per token
        self.importance_decay = 0.95
        
        # Conversation state
        self.conversation_id: Optional[str] = None
        self.participants: List[str] = []
        self.context_metadata: Dict[str, Any] = {}
        
        # Register handlers
        self._register_context_handlers()
        
    async def _initialize(self):
        """Initialize the context agent."""
        self.logger.info("ContextAgent initializing...")
        
        # Initialize context management
        await self._initialize_context_management()
        
        self.logger.info("ContextAgent initialized successfully")
        
    async def _shutdown(self):
        """Shutdown the context agent."""
        self.logger.info("ContextAgent shutting down...")
        
        # Save current context
        await self._save_context_state()
        
        self.logger.info("ContextAgent shutdown complete")
        
    def _register_context_handlers(self):
        """Register context-specific message handlers."""
        self.register_handler("context_add", self._handle_context_add)
        self.register_handler("context_get", self._handle_context_get)
        self.register_handler("context_summarize", self._handle_context_summarize)
        self.register_handler("context_clear", self._handle_context_clear)
        self.register_handler("parse_user_input", self._handle_parse_user_input)
        self.register_handler("context_search", self._handle_context_search)
        
    def _get_background_tasks(self) -> List:
        """Get context-specific background tasks."""
        return [
            self._context_maintenance_loop(),
            self._context_optimization_loop(),
        ]
        
    async def _handle_context_add(self, message: AgentMessage):
        """Handle context add requests."""
        try:
            content = message.payload.get("content")
            content_type = message.payload.get("type", "user")
            metadata = message.payload.get("metadata", {})
            importance = message.payload.get("importance", 1.0)
            
            if not content:
                await self._send_error_response(message, "Missing content")
                return
                
            # Add to context
            context_id = await self.add_context(content, content_type, metadata, importance)
            
            await self._send_response(message, {
                "status": "success",
                "context_id": context_id,
                "context_size": self.current_context.total_tokens
            })
            
        except Exception as e:
            self.logger.error(f"Error handling context add: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_context_get(self, message: AgentMessage):
        """Handle context get requests."""
        try:
            include_history = message.payload.get("include_history", False)
            format_type = message.payload.get("format", "raw")
            
            # Get context
            context_data = await self.get_context(include_history, format_type)
            
            await self._send_response(message, {
                "status": "success",
                "context": context_data,
                "context_size": self.current_context.total_tokens
            })
            
        except Exception as e:
            self.logger.error(f"Error handling context get: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_context_summarize(self, message: AgentMessage):
        """Handle context summarization requests."""
        try:
            force = message.payload.get("force", False)
            
            # Summarize context
            summary = await self.summarize_context(force)
            
            await self._send_response(message, {
                "status": "success",
                "summary": summary,
                "context_size": self.current_context.total_tokens
            })
            
        except Exception as e:
            self.logger.error(f"Error handling context summarize: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_context_clear(self, message: AgentMessage):
        """Handle context clear requests."""
        try:
            save_summary = message.payload.get("save_summary", True)
            
            # Clear context
            await self.clear_context(save_summary)
            
            await self._send_response(message, {
                "status": "success",
                "message": "Context cleared"
            })
            
        except Exception as e:
            self.logger.error(f"Error handling context clear: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_parse_user_input(self, message: AgentMessage):
        """Handle user input parsing requests."""
        try:
            user_input = message.payload.get("input")
            
            if not user_input:
                await self._send_error_response(message, "Missing input")
                return
                
            # Parse user input
            parsed_input = await self.parse_user_input(user_input)
            
            await self._send_response(message, {
                "status": "success",
                "parsed_input": parsed_input
            })
            
        except Exception as e:
            self.logger.error(f"Error handling parse user input: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_context_search(self, message: AgentMessage):
        """Handle context search requests."""
        try:
            query = message.payload.get("query")
            max_results = message.payload.get("max_results", 10)
            
            if not query:
                await self._send_error_response(message, "Missing query")
                return
                
            # Search context
            results = await self.search_context(query, max_results)
            
            await self._send_response(message, {
                "status": "success",
                "results": results,
                "count": len(results)
            })
            
        except Exception as e:
            self.logger.error(f"Error handling context search: {e}")
            await self._send_error_response(message, str(e))
            
    async def add_context(self, content: str, content_type: str = "user", 
                         metadata: Optional[Dict[str, Any]] = None, 
                         importance: float = 1.0) -> str:
        """
        Add content to the current context.
        
        Args:
            content: Content to add
            content_type: Type of content (user, assistant, system)
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            
        Returns:
            Context item ID
        """
        # Create context item
        context_id = hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16]
        
        context_item = ContextItem(
            id=context_id,
            content=content,
            timestamp=time.time(),
            type=content_type,
            metadata=metadata or {},
            importance=importance
        )
        
        # Add to current context
        self.current_context.items.append(context_item)
        self.current_context.total_tokens += self._estimate_tokens(content)
        self.current_context.end_time = time.time()
        
        # Check if we need to summarize
        if self.current_context.total_tokens > self.auto_summarize_threshold:
            await self.summarize_context()
            
        self.logger.debug(f"Added context item {context_id}: {content[:50]}...")
        
        return context_id
        
    async def get_context(self, include_history: bool = False, 
                         format_type: str = "raw") -> Dict[str, Any]:
        """
        Get the current context.
        
        Args:
            include_history: Whether to include context history
            format_type: Format type (raw, formatted, compressed)
            
        Returns:
            Context data
        """
        context_data = {
            "current_window": await self._format_context_window(self.current_context, format_type),
            "total_tokens": self.current_context.total_tokens,
            "conversation_id": self.conversation_id,
            "participants": self.participants,
            "metadata": self.context_metadata
        }
        
        if include_history:
            history = []
            for window in self.context_history:
                history.append(await self._format_context_window(window, format_type))
            context_data["history"] = history
            
        return context_data
        
    async def summarize_context(self, force: bool = False) -> Optional[str]:
        """
        Summarize the current context.
        
        Args:
            force: Force summarization even if threshold not met
            
        Returns:
            Summary text
        """
        if not force and self.current_context.total_tokens < self.auto_summarize_threshold:
            return None
            
        # Create summary
        summary = await self._create_context_summary(self.current_context)
        
        # Archive current context
        self.current_context.summary = summary
        self.context_history.append(self.current_context)
        
        # Start new context window
        self.current_context = ContextWindow()
        
        # Add summary as context
        if summary:
            await self.add_context(
                content=f"Previous context summary: {summary}",
                content_type="system",
                metadata={"type": "summary"},
                importance=0.8
            )
            
        self.logger.info(f"Context summarized: {len(summary) if summary else 0} characters")
        
        return summary
        
    async def clear_context(self, save_summary: bool = True):
        """
        Clear the current context.
        
        Args:
            save_summary: Whether to save a summary before clearing
        """
        if save_summary and self.current_context.items:
            await self.summarize_context(force=True)
        else:
            # Just clear without summary
            self.current_context = ContextWindow()
            
        self.logger.info("Context cleared")
        
    async def parse_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Parse user input to extract intent and entities.
        
        Args:
            user_input: User input text
            
        Returns:
            Parsed input data
        """
        parsed_data = {
            "original_text": user_input,
            "cleaned_text": self._clean_text(user_input),
            "intent": await self._extract_intent(user_input),
            "entities": await self._extract_entities(user_input),
            "sentiment": await self._analyze_sentiment(user_input),
            "keywords": await self._extract_keywords(user_input),
            "metadata": {
                "length": len(user_input),
                "word_count": len(user_input.split()),
                "timestamp": time.time()
            }
        }
        
        return parsed_data
        
    async def search_context(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search context for relevant items.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching context items
        """
        results = []
        query_lower = query.lower()
        
        # Search current context
        for item in self.current_context.items:
            if query_lower in item.content.lower():
                results.append({
                    "id": item.id,
                    "content": item.content,
                    "type": item.type,
                    "timestamp": item.timestamp,
                    "importance": item.importance,
                    "metadata": item.metadata,
                    "window": "current"
                })
                
        # Search context history
        for i, window in enumerate(self.context_history):
            for item in window.items:
                if query_lower in item.content.lower():
                    results.append({
                        "id": item.id,
                        "content": item.content,
                        "type": item.type,
                        "timestamp": item.timestamp,
                        "importance": item.importance,
                        "metadata": item.metadata,
                        "window": f"history_{i}"
                    })
                    
            # Also search summaries
            if window.summary and query_lower in window.summary.lower():
                results.append({
                    "id": f"summary_{i}",
                    "content": window.summary,
                    "type": "summary",
                    "timestamp": window.end_time,
                    "importance": 0.9,
                    "metadata": {"type": "summary"},
                    "window": f"history_{i}"
                })
                
        # Sort by relevance and importance
        results.sort(key=lambda x: x["importance"], reverse=True)
        
        return results[:max_results]
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.token_estimator_ratio
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)
        
        return text
        
    async def _extract_intent(self, text: str) -> str:
        """Extract intent from text (simple keyword-based)."""
        text_lower = text.lower()
        
        # Simple intent classification
        if any(word in text_lower for word in ['what', 'how', 'when', 'where', 'why', 'who']):
            return "question"
        elif any(word in text_lower for word in ['please', 'can you', 'could you', 'would you']):
            return "request"
        elif any(word in text_lower for word in ['remember', 'save', 'store']):
            return "memory"
        elif any(word in text_lower for word in ['hello', 'hi', 'hey']):
            return "greeting"
        elif any(word in text_lower for word in ['bye', 'goodbye', 'see you']):
            return "farewell"
        else:
            return "statement"
            
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text (simple pattern-based)."""
        entities = []
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            entities.append({"type": "email", "value": email})
            
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        for url in urls:
            entities.append({"type": "url", "value": url})
            
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, text)
        for number in numbers:
            entities.append({"type": "number", "value": float(number) if '.' in number else int(number)})
            
        return entities
        
    async def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text (simple keyword-based)."""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
            
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction (remove stop words and short words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates and return most common
        keyword_freq = {}
        for keyword in keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [keyword for keyword, freq in sorted_keywords[:10]]
        
    async def _create_context_summary(self, context_window: ContextWindow) -> str:
        """Create a summary of a context window."""
        if not context_window.items:
            return ""
            
        # Simple extractive summarization
        important_items = [item for item in context_window.items if item.importance > 0.7]
        
        if not important_items:
            # Fall back to most recent items
            important_items = context_window.items[-3:]
            
        # Create summary
        summary_parts = []
        for item in important_items:
            if item.type == "user":
                summary_parts.append(f"User: {item.content[:100]}...")
            elif item.type == "assistant":
                summary_parts.append(f"Assistant: {item.content[:100]}...")
            elif item.type == "system":
                summary_parts.append(f"System: {item.content[:100]}...")
                
        summary = " ".join(summary_parts)
        
        # Ensure summary is not too long
        if len(summary) > 500:
            summary = summary[:497] + "..."
            
        return summary
        
    async def _format_context_window(self, window: ContextWindow, format_type: str) -> Dict[str, Any]:
        """Format a context window for output."""
        if format_type == "raw":
            return {
                "items": [
                    {
                        "id": item.id,
                        "content": item.content,
                        "type": item.type,
                        "timestamp": item.timestamp,
                        "importance": item.importance,
                        "metadata": item.metadata
                    }
                    for item in window.items
                ],
                "total_tokens": window.total_tokens,
                "start_time": window.start_time,
                "end_time": window.end_time,
                "summary": window.summary
            }
        elif format_type == "formatted":
            formatted_items = []
            for item in window.items:
                formatted_items.append(f"[{item.type}] {item.content}")
            return {
                "formatted_content": "\n".join(formatted_items),
                "total_tokens": window.total_tokens,
                "summary": window.summary
            }
        elif format_type == "compressed":
            return {
                "summary": window.summary or await self._create_context_summary(window),
                "total_tokens": window.total_tokens,
                "item_count": len(window.items)
            }
        else:
            return await self._format_context_window(window, "raw")
            
    async def _context_maintenance_loop(self):
        """Background context maintenance loop."""
        while self.is_running:
            try:
                # Decay importance of old items
                await self._decay_importance()
                
                # Clean up old context
                await self._cleanup_old_context()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in context maintenance loop: {e}")
                await asyncio.sleep(600)
                
    async def _context_optimization_loop(self):
        """Background context optimization loop."""
        while self.is_running:
            try:
                # Optimize context usage
                await self._optimize_context()
                
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in context optimization loop: {e}")
                await asyncio.sleep(900)
                
    async def _decay_importance(self):
        """Decay importance of context items over time."""
        current_time = time.time()
        
        for item in self.current_context.items:
            age_hours = (current_time - item.timestamp) / 3600
            if age_hours > 1:  # Start decaying after 1 hour
                decay_factor = self.importance_decay ** age_hours
                item.importance *= decay_factor
                
    async def _cleanup_old_context(self):
        """Clean up old context items."""
        # Remove items with very low importance
        self.current_context.items = [
            item for item in self.current_context.items 
            if item.importance > 0.1
        ]
        
        # Recalculate total tokens
        self.current_context.total_tokens = sum(
            self._estimate_tokens(item.content) 
            for item in self.current_context.items
        )
        
    async def _optimize_context(self):
        """Optimize context usage."""
        # Auto-summarize if getting close to threshold
        if self.current_context.total_tokens > self.auto_summarize_threshold * 0.8:
            await self.summarize_context()
            
    async def _initialize_context_management(self):
        """Initialize context management."""
        # Any initialization specific to context management
        pass
        
    async def _save_context_state(self):
        """Save current context state."""
        # Implementation for saving context state
        pass
        
    async def _send_response(self, original_message: AgentMessage, response: Dict[str, Any]):
        """Send response to a message."""
        # Implementation depends on message bus integration
        pass
        
    async def _send_error_response(self, original_message: AgentMessage, error: str):
        """Send error response to a message."""
        await self._send_response(original_message, {
            "status": "error",
            "error": error
        })