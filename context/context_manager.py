from typing import List, Optional, Dict
from datetime import datetime
import logging
from collections import defaultdict

from models.schemas import (
    ErrorContext, CodeChange, SessionAttempt,
    ChatMessage, DebugContext, PatternMatch
)
from context.priority_scorer import PriorityScorer

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Master orchestrator for debugging context.
    Manages session state, prioritizes information, and maintains history.
    """

    def __init__(self):
        # Active sessions: {session_id: DebugContext}
        self.sessions: Dict[str, DebugContext] = {}

        # Pattern matching (will integrate with knowledge base later)
        self.known_patterns: List[PatternMatch] = []

        logger.info("ContextManager initialized")

    def get_or_create_session(self, session_id: str, project_root: str) -> DebugContext:
        """
        Get existing session or create new one.

        Args:
            session_id: Unique session identifier
            project_root: Project root path

        Returns:
            DebugContext for the session
        """
        if session_id not in self.sessions:
            logger.info(f"Creating new session: {session_id}")
            self.sessions[session_id] = DebugContext(
                session_id=session_id,
                project_root=project_root,
                current_error=None,
                recent_changes=[],
                session_attempts=[],
                chat_history=[]
            )

        return self.sessions[session_id]

    def update_error(self, session_id: str, error: ErrorContext) -> None:
        """
        Update current error in session.

        Args:
            session_id: Session identifier
            error: New error context
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found, creating new one")
            self.sessions[session_id] = DebugContext(
                session_id=session_id,
                project_root="",
                current_error=error,
                recent_changes=[],
                session_attempts=[],
                chat_history=[]
            )
        else:
            session = self.sessions[session_id]
            session.current_error = error

            # Calculate priority
            error.priority = PriorityScorer.score_error(error.timestamp)

            logger.info(
                f"Error updated for {session_id}: "
                f"{error.error_type.value} - Priority: {error.priority:.1f}"
            )

    def add_code_change(self, session_id: str, change: CodeChange) -> None:
        """
        Add code change to session context.

        Args:
            session_id: Session identifier
            change: Code change information
        """
        session = self.get_or_create_session(session_id, "")

        # Calculate priority
        change.priority = PriorityScorer.score_code_change(change.timestamp)

        # Add to recent changes
        session.recent_changes.append(change)

        # Keep only top 10 by priority
        session.recent_changes = PriorityScorer.rank_items(
            session.recent_changes,
            lambda c: c.priority
        )[:10]

        logger.info(f"Code change added: {change.file_path} - Priority: {change.priority:.1f}")

    def add_attempt(self, session_id: str, attempt: SessionAttempt) -> None:
        """
        Record a debugging attempt.

        Args:
            session_id: Session identifier
            attempt: Attempt details
        """
        session = self.get_or_create_session(session_id, "")
        session.session_attempts.append(attempt)

        logger.info(
            f"Attempt recorded: {attempt.attempted_solution} -> {attempt.outcome}"
        )

        # If error was resolved, clear current error
        if attempt.error_resolved and session.current_error:
            logger.info("Error resolved! Clearing current error.")
            session.current_error = None

    def add_user_message(self, message: str, context: DebugContext) -> None:
        """
        Add user message to chat history.

        Args:
            message: User's message
            context: Current debug context
        """
        session = self.get_or_create_session(context.session_id, context.project_root)

        chat_msg = ChatMessage(
            role="user",
            content=message,
            timestamp=datetime.now()
        )

        session.chat_history.append(chat_msg)

        # Keep last 20 messages
        if len(session.chat_history) > 20:
            session.chat_history = session.chat_history[-20:]

    def add_assistant_message(self, session_id: str, message: str) -> None:
        """
        Add assistant response to chat history.

        Args:
            session_id: Session identifier
            message: Assistant's response
        """
        session = self.get_or_create_session(session_id, "")

        chat_msg = ChatMessage(
            role="assistant",
            content=message,
            timestamp=datetime.now()
        )

        session.chat_history.append(chat_msg)

    def get_prioritized_context(self, session_id: str) -> DebugContext:
        """
        Get context with items sorted by priority.

        Args:
            session_id: Session identifier

        Returns:
            DebugContext with prioritized items
        """
        session = self.get_or_create_session(session_id, "")

        # Recalculate priorities based on current time
        if session.current_error:
            session.current_error.priority = PriorityScorer.score_error(
                session.current_error.timestamp
            )

        for change in session.recent_changes:
            change.priority = PriorityScorer.score_code_change(change.timestamp)

        # Filter out low-priority items
        session.recent_changes = [
            c for c in session.recent_changes
            if PriorityScorer.should_include(c.priority, threshold=10.0)
        ]

        # Sort by priority
        session.recent_changes.sort(key=lambda x: x.priority, reverse=True)

        return session

    def find_similar_patterns(self, error: ErrorContext) -> List[str]:
        """
        Find similar error patterns from knowledge base.

        Args:
            error: Current error

        Returns:
            List of suggested solutions
        """
        # TODO: Integrate with vector database for semantic similarity
        # For now, simple keyword matching

        suggestions = []

        error_msg_lower = error.error_message.lower()

        # Common patterns
        if "cannot find module" in error_msg_lower or "modulenotfounderror" in error_msg_lower:
            suggestions.append("Install missing dependency: `npm install <module>` or `pip install <module>`")
            suggestions.append("Check import path spelling")

        if "typeerror" in error_msg_lower:
            suggestions.append("Check variable types and function signatures")
            suggestions.append("Verify object properties exist before accessing")

        if "syntaxerror" in error_msg_lower:
            suggestions.append("Check for missing brackets, quotes, or semicolons")
            suggestions.append("Verify indentation (for Python)")

        if "null" in error_msg_lower or "undefined" in error_msg_lower:
            suggestions.append("Add null/undefined checks before accessing properties")
            suggestions.append("Initialize variables before use")

        return suggestions

    def get_session(self, session_id: str) -> Optional[DebugContext]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear session data."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session {session_id} cleared")

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.sessions.keys())
