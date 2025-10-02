from models.schemas import DebugContext, ErrorContext, CodeChange
from typing import List
import logging

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds contextualized prompts for the LLM with priority-based information.
    """

    SYSTEM_PROMPT = """You are an expert debugging assistant integrated into VS Code.
You help developers fix errors by analyzing:
- Current errors and stack traces
- Recent code changes (git diffs)
- Previous debugging attempts in this session
- Similar past errors and their solutions

Provide concise, actionable solutions. Reference specific line numbers and files.
If a solution was already tried, suggest alternative approaches.
Focus on the most recent context first."""

    def __init__(self, max_tokens: int = 4000):
        """
        Args:
            max_tokens: Maximum tokens for context (leave room for response)
        """
        self.max_tokens = max_tokens

    def build_prompt(self, user_message: str, context: DebugContext) -> str:
        """
        Build complete prompt with prioritized context.

        Priority order:
        1. Current error (if exists)
        2. Recent code changes (last 5 minutes)
        3. Session attempts (what was tried)
        4. User message

        Args:
            user_message: User's question/request
            context: Full debugging context

        Returns:
            Complete prompt string
        """
        sections = []

        # 1. Current Error (Highest Priority)
        if context.current_error:
            sections.append(self._format_error(context.current_error))

        # 2. Recent Code Changes
        if context.recent_changes:
            sections.append(self._format_code_changes(context.recent_changes[:3]))

        # 3. Previous Attempts
        if context.session_attempts:
            sections.append(self._format_attempts(context.session_attempts[-5:]))

        # 4. Chat History (for context continuity)
        if context.chat_history:
            sections.append(self._format_chat_history(context.chat_history[-4:]))

        # 5. User Message
        sections.append(f"## Current Question\n{user_message}")

        # Combine all sections
        full_context = "\n\n".join(sections)

        # Build final prompt
        prompt = f"""{self.SYSTEM_PROMPT}

{full_context}

Provide a clear, actionable solution:"""

        # Log token estimate
        estimated_tokens = len(prompt.split()) * 1.3  # rough estimate
        logger.info(f"Prompt built: ~{int(estimated_tokens)} tokens")

        return prompt

    def build_chat_messages(self, user_message: str, context: DebugContext) -> List[dict]:
        """
        Build chat-style messages for chat API.

        Returns:
            List of message dicts: [{"role": "system", "content": "..."}, ...]
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        # Add context as system message
        context_parts = []

        if context.current_error:
            context_parts.append(self._format_error(context.current_error))

        if context.recent_changes:
            context_parts.append(self._format_code_changes(context.recent_changes[:3]))

        if context.session_attempts:
            context_parts.append(self._format_attempts(context.session_attempts[-5:]))

        if context_parts:
            messages.append({
                "role": "system",
                "content": "\n\n".join(context_parts)
            })

        # Add recent chat history
        for msg in context.chat_history[-4:]:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        return messages

    def _format_error(self, error: ErrorContext) -> str:
        """Format current error for prompt."""
        parts = [
            "## Current Error",
            f"**Type:** {error.error_type.value}",
            f"**Message:** {error.error_message}"
        ]

        if error.file_path:
            location = f"{error.file_path}"
            if error.line_number:
                location += f":{error.line_number}"
            parts.append(f"**Location:** {location}")

        if error.stack_trace:
            parts.append(f"**Stack Trace:**\n```\n{error.stack_trace[:500]}\n```")

        return "\n".join(parts)

    def _format_code_changes(self, changes: List[CodeChange]) -> str:
        """Format recent code changes."""
        if not changes:
            return ""

        parts = ["## Recent Code Changes"]

        for i, change in enumerate(changes, 1):
            parts.append(f"\n### Change {i}: {change.file_path}")
            if change.commit_hash:
                parts.append(f"Commit: {change.commit_hash[:8]}")

            # Truncate long diffs
            diff = change.diff[:800] if len(change.diff) > 800 else change.diff
            parts.append(f"```diff\n{diff}\n```")

        return "\n".join(parts)

    def _format_attempts(self, attempts: List) -> str:
        """Format previous debugging attempts."""
        if not attempts:
            return ""

        parts = ["## Previous Attempts in This Session"]

        for i, attempt in enumerate(attempts, 1):
            outcome_emoji = "✓" if attempt.error_resolved else "✗"
            parts.append(
                f"{i}. {outcome_emoji} {attempt.attempted_solution} → {attempt.outcome}"
            )

        return "\n".join(parts)

    def _format_chat_history(self, history: List) -> str:
        """Format recent chat for context."""
        if not history:
            return ""

        parts = ["## Recent Conversation"]

        for msg in history:
            role = msg.role.upper()
            parts.append(f"**{role}:** {msg.content[:200]}")

        return "\n".join(parts)
