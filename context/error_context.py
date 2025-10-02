import re
from typing import Optional, Tuple
from models.schemas import ErrorContext, ErrorType
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ErrorParser:
    """
    Parses error messages and stack traces from various sources.
    Extracts: error type, message, file path, line number, stack trace.
    """

    # Regex patterns for common error formats
    PATTERNS = {
        # Python: "  File "main.py", line 42, in function_name"
        'python_traceback': re.compile(
            r'File "([^"]+)", line (\d+)(?:, in (\w+))?'
        ),

        # JavaScript/TypeScript: "    at function (file.js:42:10)"
        'js_stack': re.compile(
            r'at .+ \((.+):(\d+):(\d+)\)'
        ),

        # Generic: "Error in file.py:42"
        'generic_location': re.compile(
            r'(?:in |at )?([a-zA-Z0-9_./\\-]+\.(?:py|js|ts|jsx|tsx|java|cpp|c)):(\d+)'
        ),

        # Python errors: "NameError: name 'x' is not defined"
        'python_error': re.compile(
            r'((?:Syntax|Name|Type|Value|Import|Attribute|Key|Index|Runtime|Module)Error): (.+)'
        ),

        # JavaScript errors: "TypeError: Cannot read property"
        'js_error': re.compile(
            r'((?:Type|Reference|Syntax|Range)Error): (.+)'
        ),

        # Java: "Exception in thread ... ClassName: message"
        'java_error': re.compile(
            r'(\w+Exception): (.+)'
        ),
    }

    @classmethod
    def parse_terminal_error(cls, terminal_output: str) -> Optional[ErrorContext]:
        """
        Parse error from terminal output.

        Args:
            terminal_output: Raw terminal text (last 50-100 lines)

        Returns:
            ErrorContext or None if no error found
        """
        lines = terminal_output.strip().split('\n')

        # Look for error type first
        error_type = ErrorType.UNKNOWN
        error_message = ""
        stack_trace = ""
        file_path = None
        line_number = None

        # Search for error patterns
        for i, line in enumerate(lines):
            # Python error
            match = cls.PATTERNS['python_error'].search(line)
            if match:
                error_type = cls._classify_error_type(match.group(1))
                error_message = f"{match.group(1)}: {match.group(2)}"

                # Get stack trace (previous lines)
                if i > 0:
                    stack_trace = '\n'.join(lines[max(0, i-10):i+1])

                # Extract file and line from traceback
                file_path, line_number = cls._extract_python_location(lines[:i+1])
                break

            # JavaScript error
            match = cls.PATTERNS['js_error'].search(line)
            if match:
                error_type = cls._classify_error_type(match.group(1))
                error_message = f"{match.group(1)}: {match.group(2)}"

                # Get stack trace
                if i < len(lines) - 1:
                    stack_trace = '\n'.join(lines[i:min(i+10, len(lines))])

                # Extract file and line
                file_path, line_number = cls._extract_js_location(lines[i:])
                break

            # Java exception
            match = cls.PATTERNS['java_error'].search(line)
            if match:
                error_type = cls._classify_error_type(match.group(1))
                error_message = f"{match.group(1)}: {match.group(2)}"
                stack_trace = '\n'.join(lines[i:min(i+10, len(lines))])
                break

        if not error_message:
            # No structured error found, check for generic error keywords
            for line in reversed(lines):  # Start from bottom
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed']):
                    error_message = line.strip()
                    error_type = ErrorType.RUNTIME
                    break

        if error_message:
            return ErrorContext(
                error_message=error_message,
                error_type=error_type,
                stack_trace=stack_trace or None,
                file_path=file_path,
                line_number=line_number,
                timestamp=datetime.now()
            )

        return None

    @classmethod
    def parse_diagnostic(cls, diagnostic: dict) -> ErrorContext:
        """
        Parse VS Code diagnostic (from language server).

        Args:
            diagnostic: VS Code diagnostic object
                {
                    "message": "...",
                    "severity": 1,  # 1=Error, 2=Warning
                    "source": "typescript",
                    "range": {"start": {"line": 41, ...}, ...}
                }

        Returns:
            ErrorContext
        """
        message = diagnostic.get('message', 'Unknown error')
        source = diagnostic.get('source', 'unknown')

        # Classify by source
        if 'typescript' in source or 'javascript' in source:
            error_type = ErrorType.TYPE
        elif 'python' in source or 'pylance' in source:
            error_type = ErrorType.TYPE
        else:
            error_type = ErrorType.SYNTAX

        # Extract location
        range_info = diagnostic.get('range', {})
        start = range_info.get('start', {})
        line_number = start.get('line', 0) + 1  # VS Code is 0-indexed

        return ErrorContext(
            error_message=message,
            error_type=error_type,
            stack_trace=None,
            file_path=None,  # Will be set by caller
            line_number=line_number,
            timestamp=datetime.now()
        )

    @classmethod
    def _extract_python_location(cls, lines: list) -> Tuple[Optional[str], Optional[int]]:
        """Extract file path and line number from Python traceback."""
        for line in reversed(lines):
            match = cls.PATTERNS['python_traceback'].search(line)
            if match:
                return match.group(1), int(match.group(2))

        return None, None

    @classmethod
    def _extract_js_location(cls, lines: list) -> Tuple[Optional[str], Optional[int]]:
        """Extract file path and line number from JavaScript stack trace."""
        for line in lines:
            match = cls.PATTERNS['js_stack'].search(line)
            if match:
                return match.group(1), int(match.group(2))

            # Try generic pattern
            match = cls.PATTERNS['generic_location'].search(line)
            if match:
                return match.group(1), int(match.group(2))

        return None, None

    @classmethod
    def _classify_error_type(cls, error_name: str) -> ErrorType:
        """Classify error type from error name."""
        error_lower = error_name.lower()

        if 'syntax' in error_lower:
            return ErrorType.SYNTAX
        elif 'type' in error_lower:
            return ErrorType.TYPE
        elif 'import' in error_lower or 'module' in error_lower:
            return ErrorType.IMPORT
        elif any(x in error_lower for x in ['name', 'reference', 'attribute', 'key']):
            return ErrorType.RUNTIME
        else:
            return ErrorType.RUNTIME

    @classmethod
    def extract_error_signature(cls, error: ErrorContext) -> str:
        """
        Create a fingerprint/signature for error matching.

        Returns:
            String signature like "TypeError_cannot_read_property"
        """
        # Remove specific values, keep structure
        signature_parts = [error.error_type.value]

        # Simplify error message
        msg = error.error_message.lower()

        # Remove specific variable names, numbers, paths
        msg = re.sub(r"'[^']+'", "'<var>'", msg)
        msg = re.sub(r'"[^"]+"', '"<var>"', msg)
        msg = re.sub(r'\b\d+\b', '<num>', msg)
        msg = re.sub(r'/[^\s]+', '<path>', msg)

        # Take first 50 chars
        msg_clean = re.sub(r'[^a-z0-9_]', '_', msg[:50])

        signature_parts.append(msg_clean)

        return '_'.join(signature_parts)
