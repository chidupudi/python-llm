import sqlite3
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import logging
import json

from models.schemas import PatternMatch, ErrorContext
from context.error_context import ErrorParser

logger = logging.getLogger(__name__)


class PatternStore:
    """
    SQLite-based storage for error patterns and successful solutions.
    Learns from each debugging session to improve suggestions.
    """

    def __init__(self, db_path: str = "debug_patterns.db"):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """Initialize database with required tables."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

            # Create tables
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS error_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_signature TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    file_pattern TEXT,
                    times_successful INTEGER DEFAULT 1,
                    times_failed INTEGER DEFAULT 0,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    project_root TEXT,
                    metadata TEXT
                )
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_signature
                ON error_patterns(error_signature)
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_type
                ON error_patterns(error_type)
            """)

            self.conn.commit()
            logger.info(f"Pattern store initialized: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def add_successful_pattern(self,
                              error: ErrorContext,
                              solution: str,
                              project_root: str = "") -> None:
        """
        Store a successful error->solution pattern.

        Args:
            error: The error that was resolved
            solution: Description of what fixed it
            project_root: Project path (for project-specific patterns)
        """
        try:
            signature = ErrorParser.extract_error_signature(error)

            # Check if pattern exists
            existing = self.conn.execute(
                "SELECT id, times_successful FROM error_patterns WHERE error_signature = ?",
                (signature,)
            ).fetchone()

            if existing:
                # Update existing pattern
                self.conn.execute("""
                    UPDATE error_patterns
                    SET times_successful = times_successful + 1,
                        last_used = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (existing['id'],))

                logger.info(f"Updated pattern: {signature} (success count: {existing['times_successful'] + 1})")
            else:
                # Insert new pattern
                file_pattern = Path(error.file_path).suffix if error.file_path else None

                self.conn.execute("""
                    INSERT INTO error_patterns
                    (error_signature, error_type, error_message, solution, file_pattern, project_root)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    signature,
                    error.error_type.value,
                    error.error_message[:500],  # Truncate long messages
                    solution,
                    file_pattern,
                    project_root
                ))

                logger.info(f"New pattern stored: {signature}")

            self.conn.commit()

        except Exception as e:
            logger.error(f"Failed to store pattern: {e}")

    def record_failed_attempt(self, error: ErrorContext, attempted_solution: str) -> None:
        """
        Record a solution that didn't work (to avoid suggesting it again).

        Args:
            error: The error
            attempted_solution: What was tried that didn't work
        """
        try:
            signature = ErrorParser.extract_error_signature(error)

            self.conn.execute("""
                UPDATE error_patterns
                SET times_failed = times_failed + 1
                WHERE error_signature = ? AND solution = ?
            """, (signature, attempted_solution))

            self.conn.commit()

        except Exception as e:
            logger.error(f"Failed to record failure: {e}")

    def find_similar_patterns(self,
                             error: ErrorContext,
                             limit: int = 5) -> List[PatternMatch]:
        """
        Find similar error patterns from database.

        Args:
            error: Current error to match
            limit: Maximum number of results

        Returns:
            List of similar patterns with solutions
        """
        try:
            signature = ErrorParser.extract_error_signature(error)

            # First, try exact signature match
            exact_matches = self.conn.execute("""
                SELECT * FROM error_patterns
                WHERE error_signature = ?
                ORDER BY times_successful DESC, last_used DESC
                LIMIT ?
            """, (signature, limit)).fetchall()

            matches = []

            for row in exact_matches:
                matches.append(PatternMatch(
                    error_signature=row['error_signature'],
                    solution=row['solution'],
                    similarity_score=1.0,  # Exact match
                    times_successful=row['times_successful'],
                    last_used=datetime.fromisoformat(row['last_used'])
                ))

            # If no exact matches, try same error type
            if not matches:
                type_matches = self.conn.execute("""
                    SELECT * FROM error_patterns
                    WHERE error_type = ?
                    ORDER BY times_successful DESC, last_used DESC
                    LIMIT ?
                """, (error.error_type.value, limit)).fetchall()

                for row in type_matches:
                    matches.append(PatternMatch(
                        error_signature=row['error_signature'],
                        solution=row['solution'],
                        similarity_score=0.5,  # Partial match (same type)
                        times_successful=row['times_successful'],
                        last_used=datetime.fromisoformat(row['last_used'])
                    ))

            return matches[:limit]

        except Exception as e:
            logger.error(f"Failed to find patterns: {e}")
            return []

    def get_project_patterns(self, project_root: str, limit: int = 10) -> List[PatternMatch]:
        """
        Get patterns specific to a project.

        Args:
            project_root: Project path
            limit: Max results

        Returns:
            List of project-specific patterns
        """
        try:
            rows = self.conn.execute("""
                SELECT * FROM error_patterns
                WHERE project_root = ?
                ORDER BY times_successful DESC, last_used DESC
                LIMIT ?
            """, (project_root, limit)).fetchall()

            matches = []
            for row in rows:
                matches.append(PatternMatch(
                    error_signature=row['error_signature'],
                    solution=row['solution'],
                    similarity_score=0.8,  # Project-specific boost
                    times_successful=row['times_successful'],
                    last_used=datetime.fromisoformat(row['last_used'])
                ))

            return matches

        except Exception as e:
            logger.error(f"Failed to get project patterns: {e}")
            return []

    def get_statistics(self) -> dict:
        """Get database statistics."""
        try:
            total = self.conn.execute("SELECT COUNT(*) as count FROM error_patterns").fetchone()

            top_errors = self.conn.execute("""
                SELECT error_type, COUNT(*) as count
                FROM error_patterns
                GROUP BY error_type
                ORDER BY count DESC
            """).fetchall()

            return {
                "total_patterns": total['count'],
                "error_distribution": {row['error_type']: row['count'] for row in top_errors}
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Pattern store closed")
