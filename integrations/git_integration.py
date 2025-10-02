from git import Repo, InvalidGitRepositoryError
from typing import Optional, List
from datetime import datetime, timedelta
import logging
from pathlib import Path

from models.schemas import CodeChange

logger = logging.getLogger(__name__)


class GitIntegration:
    """
    Extracts code changes from git repository.
    Provides diffs, commit history, and recent changes.
    """

    def __init__(self, project_root: str):
        """
        Args:
            project_root: Path to project root (should contain .git)
        """
        self.project_root = Path(project_root)
        self.repo: Optional[Repo] = None

        try:
            self.repo = Repo(project_root)
            logger.info(f"Git repository found: {project_root}")
        except InvalidGitRepositoryError:
            logger.warning(f"Not a git repository: {project_root}")
            self.repo = None

    def is_available(self) -> bool:
        """Check if git is available for this project."""
        return self.repo is not None

    def get_recent_changes(self, minutes: int = 10) -> List[CodeChange]:
        """
        Get code changes from the last N minutes.

        Args:
            minutes: Look back this many minutes

        Returns:
            List of CodeChange objects with diffs
        """
        if not self.repo:
            return []

        changes = []

        try:
            # Get uncommitted changes (staged + unstaged)
            uncommitted = self.get_uncommitted_changes()
            if uncommitted:
                changes.extend(uncommitted)

            # Get recent commits
            since = datetime.now() - timedelta(minutes=minutes)

            for commit in self.repo.iter_commits(max_count=5):
                commit_time = datetime.fromtimestamp(commit.committed_date)

                if commit_time < since:
                    break

                # Get diff for this commit
                if commit.parents:
                    parent = commit.parents[0]
                    diff = parent.diff(commit, create_patch=True)

                    for diff_item in diff:
                        if diff_item.diff:
                            change = CodeChange(
                                file_path=diff_item.b_path or diff_item.a_path or "unknown",
                                diff=diff_item.diff.decode('utf-8', errors='ignore'),
                                timestamp=commit_time,
                                commit_hash=commit.hexsha
                            )
                            changes.append(change)

        except Exception as e:
            logger.error(f"Error getting recent changes: {e}")

        return changes

    def get_uncommitted_changes(self) -> List[CodeChange]:
        """
        Get current uncommitted changes (staged + unstaged).

        Returns:
            List of CodeChange objects
        """
        if not self.repo:
            return []

        changes = []

        try:
            # Unstaged changes
            unstaged = self.repo.index.diff(None, create_patch=True)
            for diff_item in unstaged:
                if diff_item.diff:
                    change = CodeChange(
                        file_path=diff_item.b_path or diff_item.a_path or "unknown",
                        diff=diff_item.diff.decode('utf-8', errors='ignore'),
                        timestamp=datetime.now(),
                        commit_hash=None
                    )
                    changes.append(change)

            # Staged changes
            staged = self.repo.index.diff('HEAD', create_patch=True)
            for diff_item in staged:
                if diff_item.diff:
                    change = CodeChange(
                        file_path=diff_item.b_path or diff_item.a_path or "unknown",
                        diff=diff_item.diff.decode('utf-8', errors='ignore'),
                        timestamp=datetime.now(),
                        commit_hash=None
                    )
                    changes.append(change)

        except Exception as e:
            logger.error(f"Error getting uncommitted changes: {e}")

        return changes

    def get_file_history(self, file_path: str, max_commits: int = 5) -> List[str]:
        """
        Get commit history for a specific file.

        Args:
            file_path: Relative path to file from project root
            max_commits: Maximum number of commits to retrieve

        Returns:
            List of commit messages
        """
        if not self.repo:
            return []

        try:
            commits = list(self.repo.iter_commits(paths=file_path, max_count=max_commits))
            return [f"{c.hexsha[:8]}: {c.message.strip()}" for c in commits]
        except Exception as e:
            logger.error(f"Error getting file history: {e}")
            return []

    def get_current_branch(self) -> Optional[str]:
        """Get current git branch name."""
        if not self.repo:
            return None

        try:
            return self.repo.active_branch.name
        except Exception as e:
            logger.error(f"Error getting branch: {e}")
            return None

    def get_diff_for_file(self, file_path: str) -> Optional[str]:
        """
        Get current diff for a specific file.

        Args:
            file_path: Path to file (relative to project root)

        Returns:
            Diff string or None
        """
        if not self.repo:
            return None

        try:
            # Check unstaged changes first
            diff = self.repo.git.diff(file_path)

            if not diff:
                # Check staged changes
                diff = self.repo.git.diff('--cached', file_path)

            return diff if diff else None

        except Exception as e:
            logger.error(f"Error getting diff for {file_path}: {e}")
            return None

    def blame_line(self, file_path: str, line_number: int) -> Optional[str]:
        """
        Git blame for a specific line (who last changed it).

        Args:
            file_path: File path relative to project root
            line_number: Line number (1-indexed)

        Returns:
            Blame info string or None
        """
        if not self.repo:
            return None

        try:
            blame = self.repo.git.blame(file_path, L=f"{line_number},{line_number}")
            return blame
        except Exception as e:
            logger.error(f"Error blaming line: {e}")
            return None
