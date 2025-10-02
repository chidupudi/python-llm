import math
from datetime import datetime, timedelta
from typing import Any


class PriorityScorer:
    """
    Calculates priority scores for context items using decay algorithms.
    Recent items have higher priority, but important patterns persist longer.
    """

    # Base priorities by context type
    BASE_PRIORITIES = {
        "current_error": 100.0,
        "recent_change": 80.0,
        "session_attempt": 60.0,
        "cached_pattern": 40.0,
        "project_context": 20.0
    }

    # Decay constants (in minutes) - how quickly priority decays
    DECAY_CONSTANTS = {
        "current_error": 5.0,      # Errors decay fast (5 min half-life)
        "recent_change": 10.0,     # Changes decay slower
        "session_attempt": 15.0,   # Attempts decay even slower
        "cached_pattern": 1440.0,  # Patterns persist (1 day)
        "project_context": float('inf')  # Project context never decays
    }

    @classmethod
    def score_error(cls, timestamp: datetime) -> float:
        """
        Score an error based on recency.

        Recent errors (< 30 sec) = 100
        Errors from 5 min ago = ~50
        Errors from 15 min ago = ~12.5
        """
        return cls._calculate_score(
            timestamp=timestamp,
            base_priority=cls.BASE_PRIORITIES["current_error"],
            decay_constant=cls.DECAY_CONSTANTS["current_error"]
        )

    @classmethod
    def score_code_change(cls, timestamp: datetime) -> float:
        """
        Score code changes based on recency.

        Changes in last minute = 80
        Changes from 10 min ago = ~40
        Changes from 30 min ago = ~10
        """
        return cls._calculate_score(
            timestamp=timestamp,
            base_priority=cls.BASE_PRIORITIES["recent_change"],
            decay_constant=cls.DECAY_CONSTANTS["recent_change"]
        )

    @classmethod
    def score_attempt(cls, timestamp: datetime, was_successful: bool = False) -> float:
        """
        Score debugging attempts.

        Successful attempts get boosted priority (remember what worked!)
        Failed attempts still matter (avoid repeating)
        """
        base = cls.BASE_PRIORITIES["session_attempt"]

        # Boost successful attempts
        if was_successful:
            base *= 1.5

        return cls._calculate_score(
            timestamp=timestamp,
            base_priority=base,
            decay_constant=cls.DECAY_CONSTANTS["session_attempt"]
        )

    @classmethod
    def score_pattern(cls,
                     similarity: float,
                     times_successful: int,
                     last_used: datetime) -> float:
        """
        Score cached patterns from knowledge base.

        Factors:
        - Similarity to current error (0-1)
        - How many times it worked before
        - Recency of last use (but decays very slowly)
        """
        # Base score from pattern database
        base = cls.BASE_PRIORITIES["cached_pattern"]

        # Similarity boost (0-1 becomes 0-2x multiplier)
        similarity_boost = 1.0 + similarity

        # Success boost (log scale to avoid runaway)
        success_boost = 1.0 + math.log(times_successful + 1) * 0.5

        # Recency (decays very slowly - 1 day half-life)
        recency_score = cls._calculate_score(
            timestamp=last_used,
            base_priority=1.0,
            decay_constant=cls.DECAY_CONSTANTS["cached_pattern"]
        )

        return base * similarity_boost * success_boost * recency_score

    @classmethod
    def _calculate_score(cls,
                        timestamp: datetime,
                        base_priority: float,
                        decay_constant: float) -> float:
        """
        Calculate exponential decay score.

        Formula: priority = base * e^(-time_elapsed / decay_constant)

        Args:
            timestamp: When the item was created
            base_priority: Starting priority (0-100)
            decay_constant: How quickly it decays (in minutes)

        Returns:
            Decayed priority score
        """
        if decay_constant == float('inf'):
            return base_priority  # Never decays

        # Time elapsed in minutes
        elapsed = (datetime.now() - timestamp).total_seconds() / 60.0

        # Exponential decay
        decay_factor = math.exp(-elapsed / decay_constant)

        return base_priority * decay_factor

    @classmethod
    def should_include(cls, priority: float, threshold: float = 10.0) -> bool:
        """
        Determine if an item should be included in context.

        Args:
            priority: Item's priority score
            threshold: Minimum priority to include (default: 10)

        Returns:
            True if priority is above threshold
        """
        return priority >= threshold

    @classmethod
    def rank_items(cls, items: list, get_priority_fn) -> list:
        """
        Rank items by priority score.

        Args:
            items: List of items to rank
            get_priority_fn: Function that extracts priority from item

        Returns:
            Items sorted by priority (highest first)
        """
        return sorted(items, key=get_priority_fn, reverse=True)
