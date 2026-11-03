"""Suspicion scoring engine with rolling window evaluation.

Maintains alert history over a configured time window (default 30 seconds)
and computes a normalized weighted suspicion score (0-100).
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

from config import SCORING_CONFIG, ScoringConfig


@dataclass
class AlertRecord:
    """Represents a single timestamped alert occurrence."""
    timestamp: float
    alert_type: str
    weight: float


class SuspicionScorer:
    """Computes a rolling-window weighted suspicion score for exam proctoring."""

    def __init__(self, config: ScoringConfig = SCORING_CONFIG) -> None:
        self.config = config
        self.history: Deque[AlertRecord] = deque()
        self._last_alert_logged_time: Dict[str, float] = {}
        # Min cooldown in seconds between recording the same event to avoid
        # 30fps burst inflation within the same second
        self.cooldown_seconds: float = 1.0

    def prune_old_records(self, current_time: float) -> None:
        """Removes records older than the rolling window length."""
        cutoff = current_time - self.config.WINDOW_SECONDS
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

    def add_alerts(self, alerts: List[str], current_time: float = None) -> float:
        """Registers active alerts and updates the suspicion score.

        Args:
            alerts: List of active alert message strings.
            current_time: Optional explicit timestamp (defaults to time.time()).

        Returns:
            Current calculated suspicion score between 0.0 and 100.0.
        """
        if current_time is None:
            current_time = time.time()

        for alert in alerts:
            # Map alert text to configured weight key
            matched_key = None
            for key in self.config.WEIGHTS:
                if key.lower() in alert.lower() or alert.lower() in key.lower():
                    matched_key = key
                    break

            if matched_key:
                last_time = self._last_alert_logged_time.get(matched_key, 0.0)
                if current_time - last_time >= self.cooldown_seconds:
                    self._last_alert_logged_time[matched_key] = current_time
                    weight = self.config.WEIGHTS[matched_key]
                    self.history.append(
                        AlertRecord(
                            timestamp=current_time,
                            alert_type=matched_key,
                            weight=weight,
                        )
                    )

        self.prune_old_records(current_time)
        return self.compute_score(current_time)

    def compute_score(self, current_time: float = None) -> float:
        """Calculates current suspicion score from active window records."""
        if current_time is None:
            current_time = time.time()
        self.prune_old_records(current_time)

        raw_score = sum(record.weight for record in self.history)
        return min(self.config.MAX_SCORE, float(raw_score))

    def get_status(self, score: float = None) -> Tuple[str, str]:
        """Returns risk category name and hex color based on score."""
        if score is None:
            score = self.compute_score()

        if score < self.config.NORMAL_THRESHOLD:
            return "NORMAL", "#28a745"   # Green
        elif score < self.config.WARNING_THRESHOLD:
            return "WARNING", "#fd7e14"  # Orange
        else:
            return "HIGH RISK", "#dc3545" # Red

    def reset(self) -> None:
        """Clears all historical alerts."""
        self.history.clear()
        self._last_alert_logged_time.clear()

