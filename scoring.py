"""Calibrated risk scoring engine with rolling-window exponential decay.

Maintains alert history over a configured sliding window (default 30 seconds),
prunes expired records and cooldown state, and calculates:
1. Time-decayed continuous suspicion index (0 - 100).
2. Calibrated Cheating Probability P(Cheating in [0.0, 1.0]).
3. Risk categorization tiers: NORMAL, WARNING, HIGH RISK.
"""

import math
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
    """Computes a calibrated time-decayed suspicion score and cheating probability."""

    def __init__(self, config: ScoringConfig = SCORING_CONFIG) -> None:
        self.config = config
        self.history: Deque[AlertRecord] = deque()
        self._last_alert_logged_time: Dict[str, float] = {}
        # Precompute decay lambda from half-life: lambda = ln(2) / half_life
        self._decay_lambda = math.log(2.0) / max(1.0, self.config.DECAY_HALF_LIFE_SECONDS)

    def prune_old_records(self, current_time: float) -> None:
        """Removes expired alerts and cleans up stale cooldown timestamps."""
        cutoff = current_time - self.config.WINDOW_SECONDS

        # 1. Prune history records outside sliding window
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

        # 2. Prune stale cooldown timestamps to prevent indefinite memory/state leaks
        expired_keys = [
            k for k, t in self._last_alert_logged_time.items()
            if (current_time - t) > self.config.WINDOW_SECONDS
        ]
        for k in expired_keys:
            del self._last_alert_logged_time[k]

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
                # Apply per-event cooldown
                if current_time - last_time >= self.config.EVENT_COOLDOWN_SECONDS:
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
        """Calculates time-decayed suspicion score from active window records."""
        if current_time is None:
            current_time = time.time()
        self.prune_old_records(current_time)

        # Exponential decay: recent violations in last few seconds carry maximum weight;
        # older violations smoothly fade rather than abruptly falling off a cliff.
        raw_score = 0.0
        for record in self.history:
            delta_t = max(0.0, current_time - record.timestamp)
            decay_factor = math.exp(-self._decay_lambda * delta_t)
            raw_score += record.weight * decay_factor

        return min(self.config.MAX_SCORE, float(raw_score))

    def get_cheating_probability(self, score: float = None) -> float:
        """Computes calibrated probability of cheating P(Cheating) in [0.0, 1.0].

        Uses a calibrated logistic sigmoid function parameterized by
        center threshold theta and scale beta.
        """
        if score is None:
            score = self.compute_score()

        if score <= 0.0:
            return 0.0

        # Logistic sigmoid mapping: 1 / (1 + exp(-(score - theta) / beta))
        exponent = -(score - self.config.SIGMOID_CENTER_THETA) / max(1.0, self.config.SIGMOID_SCALE_BETA)
        # Numerical stability clamp
        exponent = max(-20.0, min(20.0, exponent))
        prob = 1.0 / (1.0 + math.exp(exponent))
        return float(prob)

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
        """Clears all historical alerts and cooldown trackers."""
        self.history.clear()
        self._last_alert_logged_time.clear()
