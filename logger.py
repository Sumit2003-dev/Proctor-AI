"""Thread-safe and buffered alert logger for ProctorAI.

Records timestamped integrity events into a CSV file and provides
access for live Streamlit presentation and CSV download.
"""

import csv
import datetime
import threading
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import DEFAULT_LOG_PATH, LOGS_DIR


class AlertLogger:
    """Manages CSV persistence and in-memory caching of proctoring alerts."""

    CSV_HEADERS = ["Timestamp", "Alert_Type", "Suspicion_Score", "Details"]

    def __init__(self, log_path: Path = DEFAULT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.lock = threading.Lock()
        self._init_csv()

    def _init_csv(self) -> None:
        """Initializes the CSV file with headers if it does not exist."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

    def log(
        self,
        alert_type: str,
        suspicion_score: float,
        details: str = "",
        timestamp: Optional[datetime.datetime] = None,
    ) -> None:
        """Appends a new alert entry to the CSV log.

        Args:
            alert_type: Name or category of the violation.
            suspicion_score: Current rolling suspicion score (0-100).
            details: Contextual details (e.g. 'Looked right for 2.4s', 'Count: 2').
            timestamp: Optional datetime; uses now() if not provided.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        with self.lock:
            with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp_str, alert_type, f"{suspicion_score:.1f}", details])

    def get_dataframe(self, limit: Optional[int] = 50) -> pd.DataFrame:
        """Loads alerts into a pandas DataFrame."""
        with self.lock:
            if not self.log_path.exists() or self.log_path.stat().st_size == 0:
                return pd.DataFrame(columns=self.CSV_HEADERS)
            try:
                df = pd.read_csv(self.log_path)
                if limit and len(df) > limit:
                    df = df.tail(limit)
                return df
            except Exception:
                return pd.DataFrame(columns=self.CSV_HEADERS)

    def get_csv_bytes(self) -> bytes:
        """Returns the raw CSV file bytes for Streamlit download."""
        with self.lock:
            if self.log_path.exists():
                return self.log_path.read_bytes()
            return ",".join(self.CSV_HEADERS).encode("utf-8")

    def clear(self) -> None:
        """Clears the CSV log and re-writes headers."""
        with self.lock:
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

