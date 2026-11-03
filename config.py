"""Configuration module for ProctorAI.

Contains all configurable thresholds, weights, durations, and file paths.
Avoids hardcoded constants throughout the application.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


# Base paths
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
SNAPSHOTS_DIR = LOGS_DIR / "snapshots"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_PATH = LOGS_DIR / "alerts.csv"

# Model paths
DEFAULT_MODEL_NAME = "yolov8n.pt"
DATA_CONFIG_PATH = BASE_DIR / "data" / "data.yaml"


@dataclass(frozen=True)
class AudioConfig:
    """Microphone and acoustic monitoring configuration."""

    DEFAULT_RMS_THRESHOLD: float = 0.04
    SUSTAINED_AUDIO_SECONDS: float = 1.0
    SAMPLE_RATE: int = 16000


@dataclass(frozen=True)
class DetectionConfig:
    """Detection thresholds and class IDs."""

    # COCO Class IDs
    PERSON_CLASS_ID: int = 0
    CELL_PHONE_CLASS_ID: int = 67

    # Confidence thresholds
    DEFAULT_CONF_THRESHOLD: float = 0.5
    MIN_CONF_THRESHOLD: float = 0.2
    MAX_CONF_THRESHOLD: float = 0.95

    # Head Pose / MediaPipe Face Mesh parameters
    # Key landmark indices (Face Mesh canonical):
    # 1: Nose tip
    # 33: Left eye outer corner
    # 263: Right eye outer corner
    NOSE_TIP_IDX: int = 1
    LEFT_EYE_OUTER_IDX: int = 33
    RIGHT_EYE_OUTER_IDX: int = 263

    # Yaw ratio threshold for determining lookaway
    DEFAULT_YAW_THRESHOLD: float = 0.25

    # Duration threshold: lookaway must be sustained for >= this many seconds
    SUSTAINED_LOOKAWAY_SECONDS: float = 2.0

    # Snapshot evidence cooldown in seconds
    SNAPSHOT_COOLDOWN_SECONDS: float = 5.0


@dataclass(frozen=True)
class ScoringConfig:
    """Weights and scoring rules for suspicion calculation."""

    # Rolling window length in seconds
    WINDOW_SECONDS: float = 30.0

    # Severity weights per alert occurrence
    WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "Cell phone detected": 35.0,              # High severity
            "Multiple people detected": 30.0,          # High severity
            "No person in frame": 20.0,                # Medium severity
            "Looking away sustained": 15.0,            # Medium severity
            "Suspicious audio/speech detected": 20.0,  # Medium severity
        }
    )

    # Maximum score cap
    MAX_SCORE: float = 100.0

    # Risk category thresholds
    NORMAL_THRESHOLD: float = 30.0
    WARNING_THRESHOLD: float = 70.0


# Default global instances for convenience
AUDIO_CONFIG = AudioConfig()
DETECTION_CONFIG = DetectionConfig()
SCORING_CONFIG = ScoringConfig()

