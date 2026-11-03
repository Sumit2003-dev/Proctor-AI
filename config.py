"""Configuration module for ProctorAI.

Contains all configurable thresholds, weights, durations, 3D head-pose parameters,
and file paths. Avoids hardcoded constants throughout the application.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# Base paths
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
SNAPSHOTS_DIR = LOGS_DIR / "snapshots"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_PATH = LOGS_DIR / "alerts.csv"

# Model paths
DEFAULT_MODEL_NAME = "models/best.pt" if (BASE_DIR / "models" / "best.pt").exists() else "yolov8n.pt"
DATA_CONFIG_PATH = BASE_DIR / "data" / "data.yaml"


@dataclass(frozen=True)
class AudioConfig:
    """Microphone and acoustic monitoring configuration."""

    DEFAULT_RMS_THRESHOLD: float = 0.04
    SUSTAINED_AUDIO_SECONDS: float = 1.0
    SAMPLE_RATE: int = 16000


@dataclass(frozen=True)
class HeadPoseConfig:
    """3D Head-Pose Estimation parameters using cv2.solvePnP."""

    # 6 Canonical 3D Anthropometric Model Points (X, Y, Z in mm)
    MODEL_POINTS_3D: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),             # Nose tip (Landmark 1)
        (0.0, -330.0, -65.0),         # Chin (Landmark 152)
        (-225.0, 170.0, -135.0),      # Left eye outer corner (Landmark 33)
        (225.0, 170.0, -135.0),       # Right eye outer corner (Landmark 263)
        (-150.0, -150.0, -125.0),     # Left mouth corner (Landmark 61)
        (150.0, -150.0, -125.0),      # Right mouth corner (Landmark 291)
    )

    # Corresponding MediaPipe FaceMesh landmark indices
    LANDMARK_INDICES: Tuple[int, ...] = (1, 152, 33, 263, 61, 291)

    # Angular thresholds in degrees
    DEFAULT_YAW_THRESHOLD_DEG: float = 20.0     # Horizontal head rotation
    DEFAULT_PITCH_DOWN_THRESHOLD_DEG: float = 15.0 # Looking down at phone / notes
    DEFAULT_PITCH_UP_THRESHOLD_DEG: float = 25.0   # Looking up / away


@dataclass(frozen=True)
class DetectionConfig:
    """Detection thresholds and class IDs."""

    # Class IDs for custom fine-tuned weights (class 0: person, class 1: cell_phone)
    # and standard COCO fallback (class 0: person, class 67: cell_phone)
    PERSON_CLASS_IDS: Tuple[int, ...] = (0,)
    PHONE_CLASS_IDS: Tuple[int, ...] = (1, 67)

    # Confidence thresholds
    DEFAULT_CONF_THRESHOLD: float = 0.45
    MIN_CONF_THRESHOLD: float = 0.2
    MAX_CONF_THRESHOLD: float = 0.95

    # Head pose settings
    HEAD_POSE: HeadPoseConfig = HeadPoseConfig()

    # Duration threshold: lookaway must be sustained for >= this many seconds
    SUSTAINED_LOOKAWAY_SECONDS: float = 2.0

    # Snapshot evidence cooldown in seconds
    SNAPSHOT_COOLDOWN_SECONDS: float = 5.0


@dataclass(frozen=True)
class ScoringConfig:
    """Calibrated probability and scoring configuration."""

    # Rolling window length in seconds
    WINDOW_SECONDS: float = 30.0

    # Event cooldown to prevent frame-by-frame inflation
    EVENT_COOLDOWN_SECONDS: float = 1.0

    # Half-life for exponential time decay in seconds
    DECAY_HALF_LIFE_SECONDS: float = 15.0

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

    # Sigmoid calibration parameters for cheating probability
    SIGMOID_CENTER_THETA: float = 35.0
    SIGMOID_SCALE_BETA: float = 12.0

    # Maximum score cap
    MAX_SCORE: float = 100.0

    # Risk category thresholds (%)
    NORMAL_THRESHOLD: float = 30.0
    WARNING_THRESHOLD: float = 70.0


# Default global instances for convenience
AUDIO_CONFIG = AudioConfig()
DETECTION_CONFIG = DetectionConfig()
SCORING_CONFIG = ScoringConfig()
