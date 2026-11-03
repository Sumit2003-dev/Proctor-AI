"""Real-time detection engine combining YOLOv8 and MediaPipe Face Mesh.

Detects:
1. Cell phone in frame (YOLOv8 class 67)
2. Multiple people in frame (YOLOv8 class 0, count > 1)
3. No person in frame (YOLOv8 class 0, count == 0)
4. Looking away (Head yaw from MediaPipe landmarks 1, 33, 263, sustained > 2.0s)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
torch.set_num_threads(1)
from ultralytics import YOLO

from config import DETECTION_CONFIG, DetectionConfig, SNAPSHOTS_DIR


@dataclass
class DetectionResult:
    """Structured detection outcome for a single video frame."""
    annotated_frame: np.ndarray
    alerts: List[str]
    person_count: int
    phone_count: int
    head_direction: str
    yaw_ratio: float
    lookaway_duration: float
    snapshot_path: Optional[str] = None


class ProctorDetector:
    """Performs object detection and facial orientation analysis on video frames."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        config: DetectionConfig = DETECTION_CONFIG,
    ) -> None:
        self.config = config
        self.model = YOLO(model_path)

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # State tracking for sustained looking away
        self.lookaway_start_time: Optional[float] = None
        self.current_direction: str = "forward"
        self._last_snapshot_time: float = 0.0

    def calculate_head_direction(
        self,
        landmarks,
        yaw_threshold: float,
    ) -> Tuple[str, float]:
        """Estimates horizontal head direction (yaw) using Face Mesh key landmarks.

        Uses:
        - Landmark 1: Nose tip
        - Landmark 33: Left eye outer corner
        - Landmark 263: Right eye outer corner

        Returns:
            Tuple of (direction_str, yaw_ratio).
        """
        nose = landmarks[self.config.NOSE_TIP_IDX]
        l_eye = landmarks[self.config.LEFT_EYE_OUTER_IDX]
        r_eye = landmarks[self.config.RIGHT_EYE_OUTER_IDX]

        eye_mid_x = (l_eye.x + r_eye.x) / 2.0
        eye_dist = abs(r_eye.x - l_eye.x)
        yaw_ratio = (nose.x - eye_mid_x) / (eye_dist + 1e-6)

        if yaw_ratio > yaw_threshold:
            return "Looking Right", yaw_ratio
        elif yaw_ratio < -yaw_threshold:
            return "Looking Left", yaw_ratio
        return "Forward", yaw_ratio

    def analyze(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        yaw_threshold: Optional[float] = None,
        sustained_duration: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> DetectionResult:
        """Analyzes a frame for cheating behaviors.

        Args:
            frame: Input BGR image from OpenCV.
            conf_threshold: Optional override for YOLO detection confidence.
            yaw_threshold: Optional override for head yaw ratio.
            sustained_duration: Optional override for sustained lookaway trigger time.
            current_time: Optional explicit timestamp (defaults to time.time()).

        Returns:
            DetectionResult containing annotated frame, alerts list, counts, and metrics.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                alerts=["No frame available"],
                person_count=0,
                phone_count=0,
                head_direction="Unknown",
                yaw_ratio=0.0,
                lookaway_duration=0.0,
            )

        if conf_threshold is None:
            conf_threshold = self.config.DEFAULT_CONF_THRESHOLD
        if yaw_threshold is None:
            yaw_threshold = self.config.DEFAULT_YAW_THRESHOLD
        if sustained_duration is None:
            sustained_duration = self.config.SUSTAINED_LOOKAWAY_SECONDS
        if current_time is None:
            current_time = time.time()

        h, w = frame.shape[:2]
        annotated = frame.copy()
        alerts: List[str] = []

        # ---------------- 1. YOLOv8 Object Detection ----------------
        results = self.model(annotated, conf=conf_threshold, verbose=False)[0]
        person_count = 0
        phone_count = 0

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if conf < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if cls_id == self.config.PERSON_CLASS_ID:
                    person_count += 1
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"Person {conf:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                elif cls_id == self.config.CELL_PHONE_CLASS_ID:
                    phone_count += 1
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        annotated,
                        f"CELL PHONE {conf:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )

        # Flag object detection violations
        if phone_count > 0:
            alerts.append("Cell phone detected")

        if person_count == 0:
            alerts.append("No person in frame")
        elif person_count > 1:
            alerts.append("Multiple people detected")

        # ---------------- 2. MediaPipe Face Mesh Head Pose ----------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(rgb_frame)

        head_dir = "Forward"
        yaw_ratio = 0.0
        lookaway_elapsed = 0.0

        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0].landmark
            head_dir, yaw_ratio = self.calculate_head_direction(
                face_landmarks, yaw_threshold=yaw_threshold
            )
            self.current_direction = head_dir

            # Draw visual landmarks: nose tip and eye corners
            for idx, color in [
                (self.config.NOSE_TIP_IDX, (0, 255, 255)),       # Yellow
                (self.config.LEFT_EYE_OUTER_IDX, (255, 255, 0)), # Cyan
                (self.config.RIGHT_EYE_OUTER_IDX, (255, 255, 0)),# Cyan
            ]:
                pt = face_landmarks[idx]
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(annotated, (cx, cy), 4, color, -1)

            # Temporal sustained lookaway logic (> 2.0s sustained)
            if head_dir != "Forward":
                if self.lookaway_start_time is None:
                    self.lookaway_start_time = current_time
                lookaway_elapsed = current_time - self.lookaway_start_time

                if lookaway_elapsed >= sustained_duration:
                    alerts.append(f"Looking away sustained ({head_dir.lower()})")
                    # Visual red warning text for sustained lookaway
                    cv2.putText(
                        annotated,
                        f"GAZE ALERT: {head_dir.upper()} ({lookaway_elapsed:.1f}s)",
                        (w - 360, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                else:
                    # Looking away, but within grace period (< 2.0s)
                    cv2.putText(
                        annotated,
                        f"Distracted: {lookaway_elapsed:.1f}s / {sustained_duration:.1f}s",
                        (w - 300, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )
            else:
                self.lookaway_start_time = None
        else:
            # No face detected
            self.lookaway_start_time = None
            head_dir = "No face"

        # ---------------- 3. Evidentiary Snapshot Capture ----------------
        saved_snapshot: Optional[str] = None
        if alerts and (current_time - self._last_snapshot_time >= self.config.SNAPSHOT_COOLDOWN_SECONDS):
            self._last_snapshot_time = current_time
            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            clean_tag = alerts[0].lower().replace(" ", "_")[:18]
            snap_filename = f"{clean_tag}_{int(current_time)}.jpg"
            snap_full_path = SNAPSHOTS_DIR / snap_filename
            cv2.imwrite(str(snap_full_path), annotated)
            saved_snapshot = str(snap_full_path)

        # ---------------- 4. HUD Overlay ----------------
        # Top-left alert summary
        y_offset = 35
        for alert_msg in alerts:
            # Draw dark outline for readability
            cv2.putText(
                annotated,
                f"ALERT: {alert_msg}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                annotated,
                f"ALERT: {alert_msg}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )
            y_offset += 32

        # Bottom status bar
        status_text = (
            f"People: {person_count} | Phones: {phone_count} | "
            f"Gaze: {head_dir} (yaw: {yaw_ratio:+.2f})"
        )
        cv2.rectangle(annotated, (0, h - 30), (w, h), (30, 30, 30), -1)
        cv2.putText(
            annotated,
            status_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )

        return DetectionResult(
            annotated_frame=annotated,
            alerts=alerts,
            person_count=person_count,
            phone_count=phone_count,
            head_direction=head_dir,
            yaw_ratio=yaw_ratio,
            lookaway_duration=lookaway_elapsed,
            snapshot_path=saved_snapshot,
        )

    def close(self) -> None:
        """Releases MediaPipe and vision resources."""
        if hasattr(self, "face_mesh") and self.face_mesh:
            self.face_mesh.close()
