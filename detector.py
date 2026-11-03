"""Real-time detection engine combining YOLOv8 and 3D Head-Pose Estimation via solvePnP.

Detects:
1. Cell phone in frame (fine-tuned custom class 1 or COCO class 67).
2. Multiple people in frame (class 0, count > 1).
3. No person in frame (class 0, count == 0).
4. Looking away (Rigorous 3D Head-Pose via cv2.solvePnP: Pitch, Yaw, Roll, sustained > 2.0s).
5. 3D Gaze vector projection for visual HUD tracking.
"""

import math
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
class HeadPoseResult:
    """Rigorous 3D Head-Pose Estimation output in angular degrees."""
    pitch: float
    yaw: float
    roll: float
    direction: str
    gaze_start_pt: Optional[Tuple[int, int]] = None
    gaze_end_pt: Optional[Tuple[int, int]] = None


@dataclass
class DetectionResult:
    """Structured detection outcome for a single video frame."""
    annotated_frame: np.ndarray
    alerts: List[str]
    person_count: int
    phone_count: int
    head_direction: str
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    lookaway_duration: float
    snapshot_path: Optional[str] = None


class ProctorDetector:
    """Performs object detection and 3D facial orientation analysis on video frames."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        config: DetectionConfig = DETECTION_CONFIG,
    ) -> None:
        self.config = config

        # Load model with fallback to yolov8n.pt
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"
        self.model = YOLO(model_path)
        self.model_path = model_path

        # Dynamically resolve class IDs from model.names instead of hardcoding
        self.person_class_ids = [
            cls_id for cls_id, name in self.model.names.items()
            if "person" in name.lower()
        ]
        self.phone_class_ids = [
            cls_id for cls_id, name in self.model.names.items()
            if any(k in name.lower() for k in ["phone", "cell", "mobile"])
        ]
        print(f"[Detector] Loaded model: {model_path}")
        print(f"[Detector] Detected class mapping: Person IDs={self.person_class_ids}, Phone IDs={self.phone_class_ids}")
        print(f"[Detector] Full model.names: {self.model.names}")

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
        self.current_direction: str = "Forward"
        self._last_snapshot_time: float = 0.0

        # 3D Canonical anthropometric head model points (in mm)
        self.model_points_3d = np.array(
            self.config.HEAD_POSE.MODEL_POINTS_3D, dtype=np.float64
        )

    def estimate_head_pose_3d(
        self,
        landmarks,
        frame_w: int,
        frame_h: int,
        yaw_threshold: float,
        pitch_down_threshold: float,
        pitch_up_threshold: float,
    ) -> HeadPoseResult:
        """Computes true 3D Head-Pose (Pitch, Yaw, Roll) via OpenCV cv2.solvePnP.

        Extracts:
        - Landmark 1: Nose tip
        - Landmark 152: Chin
        - Landmark 33: Left eye outer corner
        - Landmark 263: Right eye outer corner
        - Landmark 61: Left mouth corner
        - Landmark 291: Right mouth corner
        """
        # Map 2D image coordinates from normalized landmark coordinates
        image_points = []
        for idx in self.config.HEAD_POSE.LANDMARK_INDICES:
            pt = landmarks[idx]
            image_points.append([float(pt.x * frame_w), float(pt.y * frame_h)])
        image_points_2d = np.array(image_points, dtype=np.float64)

        # Camera intrinsic matrix estimation
        focal_length = float(frame_w)
        center = (float(frame_w / 2.0), float(frame_h / 2.0))
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.model_points_3d,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return HeadPoseResult(
                pitch=0.0, yaw=0.0, roll=0.0, direction="Forward"
            )

        # Convert rotation vector to 3x3 rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x_rot = math.atan2(rmat[2, 1], rmat[2, 2])
            y_rot = math.atan2(-rmat[2, 0], sy)
            z_rot = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x_rot = math.atan2(-rmat[1, 2], rmat[1, 1])
            y_rot = math.atan2(-rmat[2, 0], sy)
            z_rot = 0.0

        # Convert radians to degrees
        # Pitch: tilting head down/up
        # Yaw: turning head left/right
        # Roll: tilting head sideways
        pitch = float(math.degrees(x_rot))
        # Normalize pitch around forward gaze baseline (~170-180° depending on coordinate frame)
        if pitch > 90.0:
            pitch = pitch - 180.0
        elif pitch < -90.0:
            pitch = pitch + 180.0

        yaw = float(math.degrees(y_rot))
        roll = float(math.degrees(z_rot))

        # Project a 3D nose gaze axis vector for visual overlay
        nose_end_3d = np.array([(0.0, 0.0, 750.0)], dtype=np.float64)
        nose_end_2d, _ = cv2.projectPoints(
            nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs
        )
        p1 = (int(image_points_2d[0][0]), int(image_points_2d[0][1]))
        p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))

        # Direction classification based on angular degrees
        if yaw > yaw_threshold:
            direction = "Looking Left"
        elif yaw < -yaw_threshold:
            direction = "Looking Right"
        elif pitch < -pitch_down_threshold:
            direction = "Looking Down"
        elif pitch > pitch_up_threshold:
            direction = "Looking Up"
        else:
            direction = "Forward"

        return HeadPoseResult(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            direction=direction,
            gaze_start_pt=p1,
            gaze_end_pt=p2,
        )

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
            yaw_threshold: Optional override for head yaw threshold in degrees.
            sustained_duration: Optional override for sustained lookaway trigger time.
            current_time: Optional explicit timestamp (defaults to time.time()).

        Returns:
            DetectionResult containing annotated frame, alerts list, counts, and 3D metrics.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                alerts=["No frame available"],
                person_count=0,
                phone_count=0,
                head_direction="Unknown",
                pitch_deg=0.0,
                yaw_deg=0.0,
                roll_deg=0.0,
                lookaway_duration=0.0,
                snapshot_path=None,
            )

        if conf_threshold is None:
            conf_threshold = self.config.DEFAULT_CONF_THRESHOLD
        if yaw_threshold is None:
            yaw_threshold = self.config.HEAD_POSE.DEFAULT_YAW_THRESHOLD_DEG
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
                label = self.model.names.get(cls_id, f"class_{cls_id}")

                # Raw detection diagnostic print
                print(f"DETECTION: {label} (ID: {cls_id}) | Conf: {conf:.3f}")

                if conf < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if cls_id in self.person_class_ids:
                    person_count += 1
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{label} {conf:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                elif cls_id in self.phone_class_ids:
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

        # ---------------- 2. 3D Head-Pose Estimation (solvePnP) ----------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(rgb_frame)

        head_dir = "Forward"
        pitch_deg = 0.0
        yaw_deg = 0.0
        roll_deg = 0.0
        lookaway_elapsed = 0.0

        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0].landmark
            pose = self.estimate_head_pose_3d(
                face_landmarks,
                frame_w=w,
                frame_h=h,
                yaw_threshold=yaw_threshold,
                pitch_down_threshold=self.config.HEAD_POSE.DEFAULT_PITCH_DOWN_THRESHOLD_DEG,
                pitch_up_threshold=self.config.HEAD_POSE.DEFAULT_PITCH_UP_THRESHOLD_DEG,
            )

            head_dir = pose.direction
            pitch_deg = pose.pitch
            yaw_deg = pose.yaw
            roll_deg = pose.roll
            self.current_direction = head_dir

            # Draw 3D Gaze Vector Ray
            if pose.gaze_start_pt and pose.gaze_end_pt:
                ray_color = (0, 255, 0) if head_dir == "Forward" else (0, 0, 255)
                cv2.line(annotated, pose.gaze_start_pt, pose.gaze_end_pt, ray_color, 3)
                cv2.circle(annotated, pose.gaze_start_pt, 4, (0, 255, 255), -1)

            # Draw the 6 Key 3D Facial Landmarks
            for idx in self.config.HEAD_POSE.LANDMARK_INDICES:
                pt = face_landmarks[idx]
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(annotated, (cx, cy), 3, (255, 255, 0), -1)

            # Temporal sustained lookaway logic (> 2.0s sustained)
            if head_dir != "Forward":
                if self.lookaway_start_time is None:
                    self.lookaway_start_time = current_time
                lookaway_elapsed = current_time - self.lookaway_start_time

                if lookaway_elapsed >= sustained_duration:
                    alerts.append(f"Looking away sustained ({head_dir.lower()})")
                    cv2.putText(
                        annotated,
                        f"GAZE ALERT: {head_dir.upper()} ({lookaway_elapsed:.1f}s)",
                        (w - 380, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        annotated,
                        f"Distracted: {lookaway_elapsed:.1f}s / {sustained_duration:.1f}s",
                        (w - 320, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )
            else:
                self.lookaway_start_time = None
        else:
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
        y_offset = 35
        for alert_msg in alerts:
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

        # Bottom 3D Head-Pose & Status Bar
        status_text = (
            f"People: {person_count} | Phones: {phone_count} | "
            f"Pose (solvePnP): {head_dir} [Yaw: {yaw_deg:+.1f}°, Pitch: {pitch_deg:+.1f}°]"
        )
        cv2.rectangle(annotated, (0, h - 30), (w, h), (25, 25, 25), -1)
        cv2.putText(
            annotated,
            status_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
        )

        return DetectionResult(
            annotated_frame=annotated,
            alerts=alerts,
            person_count=person_count,
            phone_count=phone_count,
            head_direction=head_dir,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
            lookaway_duration=lookaway_elapsed,
            snapshot_path=saved_snapshot,
        )

    def close(self) -> None:
        """Releases MediaPipe and vision resources."""
        if hasattr(self, "face_mesh") and self.face_mesh:
            self.face_mesh.close()
