"""Comprehensive smoke test for ProctorAI multi-modal and 3D solvePnP pipeline."""

import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import cv2
import numpy as np
import torch
torch.set_num_threads(1)

from audio_monitor import AudioProctor
from config import DETECTION_CONFIG, SCORING_CONFIG, SNAPSHOTS_DIR
from detector import ProctorDetector
from logger import AlertLogger
from report import generate_pdf_report
from scoring import SuspicionScorer


def create_sample_image() -> np.ndarray:
    """Creates a synthetic test frame containing an oval 'face' and mock eyes/nose."""
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    # Draw a head / face ellipse
    center = (320, 240)
    axes = (100, 140)
    cv2.ellipse(frame, center, axes, 0, 0, 360, (180, 200, 220), -1)

    # Draw two eyes
    cv2.circle(frame, (280, 210), 12, (50, 50, 50), -1)
    cv2.circle(frame, (360, 210), 12, (50, 50, 50), -1)

    # Draw nose
    cv2.circle(frame, (320, 245), 6, (70, 70, 70), -1)

    # Draw mouth
    cv2.ellipse(frame, (320, 290), (35, 10), 0, 0, 180, (50, 50, 180), 2)

    return frame


def test_pipeline():
    print("[1/6] Testing AlertLogger...")
    logger = AlertLogger()
    logger.clear()
    logger.log("Cell phone detected", 35.0, "Smoke test phone event")
    df = logger.get_dataframe()
    assert len(df) == 1, f"Expected 1 record in log, got {len(df)}"
    print(f"      AlertLogger OK: logged 1 alert, read back {len(df)} rows.")

    print("[2/6] Testing Calibrated SuspicionScorer & Cooldown Pruning...")
    scorer = SuspicionScorer(config=SCORING_CONFIG)
    t0 = 1000.0
    score1 = scorer.add_alerts(["Cell phone detected"], current_time=t0)
    assert score1 > 0.0, f"Expected score > 0, got {score1}"

    prob1 = scorer.get_cheating_probability(score1)
    assert 0.0 <= prob1 <= 1.0, f"Probability must be in [0, 1], got {prob1}"
    print(f"      Scorer OK: Score={score1:.1f}, Calibrated P(Cheating)={prob1:.1%}")

    # Multiple people alert
    score2 = scorer.add_alerts(["Multiple people detected"], current_time=t0 + 0.1)
    assert score2 > score1, f"Expected score to increase, got {score2}"

    # Verify stale cooldown expiration
    scorer.prune_old_records(current_time=t0 + 35.0)
    assert len(scorer._last_alert_logged_time) == 0, "Cooldown state should be pruned after window expiration"
    assert scorer.compute_score(current_time=t0 + 35.0) == 0.0, "Score should decay to 0 after window"
    print("      Cooldown Pruning & Window Decay OK.")

    print("[3/6] Testing ProctorDetector initialization (with fine-tuned models/best.pt fallback)...")
    detector = ProctorDetector()
    print("      ProctorDetector initialized successfully.")

    print("[4/6] Testing ProctorDetector on synthetic sample frame...")
    sample_frame = create_sample_image()
    cv2.imwrite("sample_frame.jpg", sample_frame)

    result = detector.analyze(sample_frame, current_time=100.0)
    assert result.annotated_frame is not None, "Annotated frame must not be None"
    assert isinstance(result.alerts, list), "Alerts must be a list"
    cv2.imwrite("sample_annotated.jpg", result.annotated_frame)
    print(f"      Detection OK: alerts={result.alerts}, person_count={result.person_count}")

    print("[5/6] Testing 3D Head-Pose Estimation via cv2.solvePnP...")
    # Mock landmarks for 6 canonical points:
    # 1: Nose tip, 152: Chin, 33: Left eye, 263: Right eye, 61: Left mouth, 291: Right mouth
    mock_landmarks = [type("MockPoint", (), {"x": 0.5, "y": 0.5})()] * 500
    mock_landmarks[1] = type("MockPoint", (), {"x": 0.56, "y": 0.5})()    # Nose shifted right
    mock_landmarks[152] = type("MockPoint", (), {"x": 0.54, "y": 0.75})()  # Chin
    mock_landmarks[33] = type("MockPoint", (), {"x": 0.42, "y": 0.4})()   # Left eye
    mock_landmarks[263] = type("MockPoint", (), {"x": 0.60, "y": 0.4})()  # Right eye
    mock_landmarks[61] = type("MockPoint", (), {"x": 0.47, "y": 0.62})()  # Left mouth
    mock_landmarks[291] = type("MockPoint", (), {"x": 0.58, "y": 0.62})() # Right mouth

    pose = detector.estimate_head_pose_3d(
        mock_landmarks,
        frame_w=640,
        frame_h=480,
        yaw_threshold=20.0,
        pitch_down_threshold=15.0,
        pitch_up_threshold=25.0,
    )
    assert isinstance(pose.pitch, float) and isinstance(pose.yaw, float) and isinstance(pose.roll, float)
    print(f"      solvePnP 3D Pose OK: Direction={pose.direction}, Yaw={pose.yaw:.1f}°, Pitch={pose.pitch:.1f}°, Roll={pose.roll:.1f}°")
    assert pose.gaze_start_pt is not None and pose.gaze_end_pt is not None, "3D Gaze vector must be projected"
    print("      3D Gaze Vector Projection OK.")

    print("[6/6] Testing Audio Proctor & PDF Audit Report Generator...")
    audio = AudioProctor()
    print(f"      Audio Hardware Available: {audio.hardware_available}")

    pdf_bytes = generate_pdf_report(
        candidate_name="Jane Doe",
        session_id="EXAM-VERIFIED-2026",
        start_time=datetime.datetime.now() - datetime.timedelta(minutes=45),
        end_time=datetime.datetime.now(),
        peak_score=68.0,
        status_label="WARNING",
        alerts_df=df,
        snapshot_paths=None,
    )
    assert pdf_bytes.startswith(b"%PDF"), "Must be a valid PDF document"
    assert len(pdf_bytes) > 1000, "PDF file length invalid"
    print(f"      PDF Report Generation OK: generated {len(pdf_bytes)} bytes valid PDF.")

    detector.close()
    print("\nALL SMOKE TESTS PASSED! 3D HEAD-POSE, CALIBRATED SCORING, AND LIFECYCLE VERIFIED.")


if __name__ == "__main__":
    test_pipeline()
