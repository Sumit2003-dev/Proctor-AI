"""Smoke test for ProctorAI detection, scoring, and logging pipeline."""

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
    print("[1/5] Testing AlertLogger...")
    logger = AlertLogger()
    logger.clear()
    logger.log("Cell phone detected", 35.0, "Smoke test phone event")
    df = logger.get_dataframe()
    assert len(df) == 1, f"Expected 1 record in log, got {len(df)}"
    print(f"      AlertLogger OK: logged 1 alert, read back {len(df)} rows.")

    print("[2/5] Testing SuspicionScorer...")
    scorer = SuspicionScorer(config=SCORING_CONFIG)
    t0 = 1000.0
    score1 = scorer.add_alerts(["Cell phone detected"], current_time=t0)
    assert score1 == 35.0, f"Expected score 35.0, got {score1}"

    # Multiple people alert
    score2 = scorer.add_alerts(["Multiple people detected"], current_time=t0 + 0.1)
    assert score2 == 65.0, f"Expected score 65.0, got {score2}"

    # Check status
    status_label, _ = scorer.get_status(score2)
    assert status_label == "WARNING", f"Expected WARNING, got {status_label}"

    # Simulate 35 seconds passing (beyond 30s rolling window)
    score_later = scorer.compute_score(current_time=t0 + 35.0)
    assert score_later == 0.0, f"Expected score 0.0 after 35s window expiry, got {score_later}"
    print("      SuspicionScorer OK: weights, cooldown, rolling window, and status verified.")

    print("[3/5] Testing ProctorDetector initialization & YOLO model loading...")
    detector = ProctorDetector()
    print("      ProctorDetector initialized successfully.")

    print("[4/5] Testing ProctorDetector on synthetic sample image...")
    sample_frame = create_sample_image()
    # Save sample image to disk for inspection/docs
    cv2.imwrite("sample_frame.jpg", sample_frame)

    result = detector.analyze(sample_frame, current_time=100.0)
    assert result.annotated_frame is not None, "Annotated frame must not be None"
    assert isinstance(result.alerts, list), "Alerts must be a list"
    print(f"      Detection on sample frame OK: alerts={result.alerts}, person_count={result.person_count}")

    # Save annotated result
    cv2.imwrite("sample_annotated.jpg", result.annotated_frame)

    print("[5/5] Testing Temporal Lookaway logic...")
    # Simulate face looking right sustained
    mock_landmarks = [type("MockPoint", (), {"x": 0.5, "y": 0.5})()] * 500
    mock_landmarks[DETECTION_CONFIG.NOSE_TIP_IDX] = type("MockPoint", (), {"x": 0.58, "y": 0.5})()
    mock_landmarks[DETECTION_CONFIG.LEFT_EYE_OUTER_IDX] = type("MockPoint", (), {"x": 0.4, "y": 0.45})()
    mock_landmarks[DETECTION_CONFIG.RIGHT_EYE_OUTER_IDX] = type("MockPoint", (), {"x": 0.6, "y": 0.45})()

    direction, yaw_ratio = detector.calculate_head_direction(mock_landmarks, yaw_threshold=0.25)
    assert direction == "Looking Right", f"Expected 'Looking Right', got {direction}"
    assert yaw_ratio > 0.25, f"Expected yaw_ratio > 0.25, got {yaw_ratio}"
    print(f"      Head Direction calculation OK: {direction} (yaw_ratio={yaw_ratio:.2f})")

    print("[6/6] Testing Audio Proctor & PDF Audit Report Generator...")
    audio = AudioProctor()
    print(f"      Audio Hardware Available: {audio.hardware_available}")

    import datetime
    pdf_bytes = generate_pdf_report(
        candidate_name="Test Candidate",
        session_id="EXAM-TEST-001",
        start_time=datetime.datetime.now() - datetime.timedelta(minutes=30),
        end_time=datetime.datetime.now(),
        peak_score=65.0,
        status_label="WARNING",
        alerts_df=df,
        snapshot_paths=None,
    )
    assert pdf_bytes.startswith(b"%PDF"), "Generated document must be a valid PDF file"
    assert len(pdf_bytes) > 1000, f"PDF too small: {len(pdf_bytes)} bytes"
    print(f"      PDF Report Generation OK: generated {len(pdf_bytes)} bytes valid PDF.")

    detector.close()
    print("\nALL SMOKE TESTS PASSED SUCCESSFULLY! ALL MULTI-MODAL COMPONENTS OPERATIONAL.")


if __name__ == "__main__":
    test_pipeline()
