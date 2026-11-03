"""Streamlit Dashboard for ProctorAI - Real-Time AI Exam Proctoring System.

Features:
- Live webcam streaming with real-time HUD annotations.
- MediaPipe Face Mesh sustained lookaway gaze tracking (>2.0s).
- Acoustic / microphone monitoring for speech and whispering detection.
- Active alert panel and 30-second rolling suspicion score gauge.
- Sidebar controls for detection sensitivity, audio threshold, and duration thresholds.
- Dynamic alert history log table with CSV and PDF Audit Report export.
- Evidentiary snapshot capture and inspection.
"""

import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
torch.set_num_threads(1)
from PIL import Image

from audio_monitor import AudioProctor
from config import (
    AUDIO_CONFIG,
    DEFAULT_LOG_PATH,
    DEFAULT_MODEL_NAME,
    DETECTION_CONFIG,
    SCORING_CONFIG,
    SNAPSHOTS_DIR,
)
from detector import ProctorDetector
from logger import AlertLogger
from report import generate_pdf_report
from scoring import SuspicionScorer

# Page configuration
st.set_page_config(
    page_title="ProctorAI - Multi-Modal Exam Proctoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
        border: 1px solid #333a4d;
    }
    .score-badge {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        padding: 6px 12px;
        border-radius: 8px;
    }
    .status-normal { color: #28a745; border: 2px solid #28a745; background-color: rgba(40, 167, 69, 0.1); }
    .status-warning { color: #fd7e14; border: 2px solid #fd7e14; background-color: rgba(253, 126, 20, 0.1); }
    .status-danger  { color: #dc3545; border: 2px solid #dc3545; background-color: rgba(220, 53, 69, 0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector() -> ProctorDetector:
    """Loads and caches the ProctorDetector instance."""
    return ProctorDetector(model_path=DEFAULT_MODEL_NAME, config=DETECTION_CONFIG)


@st.cache_resource
def load_audio_proctor() -> AudioProctor:
    """Loads and caches the AudioProctor instance."""
    return AudioProctor(
        sample_rate=AUDIO_CONFIG.SAMPLE_RATE,
        default_threshold=AUDIO_CONFIG.DEFAULT_RMS_THRESHOLD,
        sustained_seconds=AUDIO_CONFIG.SUSTAINED_AUDIO_SECONDS,
    )


def get_scorer() -> SuspicionScorer:
    """Retrieves or initializes the SuspicionScorer in session state."""
    if "scorer" not in st.session_state:
        st.session_state.scorer = SuspicionScorer(config=SCORING_CONFIG)
    return st.session_state.scorer


def get_logger() -> AlertLogger:
    """Retrieves or initializes the AlertLogger in session state."""
    if "logger" not in st.session_state:
        st.session_state.logger = AlertLogger(log_path=DEFAULT_LOG_PATH)
    return st.session_state.logger


def init_session_metadata():
    """Initializes session start time and metadata if missing."""
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.datetime.now()
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"EXAM-{int(time.time())}"
    if "peak_score" not in st.session_state:
        st.session_state.peak_score = 0.0
    if "evidence_snapshots" not in st.session_state:
        st.session_state.evidence_snapshots = []


def render_sidebar():
    """Renders configuration and control widgets in the sidebar."""
    st.sidebar.title("🛡️ ProctorAI Controls")
    st.sidebar.markdown("Configure multi-modal proctoring thresholds.")

    # Candidate Profile
    st.sidebar.subheader("Candidate Session")
    candidate_name = st.sidebar.text_input("Candidate Name", value="John Doe")
    session_id = st.session_state.get("session_id", "EXAM-1001")
    st.sidebar.caption(f"Session ID: `{session_id}`")

    # Vision Sensitivity Sliders
    st.sidebar.subheader("Vision Sensitivity")
    conf_thresh = st.sidebar.slider(
        "YOLO Confidence Threshold",
        min_value=0.20,
        max_value=0.90,
        value=float(DETECTION_CONFIG.DEFAULT_CONF_THRESHOLD),
        step=0.05,
        help="Higher values reduce false positive object detections.",
    )

    yaw_thresh = st.sidebar.slider(
        "Head Yaw Sensitivity",
        min_value=0.15,
        max_value=0.45,
        value=float(DETECTION_CONFIG.DEFAULT_YAW_THRESHOLD),
        step=0.02,
        help="Lower values trigger 'looking away' on smaller head turns.",
    )

    sustained_secs = st.sidebar.slider(
        "Sustained Lookaway Duration (sec)",
        min_value=1.0,
        max_value=5.0,
        value=float(DETECTION_CONFIG.SUSTAINED_LOOKAWAY_SECONDS),
        step=0.5,
        help="Alert only triggers if candidate looks away for longer than this duration.",
    )

    # Audio Proctoring Controls
    st.sidebar.subheader("Acoustic Monitoring")
    audio_proctor = load_audio_proctor()
    enable_audio = st.sidebar.checkbox(
        "Enable Audio/Whisper Detection",
        value=audio_proctor.hardware_available,
        disabled=not audio_proctor.hardware_available,
        help="Detects candidate speaking, whispering, or external verbal coaching.",
    )
    if not audio_proctor.hardware_available:
        st.sidebar.caption("⚠️ No microphone device detected.")

    audio_thresh = st.sidebar.slider(
        "Microphone Sensitivity (RMS)",
        min_value=0.01,
        max_value=0.15,
        value=float(AUDIO_CONFIG.DEFAULT_RMS_THRESHOLD),
        step=0.01,
        help="Lower value increases microphone sensitivity.",
    )
    audio_proctor.threshold = audio_thresh

    st.sidebar.markdown("---")
    st.sidebar.subheader("Reports & Exports")

    alert_logger = get_logger()
    csv_bytes = alert_logger.get_csv_bytes()

    st.sidebar.download_button(
        label="📥 Download Alert Log (CSV)",
        data=csv_bytes,
        file_name=f"{session_id}_alerts.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Generate and serve PDF Audit Report
    alerts_df = alert_logger.get_dataframe(limit=None)
    scorer = get_scorer()
    current_score = scorer.compute_score()
    status_label, _ = scorer.get_status(current_score)

    # Collect existing snapshots
    existing_snapshots = list(SNAPSHOTS_DIR.glob("*.jpg"))
    pdf_bytes = generate_pdf_report(
        candidate_name=candidate_name,
        session_id=session_id,
        start_time=st.session_state.session_start,
        end_time=datetime.datetime.now(),
        peak_score=max(st.session_state.peak_score, current_score),
        status_label=status_label,
        alerts_df=alerts_df,
        snapshot_paths=existing_snapshots,
    )

    st.sidebar.download_button(
        label="📄 Download Audit Report (PDF)",
        data=pdf_bytes,
        file_name=f"{session_id}_integrity_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    if st.sidebar.button("🔄 Reset Session & Logs", use_container_width=True):
        get_scorer().reset()
        alert_logger.clear()
        st.session_state.session_start = datetime.datetime.now()
        st.session_state.peak_score = 0.0
        st.session_state.session_id = f"EXAM-{int(time.time())}"
        st.sidebar.success("Session state & alert logs reset.")
        st.rerun()

    return conf_thresh, yaw_thresh, sustained_secs, enable_audio


def main():
    init_session_metadata()
    st.title("🛡️ ProctorAI - Multi-Modal AI Exam Proctoring System")
    st.markdown(
        "Comprehensive real-time integrity monitoring: cell phones, multi-person detection, absence, head pose gaze tracking, and acoustic anomaly detection."
    )

    conf_thresh, yaw_thresh, sustained_secs, enable_audio = render_sidebar()
    detector = load_detector()
    scorer = get_scorer()
    alert_logger = get_logger()
    audio_proctor = load_audio_proctor()

    if enable_audio:
        audio_proctor.start()
    else:
        audio_proctor.stop()

    # Input Mode Selection
    input_mode = st.radio(
        "Video Source:",
        ["Live Webcam", "Upload Test Image / Video Frame"],
        horizontal=True,
    )

    # Layout: 2 Columns
    col_video, col_dash = st.columns([3, 2])

    with col_dash:
        st.subheader("📊 Integrity Dashboard")
        score_placeholder = st.empty()
        audio_placeholder = st.empty()
        alerts_placeholder = st.empty()
        table_placeholder = st.empty()

    if input_mode == "Live Webcam":
        with col_video:
            st.subheader("📹 Live Monitoring Feed")
            run_webcam = st.toggle("Start Webcam Monitoring", value=False)
            camera_slot = st.empty()

        if run_webcam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                camera_slot.error(
                    "❌ Camera Error: Could not access webcam device 0. "
                    "Ensure webcam permissions are enabled and no other application is using it."
                )
            else:
                try:
                    while run_webcam:
                        success, frame = cap.read()
                        if not success:
                            camera_slot.warning("Failed to receive frame from webcam.")
                            break

                        now = time.time()
                        # Analyze video frame
                        result = detector.analyze(
                            frame=frame,
                            conf_threshold=conf_thresh,
                            yaw_threshold=yaw_thresh,
                            sustained_duration=sustained_secs,
                            current_time=now,
                        )

                        active_alerts = list(result.alerts)

                        # Audio anomaly detection check
                        audio_flagged, current_rms, _ = audio_proctor.check_status()
                        if enable_audio and audio_flagged:
                            active_alerts.append("Suspicious audio/speech detected")

                        # Update suspicion score
                        current_score = scorer.add_alerts(active_alerts, current_time=now)
                        if current_score > st.session_state.peak_score:
                            st.session_state.peak_score = current_score

                        status_label, status_color = scorer.get_status(current_score)

                        # Log violations
                        for alert_item in active_alerts:
                            alert_logger.log(
                                alert_type=alert_item,
                                suspicion_score=current_score,
                                details=(
                                    f"Persons: {result.person_count}, "
                                    f"Phones: {result.phone_count}, "
                                    f"Gaze: {result.head_direction}, "
                                    f"AudioRMS: {current_rms:.3f}"
                                ),
                                timestamp=None,
                            )

                        # Display webcam feed (convert BGR to RGB for Streamlit)
                        rgb_display = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                        camera_slot.image(rgb_display, channels="RGB", use_container_width=True)

                        # Render Dashboard
                        _render_score_and_alerts(
                            score_placeholder,
                            audio_placeholder,
                            alerts_placeholder,
                            current_score,
                            status_label,
                            status_color,
                            active_alerts,
                            current_rms if enable_audio else None,
                        )

                        # Update Log Table
                        recent_df = alert_logger.get_dataframe(limit=10)
                        table_placeholder.dataframe(recent_df, use_container_width=True, hide_index=True)

                        time.sleep(0.03)  # approx ~30 fps cap
                finally:
                    cap.release()
                    audio_proctor.stop()
        else:
            camera_slot.info("Click 'Start Webcam Monitoring' above to begin.")
            current_score = scorer.compute_score()
            status_label, status_color = scorer.get_status(current_score)
            _render_score_and_alerts(
                score_placeholder,
                audio_placeholder,
                alerts_placeholder,
                current_score,
                status_label,
                status_color,
                [],
                None,
            )
            table_placeholder.dataframe(
                alert_logger.get_dataframe(limit=10),
                use_container_width=True,
                hide_index=True,
            )

    else:
        # Fallback Test Mode with Uploaded Image
        with col_video:
            st.subheader("🖼️ Test Frame Analysis")
            uploaded_file = st.file_uploader(
                "Upload a test image (e.g. testing phone detection or lookaway)",
                type=["jpg", "jpeg", "png"],
            )

            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                now = time.time()
                result = detector.analyze(
                    frame=frame,
                    conf_threshold=conf_thresh,
                    yaw_threshold=yaw_thresh,
                    sustained_duration=sustained_secs,
                    current_time=now,
                )

                current_score = scorer.add_alerts(result.alerts, current_time=now)
                status_label, status_color = scorer.get_status(current_score)

                for alert_item in result.alerts:
                    alert_logger.log(
                        alert_type=alert_item,
                        suspicion_score=current_score,
                        details=f"Test Image - Gaze: {result.head_direction}",
                    )

                rgb_display = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_display, caption="Annotated Result", use_container_width=True)

                _render_score_and_alerts(
                    score_placeholder,
                    audio_placeholder,
                    alerts_placeholder,
                    current_score,
                    status_label,
                    status_color,
                    result.alerts,
                    None,
                )
                table_placeholder.dataframe(
                    alert_logger.get_dataframe(limit=10),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Upload an image to run detection on static test cases.")

    # Photographic Evidence Gallery
    st.markdown("---")
    st.subheader("📸 Captured Evidentiary Snapshots")
    snapshot_files = list(SNAPSHOTS_DIR.glob("*.jpg"))
    if snapshot_files:
        cols = st.columns(min(4, len(snapshot_files)))
        for i, snap in enumerate(reversed(snapshot_files[-8:])):
            with cols[i % len(cols)]:
                img = Image.open(snap)
                st.image(img, caption=snap.stem, use_container_width=True)
    else:
        st.caption("No violation snapshots recorded yet in this session.")


def _render_score_and_alerts(
    score_slot,
    audio_slot,
    alerts_slot,
    score: float,
    status_label: str,
    status_color: str,
    alerts: list,
    audio_rms: Optional[float] = None,
):
    """Helper to render score cards, audio meters, and active alerts."""
    tier_class = (
        "status-normal"
        if status_label == "NORMAL"
        else "status-warning" if status_label == "WARNING" else "status-danger"
    )

    with score_slot.container():
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.95rem; color: #a0aec0; margin-bottom: 4px;">
                    ROLLING 30S SUSPICION SCORE
                </div>
                <div class="score-badge {tier_class}">
                    {score:.0f} / 100 &nbsp;•&nbsp; {status_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, max(0.0, score / 100.0)))

    with audio_slot.container():
        if audio_rms is not None:
            st.caption(f"🎙️ Microphone Level (RMS): {audio_rms:.4f}")
            st.progress(min(1.0, max(0.0, audio_rms / 0.15)))

    with alerts_slot.container():
        if alerts:
            st.error("⚠️ **Active Violations Detected:**")
            for alert in alerts:
                st.markdown(f"- **{alert}**")
        else:
            st.success("✅ **Candidate Status**: Normal (No Active Violations)")


if __name__ == "__main__":
    main()
