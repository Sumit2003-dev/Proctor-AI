# ProctorAI: Real-Time AI Exam Proctoring System 🛡️

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://docs.ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/Google-MediaPipe-orange.svg)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

**ProctorAI** is an automated, real-time exam proctoring system engineered to detect integrity violations during online assessments using computer vision and deep learning.

---

## 🎯 Key Detection Features

1. **Cell Phone in Frame**: Flags unauthorized mobile devices via YOLOv8 (COCO class `67`).
2. **Multiple People in Frame**: Flags presence of unauthorized secondary persons (COCO class `0`, count > 1).
3. **Absence Detection**: Flags when the candidate leaves the camera view (person count = 0).
4. **Sustained Lookaway (Head Yaw Estimation)**: Estimates horizontal head pose via MediaPipe Face Mesh key landmarks (nose tip `1`, outer eye corners `33` and `263`). To eliminate false positives from quick natural glances, violations are triggered **only if looking away is sustained for $> 2.0$ seconds**.
5. **Rolling 30-Second Suspicion Score**: Dynamically calculates a normalized $(0 - 100)$ threat score over a sliding 30-second window, categorized into **Normal**, **Warning**, and **High Risk**.

---

## 📸 Dashboard & Detection Previews

| Real-Time Dashboard UI | Detection HUD Preview |
|:---:|:---:|
| ![ProctorAI Dashboard](proctor_dashboard.png) | ![Detection Preview](sample_annotated.jpg) |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    Webcam[Webcam Video Stream] --> Detector["detector.py: YOLOv8 + MediaPipe"]
    Detector --> Detections[Violations & Annotated Frame]
    Detections --> Scorer["scoring.py: 30s Rolling Window Engine"]
    Detections --> Logger["logger.py: alerts.csv Logger"]
    Scorer --> UI["app.py: Streamlit Dashboard"]
    Logger --> UI
    UI --> Controls["Sidebar Threshold Sliders & CSV Export"]
    TrainScript["train.py / evaluate.py"] -.-> Checkpoints["models/ (baseline / best.pt)"]
    Checkpoints -.-> Detector
```

---

## 📂 Project Structure

```text
proctor-ai/
├── app.py              # Streamlit dashboard: live webcam, gauge, alerts, history table
├── config.py           # Centralized configuration (thresholds, weights, timings, paths)
├── detector.py         # YOLOv8 + MediaPipe Face Mesh detection engine
├── scoring.py          # Rolling 30-second weighted suspicion scoring engine
├── logger.py           # Thread-safe CSV alert logger
├── train.py            # CLI fine-tuning script for YOLOv8 (data.yaml)
├── evaluate.py         # Comparative evaluation tool (Baseline vs Fine-tuned)
├── test_smoke.py       # Automated smoke test suite
├── requirements.txt    # Python dependencies
├── .gitignore          # Git exclusion rules
└── README.md           # Documentation & results
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Set Up Virtual Environment

```bash
# Clone repository
git clone https://github.com/Sumit2003-dev/Proctor-AI.git
cd Proctor-AI

# Create virtual environment (Python 3.10 or 3.11 recommended)
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Smoke Test

```bash
python test_smoke.py
```

### 4. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. Toggle **"Start Webcam Monitoring"** in the UI to begin real-time proctoring.

---

## ⚙️ Configurable Parameters & Weights

All parameters are centrally managed in `config.py` and can also be tuned interactively via the Streamlit sidebar:

| Parameter | Default | Description |
|---|:---:|---|
| `DEFAULT_CONF_THRESHOLD` | `0.50` | YOLOv8 confidence cutoff for objects and persons. |
| `DEFAULT_YAW_THRESHOLD` | `0.25` | Yaw ratio threshold $[(x_{\text{nose}} - x_{\text{mid}}) / \text{dist}]$ for detecting lookaway. |
| `SUSTAINED_LOOKAWAY_SECONDS` | `2.0 s` | Minimum continuous lookaway duration required before triggering an alert. |
| `WINDOW_SECONDS` | `30.0 s` | Sliding time window over which penalties accumulate. |
| **Penalty: Cell Phone** | `+35` | High severity violation weight. |
| **Penalty: Multiple People** | `+30` | High severity violation weight. |
| **Penalty: No Person** | `+20` | Medium severity violation weight. |
| **Penalty: Sustained Lookaway** | `+15` | Medium severity violation weight. |

---

## 🧠 Fine-Tuning YOLOv8 on Custom Datasets

To train YOLOv8 on custom phone detection or classroom cheating datasets (e.g. from Roboflow Universe):

1. Export your dataset in **YOLOv8 format** and place it in `data/` with `data/data.yaml`.
2. Run fine-tuning:
   ```bash
   python train.py --data data/data.yaml --epochs 50 --imgsz 640 --model yolov8n.pt
   ```
3. Evaluate and compare baseline vs. fine-tuned weights:
   ```bash
   python evaluate.py --data data/data.yaml --baseline yolov8n.pt --finetuned runs/detect/train/weights/best.pt
   ```

---

## 📊 Evaluation Results

> **Note**: Metrics below will be populated after executing fine-tuning on your custom dataset.

| Metric | Baseline (Pretrained YOLOv8n) | Fine-Tuned (Custom Dataset) | Improvement |
|---|:---:|:---:|:---:|
| **Precision** | `[Pending Training]` | `[Pending Training]` | `+0.00%` |
| **Recall** | `[Pending Training]` | `[Pending Training]` | `+0.00%` |
| **mAP@0.50** | `[Pending Training]` | `[Pending Training]` | `+0.00%` |
| **mAP@0.50:0.95** | `[Pending Training]` | `[Pending Training]` | `+0.00%` |

---

## 📄 License
MIT License. Built for educational and academic integrity assessment research.
