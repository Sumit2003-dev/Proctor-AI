"""Evaluation script comparing baseline YOLOv8 vs fine-tuned YOLOv8.

Reports:
- Precision
- Recall
- mAP@50
- mAP@50-95
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from config import DEFAULT_MODEL_NAME


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs fine-tuned YOLOv8 models on test data."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Path to dataset YAML file.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Path to baseline model (e.g. yolov8n.pt).",
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to fine-tuned model (e.g. runs/detect/train/weights/best.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for validation.",
    )
    return parser.parse_args()


def evaluate_model(model_path: str, data_path: str, imgsz: int = 640) -> Optional[Dict[str, float]]:
    """Runs YOLO validation on a given model and returns key metrics.

    Args:
        model_path: Path to model checkpoint file.
        data_path: Path to data.yaml dataset file.
        imgsz: Image evaluation resolution.

    Returns:
        Dictionary of metrics (Precision, Recall, mAP50, mAP50-95) or None on failure.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Warning: Model file not found at: {model_file.resolve()}")
        return None

    try:
        model = YOLO(str(model_file))
        metrics = model.val(data=data_path, imgsz=imgsz, verbose=False)
        return {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
    except Exception as e:
        print(f"Error evaluating model {model_path}: {e}")
        return None


def print_comparison_table(
    baseline_metrics: Optional[Dict[str, float]],
    finetuned_metrics: Optional[Dict[str, float]],
) -> None:
    """Prints a comparison Markdown table of the results."""
    print("\n### Evaluation Results: Baseline vs Fine-Tuned Model\n")
    print("| Metric | Baseline (Pretrained YOLOv8n) | Fine-Tuned (Custom) | Improvement |")
    print("|---|:---:|:---:|:---:|")

    metrics_keys = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("mAP@0.50", "map50"),
        ("mAP@0.50:0.95", "map50_95"),
    ]

    for label, key in metrics_keys:
        b_val = baseline_metrics.get(key) if baseline_metrics else None
        f_val = finetuned_metrics.get(key) if finetuned_metrics else None

        b_str = f"{b_val:.4f}" if b_val is not None else "N/A"
        f_str = f"{f_val:.4f}" if f_val is not None else "N/A"

        if b_val is not None and f_val is not None:
            delta = f_val - b_val
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "-"

        print(f"| {label} | {b_str} | {f_str} | {delta_str} |")
    print("")


def main() -> None:
    args = parse_args()
    data_file = Path(args.data)
    if not data_file.exists():
        print(f"Error: Dataset configuration file not found at: {data_file.resolve()}")
        print("Please place your evaluation dataset in 'data/' with a valid 'data.yaml'.")
        sys.exit(1)

    print(f"Evaluating Baseline Model: {args.baseline}...")
    baseline_results = evaluate_model(args.baseline, args.data, args.imgsz)

    print(f"Evaluating Fine-Tuned Model: {args.finetuned}...")
    finetuned_results = evaluate_model(args.finetuned, args.data, args.imgsz)

    print_comparison_table(baseline_results, finetuned_results)


if __name__ == "__main__":
    main()

