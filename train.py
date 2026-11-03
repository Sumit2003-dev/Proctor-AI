"""Fine-tuning pipeline for YOLOv8 on custom cheating/phone detection datasets.

Usage:
    python train.py --data data/data.yaml --epochs 50 --imgsz 640 --model yolov8n.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from config import DEFAULT_MODEL_NAME


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for exam proctoring / phone detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Path to dataset YAML file (e.g., Roboflow export).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (pixels).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base pretrained YOLOv8 model weights (.pt).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Output project directory for training logs and checkpoints.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run name for this training experiment.",
    )
    return parser.parse_args()


def train(
    data_path: str,
    epochs: int,
    imgsz: int,
    model_name: str,
    batch_size: int = 16,
    project: str = "runs/detect",
    name: str = "train",
) -> Optional[Path]:
    """Executes YOLOv8 fine-tuning.

    Args:
        data_path: Path to dataset YAML descriptor.
        epochs: Number of training epochs.
        imgsz: Image resolution.
        model_name: Base model name or path.
        batch_size: Batch size.
        project: Directory where training runs will be saved.
        name: Name of this specific training run.

    Returns:
        Path to best.pt weights file if training succeeded, else None.
    """
    dataset_file = Path(data_path)
    if not dataset_file.exists():
        print(f"Error: Dataset configuration file not found at: {dataset_file.resolve()}")
        print("Please place your custom dataset in 'data/' with a valid 'data.yaml'.")
        return None

    print(f"--- Starting YOLOv8 Fine-tuning ---")
    print(f"Base Model: {model_name}")
    print(f"Dataset:    {dataset_file.resolve()}")
    print(f"Epochs:     {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch_size}")
    print("-----------------------------------")

    model = YOLO(model_name)
    results = model.train(
        data=str(dataset_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project,
        name=name,
        verbose=True,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nTraining completed successfully!")
        print(f"Best weights saved to: {best_weights.resolve()}")
        return best_weights

    print("Training finished. Check runs/detect directory for artifacts.")
    return None


def main() -> None:
    args = parse_args()
    result = train(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        model_name=args.model,
        batch_size=args.batch,
        project=args.project,
        name=args.name,
    )
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()

