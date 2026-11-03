"""Fine-tune YOLOv8 using real-world datasets curated via FiftyOne.

Curates real images containing real people and cell phones from FiftyOne Zoo
(e.g., COCO-2017 or Open Images), exports them into YOLOv8 format, and trains
a production-grade detector that generalizes to real webcam environments.

Usage:
    python train_fiftyone.py --max-samples 100 --epochs 10 --imgsz 640
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["FIFTYONE_DISABLE_SERVICES"] = "true"

import argparse
import sys
from pathlib import Path
import shutil

import torch
torch.set_num_threads(1)
from ultralytics import YOLO

import fiftyone as fo
import fiftyone.zoo as foz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate real-world dataset via FiftyOne and fine-tune YOLOv8."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco-2017",
        help="FiftyOne zoo dataset to download (e.g. coco-2017, open-images-v7).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to pull samples from.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["person", "cell phone"],
        help="Classes to filter and annotate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=80,
        help="Number of real-world samples to download and curate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base YOLO model weights.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="data/fiftyone_coco",
        help="Directory to export the YOLO formatted dataset.",
    )
    return parser.parse_args()


def curate_and_export(
    dataset_name: str,
    split: str,
    classes: list,
    max_samples: int,
    export_dir: Path,
) -> Path:
    """Downloads real-world images from FiftyOne Zoo and exports to YOLO format."""
    print(f"\n[1/3] Curating real-world dataset '{dataset_name}' via FiftyOne Zoo...")
    print(f"      Target Classes: {classes}")
    print(f"      Target Samples: {max_samples}")

    dataset = foz.load_zoo_dataset(
        dataset_name,
        split=split,
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples,
        shuffle=True,
    )

    print(f"      Loaded {len(dataset)} real-world samples from FiftyOne.")

    # Clean prior export if present
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/3] Exporting to YOLO format in: {export_dir.resolve()}...")
    dataset.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=classes,
    )

    yaml_file = export_dir / "dataset.yaml"
    if not yaml_file.exists():
        # Fallback check for standard data.yaml
        alt_yaml = export_dir / "data.yaml"
        if alt_yaml.exists():
            yaml_file = alt_yaml

    print(f"      YOLO dataset descriptor generated: {yaml_file.resolve()}")
    return yaml_file


def train_yolo(
    yaml_file: Path,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
) -> Path:
    """Trains YOLOv8 on the FiftyOne curated real-world dataset."""
    print(f"\n[3/3] Fine-tuning YOLOv8 ({base_model}) on curated real-world data...")
    model = YOLO(base_model)
    results = model.train(
        data=str(yaml_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project="runs/detect",
        name="fiftyone_train",
        verbose=True,
    )

    best_weights = Path("runs/detect/fiftyone_train/weights/best.pt")
    if not best_weights.exists():
        # Check subfolder nesting if any
        nested = list(Path("runs/detect").glob("**/fiftyone_train/weights/best.pt"))
        if nested:
            best_weights = nested[0]

    if best_weights.exists():
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        target_path = models_dir / "best.pt"
        shutil.copy(best_weights, target_path)
        print(f"\nTraining completed! Fine-tuned weights updated at: {target_path.resolve()}")
        return target_path

    print("Training finished. Check runs/detect directory for weights.")
    return best_weights


def main():
    args = parse_args()
    export_dir = Path(args.export_dir)

    yaml_file = curate_and_export(
        dataset_name=args.dataset,
        split=args.split,
        classes=args.classes,
        max_samples=args.max_samples,
        export_dir=export_dir,
    )

    best_pt = train_yolo(
        yaml_file=yaml_file,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
    )

    print("\nNext step: Run evaluate.py to verify real-world mAP and precision:")
    print(f"python evaluate.py --data {yaml_file.as_posix()} --baseline yolov8n.pt --finetuned {best_pt.as_posix()}")


if __name__ == "__main__":
    main()

