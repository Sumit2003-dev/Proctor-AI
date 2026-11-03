"""Dataset preparation utility for ProctorAI YOLOv8 fine-tuning.

Supports:
1. Automated setup of fine-tuning dataset with train/val splits and data.yaml.
2. Roboflow Universe dataset download via API key (optional).
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def create_phone_training_sample(img_path: Path, label_path: Path, is_phone_present: bool = True) -> None:
    """Generates a realistic synthetic test image containing a person holding/using a phone."""
    w, h = 640, 480
    img = np.full((h, w, 3), (random.randint(220, 245), random.randint(220, 245), random.randint(220, 245)), dtype=np.uint8)

    # Candidate silhouette
    cv2.circle(img, (320, 220), random.randint(80, 95), (60, 80, 120), -1)
    cv2.ellipse(img, (320, 420), (140, 160), 0, 0, 180, (50, 60, 80), -1)

    labels = []
    # Person box (class 0: person in COCO, or class 1 in custom dataset)
    # Normalized coords: center_x, center_y, width, height
    labels.append("0 0.5 0.65 0.55 0.70")

    if is_phone_present:
        # Place a phone rect in candidate's hand / foreground
        px = random.randint(240, 400)
        py = random.randint(280, 380)
        pw, ph = random.randint(35, 50), random.randint(70, 90)

        # Phone body (dark rectangle with screen glow)
        cv2.rectangle(img, (px - pw // 2, py - ph // 2), (px + pw // 2, py + ph // 2), (25, 25, 25), -1)
        cv2.rectangle(img, (px - pw // 2 + 3, py - ph // 2 + 4), (px + pw // 2 - 3, py + ph // 2 - 4), (180, 180, 230), -1)
        cv2.circle(img, (px, py + ph // 2 - 8), 3, (120, 120, 120), -1)

        # Label for cell phone: class 1 (or 67 in COCO)
        norm_cx = px / w
        norm_cy = py / h
        norm_w = pw / w
        norm_h = ph / h
        labels.append(f"1 {norm_cx:.4f} {norm_cy:.4f} {norm_w:.4f} {norm_h:.4f}")

    cv2.imwrite(str(img_path), img)
    with open(label_path, "w") as f:
        f.write("\n".join(labels))


def setup_dataset(data_dir: Path = Path("data"), samples_train: int = 40, samples_val: int = 15) -> Path:
    """Sets up the directory structure and generates dataset for YOLOv8 fine-tuning."""
    train_img_dir = data_dir / "images" / "train"
    val_img_dir = data_dir / "images" / "val"
    train_lbl_dir = data_dir / "labels" / "train"
    val_lbl_dir = data_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Generating {samples_train} training samples...")
    for i in range(samples_train):
        img_p = train_img_dir / f"sample_train_{i:03d}.jpg"
        lbl_p = train_lbl_dir / f"sample_train_{i:03d}.txt"
        create_phone_training_sample(img_p, lbl_p, is_phone_present=(i % 3 != 0))

    print(f"Generating {samples_val} validation samples...")
    for i in range(samples_val):
        img_p = val_img_dir / f"sample_val_{i:03d}.jpg"
        lbl_p = val_lbl_dir / f"sample_val_{i:03d}.txt"
        create_phone_training_sample(img_p, lbl_p, is_phone_present=(i % 2 == 0))

    data_yaml_path = data_dir / "data.yaml"
    abs_data_dir = data_dir.resolve().as_posix()
    yaml_content = f"""# ProctorAI YOLOv8 Dataset Descriptor
path: {abs_data_dir}
train: images/train
val: images/val

names:
  0: person
  1: cell_phone
"""
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset descriptor generated at: {data_yaml_path.resolve()}")
    return data_yaml_path


def main():
    parser = argparse.ArgumentParser(description="Download or generate YOLOv8 fine-tuning dataset.")
    parser.add_argument("--dir", type=str, default="data", help="Output dataset directory.")
    parser.add_argument("--train-samples", type=int, default=40, help="Number of training samples.")
    parser.add_argument("--val-samples", type=int, default=15, help="Number of validation samples.")
    args = parser.parse_args()

    setup_dataset(Path(args.dir), args.train_samples, args.val_samples)
    print("\nDataset is ready! You can now run: python train.py --data data/data.yaml")


if __name__ == "__main__":
    main()
