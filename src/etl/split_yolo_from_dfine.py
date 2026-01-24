#!/usr/bin/env python3
"""
Convert a dataset with images/, labels/ folders and CSV split files to YOLO format.

Input structure:
    dataset_path/
        images/
            batch_1_000003.jpg
            ...
        labels/
            batch_1_000003.txt
            labels.txt  (class names, one per line)
            ...
        train.csv
        val.csv
        test.csv (optional)

Output structure:
    output_path/
        images/
            train/
            val/
            test/ (if test.csv exists)
        labels/
            train/
            val/
            test/ (if test.csv exists)
        dataset.yaml
"""

import argparse
import shutil
from pathlib import Path

import yaml


def read_csv_filenames(csv_path: Path) -> list[str]:
    """Read image filenames from a CSV file (no header, one filename per line)."""
    with open(csv_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]
    return filenames


def read_class_names(labels_txt_path: Path) -> list[str]:
    """Read class names from labels.txt file."""
    with open(labels_txt_path, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    return class_names


def get_label_filename(image_filename: str) -> str:
    """Convert image filename to corresponding label filename."""
    stem = Path(image_filename).stem
    return f"{stem}.txt"


def copy_files_for_split(
    image_filenames: list[str],
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    split_name: str,
) -> tuple[int, int]:
    """Copy image and label files for a specific split."""
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    images_copied = 0
    labels_copied = 0

    for img_filename in image_filenames:
        # Copy image
        src_img = src_images_dir / img_filename
        dst_img = dst_images_dir / img_filename

        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            images_copied += 1
        else:
            print(f"Warning: Image not found: {src_img}")

        # Copy label
        label_filename = get_label_filename(img_filename)
        src_label = src_labels_dir / label_filename
        dst_label = dst_labels_dir / label_filename

        if src_label.exists():
            shutil.copy2(src_label, dst_label)
            labels_copied += 1
        else:
            print(f"Warning: Label not found: {src_label}")

    print(f"  {split_name}: copied {images_copied} images, {labels_copied} labels")
    return images_copied, labels_copied


def create_dataset_yaml(
    output_path: Path,
    class_names: list[str],
    has_test: bool = False,
) -> None:
    """Create dataset.yaml file for YOLO training."""
    dataset_config = {
        "path": str(output_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    if has_test:
        dataset_config["test"] = "images/test"

    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    print(f"Created dataset.yaml with {len(class_names)} classes")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset with CSV splits to YOLO format")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the source dataset with images/, labels/, and CSV files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for YOLO dataset (default: dataset_path + '_yolo')",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="train.csv",
        help="Name of training CSV file (default: train.csv)",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="val.csv",
        help="Name of validation CSV file (default: val.csv)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="test.csv",
        help="Name of test CSV file (default: test.csv)",
    )
    parser.add_argument(
        "--labels-txt",
        type=str,
        default="labels.txt",
        help="Name of class names file in labels folder (default: labels.txt)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output) if args.output else Path(f"{dataset_path}_yolo")

    # Validate input paths
    src_images_dir = dataset_path / "images"
    src_labels_dir = dataset_path / "labels"

    if not src_images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {src_images_dir}")
    if not src_labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {src_labels_dir}")

    # Read class names
    labels_txt_path = src_labels_dir / args.labels_txt
    if not labels_txt_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_txt_path}")
    class_names = read_class_names(labels_txt_path)
    print(f"Found {len(class_names)} classes: {class_names}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Process each split
    splits_processed = []

    # Train split (required)
    train_csv_path = dataset_path / args.train_csv
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")

    train_filenames = read_csv_filenames(train_csv_path)
    print(f"\nProcessing train split ({len(train_filenames)} files)...")
    copy_files_for_split(
        train_filenames,
        src_images_dir,
        src_labels_dir,
        output_path / "images" / "train",
        output_path / "labels" / "train",
        "train",
    )
    splits_processed.append("train")

    # Val split (required)
    val_csv_path = dataset_path / args.val_csv
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv_path}")

    val_filenames = read_csv_filenames(val_csv_path)
    print(f"\nProcessing val split ({len(val_filenames)} files)...")
    copy_files_for_split(
        val_filenames,
        src_images_dir,
        src_labels_dir,
        output_path / "images" / "val",
        output_path / "labels" / "val",
        "val",
    )
    splits_processed.append("val")

    # Test split (optional)
    test_csv_path = dataset_path / args.test_csv
    has_test = test_csv_path.exists()
    if has_test:
        test_filenames = read_csv_filenames(test_csv_path)
        print(f"\nProcessing test split ({len(test_filenames)} files)...")
        copy_files_for_split(
            test_filenames,
            src_images_dir,
            src_labels_dir,
            output_path / "images" / "test",
            output_path / "labels" / "test",
            "test",
        )
        splits_processed.append("test")
    else:
        print(f"\nNo test CSV found at {test_csv_path}, skipping test split")

    # Create dataset.yaml
    print()
    create_dataset_yaml(output_path, class_names, has_test=has_test)

    print(f"\nDone! YOLO dataset created at: {output_path}")
    print(f"Splits processed: {', '.join(splits_processed)}")


if __name__ == "__main__":
    main()
