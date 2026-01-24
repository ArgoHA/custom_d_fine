"""
Convert YOLO polygon format labels to YOLO bounding box format.

YOLO Polygon format: class_id x1 y1 x2 y2 x3 y3 ... xn yn
YOLO BBox format: class_id x_center y_center width height

All coordinates are normalized (0-1).
"""

import argparse
from pathlib import Path

from tqdm import tqdm


def polygon_to_bbox(polygon_coords: list[float]) -> tuple[float, float, float, float]:
    """
    Convert polygon coordinates to bounding box.

    Args:
        polygon_coords: List of [x1, y1, x2, y2, ..., xn, yn] normalized coordinates

    Returns:
        Tuple of (x_center, y_center, width, height) normalized coordinates
    """
    # Extract x and y coordinates
    x_coords = polygon_coords[0::2]  # Every even index (0, 2, 4, ...)
    y_coords = polygon_coords[1::2]  # Every odd index (1, 3, 5, ...)

    # Get bounding box corners
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Convert to YOLO format (center x, center y, width, height)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def convert_label_file(input_path: Path, output_path: Path) -> int:
    """
    Convert a single label file from polygon to bbox format.

    Args:
        input_path: Path to input polygon label file
        output_path: Path to output bbox label file

    Returns:
        Number of annotations converted
    """
    annotations = []

    with open(input_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:  # Need at least class_id + 2 points (4 coords)
            continue

        class_id = parts[0]
        coords = [float(x) for x in parts[1:]]

        # Check if it's already a bbox (exactly 4 values)
        if len(coords) == 4:
            # Already in bbox format, keep as is
            x_center, y_center, width, height = coords
        else:
            # Convert polygon to bbox
            if len(coords) < 4 or len(coords) % 2 != 0:
                print(f"Warning: Invalid polygon in {input_path}: {line}")
                continue
            x_center, y_center, width, height = polygon_to_bbox(coords)

        # Clamp values to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Write output file
    with open(output_path, "w") as f:
        f.write("\n".join(annotations))
        if annotations:
            f.write("\n")

    return len(annotations)


def convert_labels_folder(
    input_folder: str | Path, output_folder: str | Path | None = None, suffix: str = "_det"
) -> None:
    """
    Convert all label files in a folder from polygon to bbox format.

    Args:
        input_folder: Path to folder containing polygon label files
        output_folder: Path to output folder (default: input_folder + suffix)
        suffix: Suffix to add to input folder name if output_folder not specified
    """
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    if output_folder is None:
        output_path = input_path.parent / f"{input_path.name}{suffix}"
    else:
        output_path = Path(output_folder)

    output_path.mkdir(parents=True, exist_ok=True)

    # Find all txt files
    txt_files = list(input_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {input_path}")
        return

    print(f"Converting {len(txt_files)} label files...")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    total_annotations = 0

    for txt_file in tqdm(txt_files, desc="Converting"):
        output_file = output_path / txt_file.name
        num_annotations = convert_label_file(txt_file, output_file)
        total_annotations += num_annotations

    print(f"\nDone! Converted {total_annotations} annotations in {len(txt_files)} files.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO polygon labels to YOLO bounding box format"
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to folder containing YOLO polygon label files (.txt)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output folder path (default: input_folder_det)",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        default="_det",
        help="Suffix to add to input folder name for output (default: _det)",
    )

    args = parser.parse_args()

    convert_labels_folder(
        input_folder=args.input_folder, output_folder=args.output, suffix=args.suffix
    )


if __name__ == "__main__":
    main()
