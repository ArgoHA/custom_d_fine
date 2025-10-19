#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def find_contours(binary: np.ndarray) -> List[np.ndarray]:
    # External contours only (ignore holes)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def to_yolo_poly(
    contour: np.ndarray, w: int, h: int, epsilon_ratio: float, n_points_max: int | None
) -> List[Tuple[float, float]]:
    # Simplify contour with Douglas-Peucker
    peri = cv2.arcLength(contour, True)
    epsilon = max(1.0, epsilon_ratio * peri)  # at least 1px to avoid degenerate approximations
    approx = cv2.approxPolyDP(contour, epsilon, True)  # shape (N,1,2)

    pts = approx.reshape(-1, 2)  # (N,2) integer pixel coords (x,y)

    # Optional downsample if still too many points
    if n_points_max is not None and pts.shape[0] > n_points_max:
        idx = np.linspace(0, pts.shape[0] - 1, num=n_points_max, dtype=int)
        pts = pts[idx]

    # Normalize to [0,1] with image width/height
    xs = np.clip(pts[:, 0] / max(1, w), 0.0, 1.0)
    ys = np.clip(pts[:, 1] / max(1, h), 0.0, 1.0)
    return list(zip(xs.tolist(), ys.tolist()))


def mask_to_yolo_lines(
    img: np.ndarray,
    class_id: int,
    thresh_invert: bool,
    min_area_px: int,
    epsilon_ratio: float,
    n_points_max: int | None,
) -> List[str]:
    # Ensure grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu threshold
    flags = cv2.THRESH_BINARY_INV if thresh_invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(img, 0, 255, flags | cv2.THRESH_OTSU)

    contours = find_contours(binary)
    h, w = img.shape[:2]

    lines: List[str] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        poly = to_yolo_poly(cnt, w, h, epsilon_ratio, n_points_max)

        # Need at least 3 points (6 numbers) to form a polygon
        if len(poly) < 3:
            continue

        coords = []
        for x, y in poly:
            coords.extend([f"{x:.6f}", f"{y:.6f}"])
        line = f"{class_id} " + " ".join(coords)
        lines.append(line)

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Convert mask images to YOLO polygon labels (one row per object)."
    )
    parser.add_argument(
        "masks_dir", type=Path, help="Path to the folder containing binary mask images (png/jpg)."
    )
    parser.add_argument(
        "--class-id", type=int, default=0, help="Class id to write as the first value (default: 0)."
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert thresholding if your masks are black objects on white background.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=32,
        help="Minimum contour area in pixels to keep (default: 32).",
    )
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.01,
        help="Douglas-Peucker simplification as fraction of contour perimeter (default: 0.01).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap on number of polygon points per object (e.g., 100).",
    )
    parser.add_argument(
        "--labels-name",
        type=str,
        default="labels",
        help="Output folder name next to masks_dir (default: labels).",
    )
    args = parser.parse_args()

    masks_dir: Path = args.masks_dir
    if not masks_dir.is_dir():
        raise SystemExit(f"Not a folder: {masks_dir}")

    # Create sibling 'labels' folder next to the masks_dir
    root = masks_dir.parent
    out_dir = root / args.labels_name
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    files = sorted([p for p in masks_dir.iterdir() if p.suffix.lower() in exts])

    if not files:
        print(f"No mask images (*.png, *.jpg) found in {masks_dir}")
        return

    n_images = 0
    n_objs = 0

    for img_path in files:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        lines = mask_to_yolo_lines(
            img,
            class_id=args.class_id,
            thresh_invert=args.invert,
            min_area_px=args.min_area,
            epsilon_ratio=args.epsilon_ratio,
            n_points_max=args.max_points,
        )

        out_txt = out_dir / f"{img_path.stem}.txt"
        # Write even if empty (common practice)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        n_images += 1
        n_objs += len(lines)

    print(f"Done. Wrote {n_images} files to {out_dir}")
    print(f"Total objects (rows) written: {n_objs}")


if __name__ == "__main__":
    main()
