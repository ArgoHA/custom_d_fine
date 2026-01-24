import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def convert_coco_json(json_dir="../coco/annotations/", use_segments=False):
    """
    Convert COCO format annotations to YOLO format.

    Args:
        json_dir: Directory containing COCO JSON annotation files
        use_segments: If True, output polygon segmentations; if False, output bounding boxes

    COCO bbox format: [top-left x, top-left y, width, height] (absolute pixels)
    YOLO bbox format: [center x, center y, width, height] (normalized 0-1)

    COCO segmentation: [[x1, y1, x2, y2, ...], ...] (absolute pixels, can have multiple polygons)
    YOLO segmentation: [x1, y1, x2, y2, ...] (normalized 0-1, single polygon per line)
    """
    save_dir = Path(json_dir).parent / "yolo_labels"
    save_dir.mkdir(exist_ok=True)

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / json_file.stem.replace("instances_", "")  # folder name
        fn.mkdir(exist_ok=True)

        with open(json_file) as f:
            data = json.load(f)

        # Build category mapping: COCO category_id -> YOLO class index (0-indexed, contiguous)
        categories = sorted(data.get("categories", []), key=lambda x: x["id"])
        cat_id_to_yolo_cls = {cat["id"]: idx for idx, cat in enumerate(categories)}
        cat_names = [cat["name"] for cat in categories]

        # Write labels.txt with category names
        labels_file = fn / "labels.txt"
        with open(labels_file, "w") as f:
            for name in cat_names:
                f.write(f"{name}\n")
        print(f"Created {labels_file} with {len(cat_names)} classes")

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}

        # Write labels file
        for x in tqdm(data["annotations"], desc=f"Annotations {json_file}"):
            if x.get("iscrowd", 0):
                continue

            img = images.get("%g" % x["image_id"])
            if img is None:
                continue

            h, w, f = img["height"], img["width"], img["file_name"]

            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(x["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            # Map COCO category_id to YOLO class index (0-indexed)
            coco_cat_id = x["category_id"]
            if coco_cat_id not in cat_id_to_yolo_cls:
                print(f"Warning: category_id {coco_cat_id} not found in categories, skipping")
                continue
            cls = cat_id_to_yolo_cls[coco_cat_id]

            # Segments
            if use_segments and "segmentation" in x and x["segmentation"]:
                # Handle each polygon separately (COCO can have multiple disjoint polygons)
                # Write each polygon as a separate line
                segmentation = x["segmentation"]

                # Skip RLE encoded segmentations (dict format)
                if isinstance(segmentation, dict):
                    continue

                for polygon in segmentation:
                    if len(polygon) < 6:  # Need at least 3 points (6 coordinates)
                        continue
                    # Normalize polygon coordinates
                    s = (np.array(polygon).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    line = (cls, *s)

                    # Handle nested folder structure in file_name
                    fname = f.split("/")[-1]

                    with open((fn / fname).with_suffix(".txt"), "a") as file:
                        file.write(("%g " * len(line)).rstrip() % line + "\n")
            else:
                # Write bounding box
                if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                    line = (cls, *box)

                    # Handle nested folder structure in file_name
                    fname = f.split("/")[-1]

                    with open((fn / fname).with_suffix(".txt"), "a") as file:
                        file.write(("%g " * len(line)).rstrip() % line + "\n")


if __name__ == "__main__":
    source_path = "path/to/directory/with/coco.json"
    convert_coco_json(source_path, use_segments=True)  # directory with *.json
