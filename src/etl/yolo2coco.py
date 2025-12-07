import json
import os
from pathlib import Path

import hydra
import pandas as pd
from PIL import Image


def yolo_to_coco(labels_dir, images_dir, output_file, split, categories_list=None):
    # Initialize COCO structure
    data_coco = {"images": [], "type": "instances", "annotations": [], "categories": []}

    annotation_id = 0
    image_id = 0

    category_set = set()

    label_files = []
    for label_file in Path(labels_dir).iterdir():
        if str(label_file.name).endswith(".txt") and (
            str(label_file.name) in split[0].tolist() if split is not None else True
        ):
            label_files.append(label_file.name)

    for label_file in label_files:
        image_filename = os.path.splitext(label_file)[0]
        # Try to find image file with common image extensions
        image_found = False
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_path = os.path.join(images_dir, image_filename + ext)
            if os.path.exists(image_path):
                image_found = True
                break
        if not image_found:
            print(f"Image file for {label_file} not found, skipping.")
            continue

        # Open image to get dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        # Add image info to COCO dataset
        image_info = {
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "id": image_id,
        }
        data_coco["images"].append(image_info)

        # Read label file
        with open(os.path.join(labels_dir, label_file), "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            if len(line.strip()) == 0:
                continue
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

            # Convert normalized coordinates to pixel values
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            x_min = x_center - bbox_width / 2
            y_min = y_center - bbox_height / 2

            # COCO bbox format: [top-left-x, top-left-y, width, height]
            bbox = [x_min, y_min, bbox_width, bbox_height]

            area = bbox_width * bbox_height
            category_id = int(class_id)
            category_set.add(category_id)

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "segmentation": [],
                "iscrowd": 0,
            }
            data_coco["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

    # Create categories list
    if categories_list is not None:
        for idx, category_name in enumerate(categories_list):
            category = {"id": idx, "name": category_name, "supercategory": "none"}
            data_coco["categories"].append(category)
    else:
        for category_id in sorted(category_set):
            category = {"id": category_id, "name": f"class_{category_id}", "supercategory": "none"}
            data_coco["categories"].append(category)

    # Save to JSON file
    with open(output_file, "w") as f_out:
        json.dump(data_coco, f_out, indent=4)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg):
    categories = cfg.train.label_to_name.values()
    dataset_path = Path(cfg.train.data_path)
    labels_path = dataset_path / "labels"
    imgs_path = dataset_path / "images"

    # for csv in ["train", "val"]:
    #     split = pd.read_csv(dataset_path / f"{csv}.csv", header=None)
    #     split[0] = split[0].str.replace(".jpg", ".txt")
    #     output = dataset_path / f"{csv}_annotations.json"

    #     yolo_to_coco(labels_path, imgs_path, output, split, categories)

    yolo_to_coco(labels_path, imgs_path, dataset_path / "annotations.json", None, categories)


if __name__ == "__main__":
    main()
