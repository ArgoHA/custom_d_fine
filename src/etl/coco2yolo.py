import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def convert_coco_json(json_dir="../coco/annotations/", use_segments=False):
    save_dir = Path(json_dir).parent / "yolo_labels"
    save_dir.mkdir(exist_ok=True)

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / json_file.stem.replace("instances_", "")  # folder name
        fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Write labels file
        for x in tqdm(data["annotations"], desc=f"Annotations {json_file}"):
            if x["iscrowd"]:
                continue

            img = images["%g" % x["image_id"]]
            h, w, f = img["height"], img["width"], img["file_name"]

            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(x["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            # Segments
            if use_segments:
                segments = [j for i in x["segmentation"] for j in i]  # all segments concatenated
                s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

            # Write
            if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                cls = x["category_id"]  # class
                line = cls, *(s if use_segments else box)  # cls, box or segments

                # if there is another level of folders
                f = f.split("/")[-1]

                with open((fn / f).with_suffix(".txt"), "a") as file:
                    file.write(("%g " * len(line)).rstrip() % line + "\n")


if __name__ == "__main__":
    source_path = "path/to/folder/with/*.json"
    convert_coco_json(source_path)  # directory with *.json
