from pathlib import Path
from shutil import copyfile

data_path = "/home/argo/Desktop/Projects/Veryfi/detector/data/dataset_trans"
data_path = Path(data_path)
images_dir = data_path / "images"
labels_dir = data_path / "labels"

img_stems = [x.stem for x in images_dir.iterdir() if not str(x.stem).startswith(".")]
label_stems = [x.stem for x in labels_dir.iterdir() if not str(x.stem).startswith(".")]

if len(img_stems) != len(set(img_stems)):
    print("Duplicate images found")

if len(label_stems) != len(set(label_stems)):
    print("Duplicate labels found")

missing_labels = set(img_stems) - set(label_stems)
if missing_labels:
    remove_empty_images = input(f"Missing labels: {len(missing_labels)}, remove images? (y/n/c): ")
    print(missing_labels)
    if remove_empty_images == "y":
        for img in missing_labels:
            (images_dir / f"{img}.jpg").unlink()
    elif remove_empty_images == "c":
        for img in set(img_stems) - set(missing_labels):
            (data_path / "labeled_copy").mkdir(exist_ok=True, parents=True)
            copyfile(images_dir / f"{img}.jpg", data_path / "labeled_copy" / f"{img}.jpg")
        for img in missing_labels:
            (data_path / "no_label_copy").mkdir(exist_ok=True, parents=True)
            copyfile(images_dir / f"{img}.jpg", data_path / "no_label_copy" / f"{img}.jpg")


missing_images = set(label_stems) - set(img_stems)
if missing_images:
    print("Removing labels with missing images:", missing_images)
    for label in missing_images:
        (labels_dir / f"{label}.txt").unlink()
