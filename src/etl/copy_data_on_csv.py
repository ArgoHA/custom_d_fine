from pathlib import Path
from shutil import copyfile

import pandas as pd

dataset_path = Path("/home/argo/Desktop/Projects/Veryfi/detector/data/dataset_trans")
imgs_path = dataset_path / "images"
labels_path = dataset_path / "labels"
train_csv = dataset_path / "train.csv"
val_csv = dataset_path / "val.csv"

train_path = dataset_path / "train"
val_path = dataset_path / "val"

train_imgs_path = train_path / "images"
train_labels_path = train_path / "labels"

val_imgs_path = val_path / "images"
val_labels_path = val_path / "labels"

train_imgs_path.mkdir(exist_ok=True, parents=True)
val_imgs_path.mkdir(exist_ok=True, parents=True)

train_labels_path.mkdir(exist_ok=True, parents=True)
val_labels_path.mkdir(exist_ok=True, parents=True)


train_df = pd.read_csv(train_csv, header=None)
val_df = pd.read_csv(val_csv, header=None)


for index, row in train_df.iterrows():
    img_name = row.values.tolist()[0]
    label_name = row.values.tolist()[0].replace(".jpg", ".txt")
    copyfile(imgs_path / img_name, train_imgs_path / img_name)
    if Path(labels_path / label_name).exists():
        copyfile(labels_path / label_name, train_labels_path / label_name)


for index, row in val_df.iterrows():
    img_name = row.values.tolist()[0]
    label_name = row.values.tolist()[0].replace(".jpg", ".txt")
    copyfile(imgs_path / img_name, val_imgs_path / img_name)
    if Path(labels_path / label_name).exists():
        copyfile(labels_path / label_name, val_labels_path / label_name)
