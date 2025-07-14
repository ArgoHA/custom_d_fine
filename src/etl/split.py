from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def split(
    data_path: Path,
    train_split: float,
    val_split: float,
    images_path: Path,
    ignore_negatives: bool,
    seed: int,
    shuffle: True,
) -> None:
    test_split = 1 - train_split - val_split
    if test_split <= 0.001:
        test_split = 0.0

    img_paths = [x.name for x in images_path.iterdir() 
                 if not str(x.name).startswith(".") 
                    and x.suffix.lower() == ".jpg"]

    if not shuffle:
        img_paths.sort()

    if ignore_negatives:
        for img_path in img_paths:
            if not (images_path.parent / "labels" / f"{Path(img_path).stem}.txt").exists():
                img_paths.remove(img_path)

    indices = np.arange(len(img_paths))

    if train_split >= 1.0:
        np.random.seed(seed)
        np.random.shuffle(indices)
        train_idxs = indices
        val_idxs = []
    else:
        train_idxs, temp_idxs = train_test_split(
            indices, test_size=(1 - train_split), random_state=seed, shuffle=shuffle
        )

        if test_split:
            test_idxs, val_idxs = train_test_split(
                temp_idxs,
                test_size=(val_split / (val_split + test_split)),
                random_state=seed,
                shuffle=shuffle,
            )
        else:
            val_idxs = temp_idxs
            test_idxs = []

    splits = {"train": train_idxs, "val": val_idxs}
    if test_split:
        splits["test"] = test_idxs

    for split_name, split in splits.items():
        csv_path = data_path / f"{split_name}.csv"
        if csv_path.exists():
            csv_path.unlink()
            logger.info(f"{csv_path} already exists. Removing...")
        if len(split) > 0:
            with open(csv_path, "w") as f:
                for idx in split:
                    f.write(str(img_paths[idx]) + "\n")
                logger.info(f"{split_name}: {len(split)} were saved to {csv_path}")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    data_path = Path(cfg.train.data_path)

    split(
        data_path,
        cfg.split.train_split,
        cfg.split.val_split,
        data_path / "images",
        cfg.split.ignore_negatives,
        cfg.train.seed,
        shuffle=cfg.split.shuffle,
    )


if __name__ == "__main__":
    main()
