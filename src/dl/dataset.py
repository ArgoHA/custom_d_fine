import random
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.dl.utils import (
    LetterboxRect,
    abs_xyxy_to_norm_xywh,
    get_mosaic_coordinate,
    norm_xywh_to_abs_xyxy,
    random_affine,
    seed_worker,
    vis_one_box,
)


class CustomDataset(Dataset):
    def __init__(
        self,
        img_size: Tuple[int, int],  # h, w
        root_path: Path,
        split: pd.DataFrame,
        debug_img_processing: bool,
        mode: str,
        cfg: DictConfig,
    ) -> None:
        self.project_path = Path(cfg.train.root)
        self.root_path = root_path
        self.split = split
        self.target_h, self.target_w = img_size
        self.norm = ([0, 0, 0], [1, 1, 1])
        self.debug_img_processing = debug_img_processing
        self.mode = mode
        self.ignore_background = False
        self.label_to_name = cfg.train.label_to_name

        self.mosaic_prob = cfg.train.mosaic_augs.mosaic_prob
        self.mosaic_scale = cfg.train.mosaic_augs.mosaic_scale
        self.degrees = cfg.train.mosaic_augs.degrees
        self.translate = cfg.train.mosaic_augs.translate
        self.shear = cfg.train.mosaic_augs.shear
        self.keep_ratio = cfg.train.keep_ratio
        self.use_one_class = cfg.train.use_one_class
        self.cases_to_debug = 20

        self._init_augs(cfg)

        self.debug_img_path = Path(cfg.train.debug_img_path)

    def _init_augs(self, cfg) -> None:
        if self.keep_ratio:
            scaleup = False
            if self.mode == "train":
                scaleup = True

            resize = [
                LetterboxRect(
                    height=self.target_h,
                    width=self.target_w,
                    color=(114, 114, 114),
                    scaleup=scaleup,
                    always_apply=True,
                )
            ]
        else:
            resize = [A.Resize(self.target_h, self.target_w, interpolation=cv2.INTER_AREA)]

        norm = [
            A.Normalize(mean=self.norm[0], std=self.norm[1]),
            ToTensorV2(),
        ]

        if self.mode == "train":
            augs = [
                A.CoarseDropout(
                    num_holes_range=(1, 2),
                    hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15),
                    p=cfg.train.augs.coarse_dropout,
                ),
                A.RandomBrightnessContrast(p=cfg.train.augs.brightness),
                A.RandomGamma(p=cfg.train.augs.gamma),
                A.Blur(p=cfg.train.augs.blur),
                A.GaussNoise(p=cfg.train.augs.noise, std_range=(0.1, 0.2)),
                A.ToGray(p=cfg.train.augs.to_gray),
                A.Affine(
                    rotate=[90, 90],
                    p=cfg.train.augs.rotate_90,
                    fit_output=True,
                ),
                A.HorizontalFlip(p=cfg.train.augs.left_right_flip),
                A.VerticalFlip(p=cfg.train.augs.up_down_flip),
                A.Rotate(
                    limit=cfg.train.augs.rotation_degree,
                    p=cfg.train.augs.rotation_p,
                    interpolation=cv2.INTER_AREA,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(114, 114, 114),
                ),
            ]

            self.transform = A.Compose(
                augs + resize + norm,
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            )
        elif self.mode in ["val", "test", "bench"]:
            self.mosaic_prob = 0
            self.transform = A.Compose(
                resize + norm,
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            )
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}, choose from ['train', 'val', 'test', 'bench']"
            )

        self.mosaic_transform = A.Compose(norm)

    def _debug_image(
        self, idx, image: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor, img_path: Path
    ) -> None:
        # Unnormalize the image
        mean = np.array(self.norm[0]).reshape(-1, 1, 1)
        std = np.array(self.norm[1]).reshape(-1, 1, 1)
        image_np = image.cpu().numpy()
        image_np = (image_np * std) + mean

        # Convert from [C, H, W] to [H, W, C]
        image_np = np.transpose(image_np, (1, 2, 0))

        # Convert pixel values from [0, 1] to [0, 255]
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)

        # Draw bounding boxes and class IDs
        boxes_np = boxes.cpu().numpy().astype(int)
        classes_np = classes.cpu().numpy()
        for box, class_id in zip(boxes_np, classes_np):
            vis_one_box(image_np, box, class_id, mode="gt", label_to_name=self.label_to_name)

        # Save the image
        save_dir = self.debug_img_path / self.mode
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{idx}_idx_{img_path.stem}_debug.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    def _get_data(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns np.ndarray RGB image; targets as np.ndarray [[class_id, x1, y1, x2, y2]]
        """
        # Get image
        image_path = Path(self.split.iloc[idx].values[0])
        image = cv2.imread(str(self.root_path / "images" / f"{image_path}"))  # BGR, HWC
        assert image is not None, f"Image wasn't loaded: {image_path}"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB, HWC
        height, width, _ = image.shape
        orig_size = torch.tensor([height, width])

        # Get labels
        labels_path = self.root_path / "labels" / f"{image_path.stem}.txt"
        if labels_path.exists() and labels_path.stat().st_size > 1:
            targets = np.loadtxt(labels_path)
            if targets.ndim == 1:  # Handle the case with only one object
                targets = targets.reshape(1, -1)

            if self.use_one_class:
                targets[:, 0] = 0

            targets[:, 1:] = norm_xywh_to_abs_xyxy(targets[:, 1:], height, width).astype(np.float32)
            return image, targets, orig_size
        targets = np.zeros((1, 5), dtype=np.float32)
        return image, targets, orig_size

    def _load_mosaic(self, idx):
        mosaic_targets = []
        yc = int(random.uniform(self.target_h * 0.6, self.target_h * 1.4))
        xc = int(random.uniform(self.target_w * 0.6, self.target_w * 1.4))
        indices = [idx] + [random.randint(0, self.__len__() - 1) for _ in range(3)]

        for i_mosaic, m_idx in enumerate(indices):
            img, targets, _ = self._get_data(m_idx)
            (h, w, c) = img.shape[:3]

            if self.keep_ratio:
                scale_h = min(1.0 * self.target_h / h, 1.0 * self.target_w / w)
                scale_w = scale_h
            else:
                scale_h, scale_w = (1.0 * self.target_h / h, 1.0 * self.target_w / w)

            img = cv2.resize(
                img, (int(w * scale_w), int(h * scale_h)), interpolation=cv2.INTER_AREA
            )
            (h, w, c) = img.shape[:3]

            if i_mosaic == 0:
                mosaic_img = np.full((self.target_h * 2, self.target_w * 2, c), 114, dtype=np.uint8)

            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, self.target_h, self.target_w
            )

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            targets = targets.copy()
            # Normalized xywh to pixel xyxy format
            if targets.size > 0:
                targets[:, 1] = scale_w * targets[:, 1] + padw
                targets[:, 2] = scale_h * targets[:, 2] + padh
                targets[:, 3] = scale_w * targets[:, 3] + padw
                targets[:, 4] = scale_h * targets[:, 4] + padh
            mosaic_targets.append(targets)

        if len(mosaic_targets):
            mosaic_targets = np.concatenate(mosaic_targets, 0)
            np.clip(mosaic_targets[:, 1], 0, 2 * self.target_w, out=mosaic_targets[:, 1])
            np.clip(mosaic_targets[:, 2], 0, 2 * self.target_h, out=mosaic_targets[:, 2])
            np.clip(mosaic_targets[:, 3], 0, 2 * self.target_w, out=mosaic_targets[:, 3])
            np.clip(mosaic_targets[:, 4], 0, 2 * self.target_h, out=mosaic_targets[:, 4])

        mosaic_img, mosaic_targets = random_affine(
            mosaic_img,
            mosaic_targets,
            segments=[],
            target_size=(self.target_w, self.target_h),
            degrees=self.degrees,
            translate=self.translate,
            scales=self.mosaic_scale,
            shear=self.shear,
        )

        # this should be in processing
        box_heights = mosaic_targets[:, 3] - mosaic_targets[:, 1]
        box_widths = mosaic_targets[:, 4] - mosaic_targets[:, 2]
        mosaic_targets = mosaic_targets[np.minimum(box_heights, box_widths) > 1]

        image = self.mosaic_transform(image=mosaic_img)["image"]
        labels = torch.tensor(mosaic_targets[:, 0], dtype=torch.int64)
        boxes = torch.tensor(mosaic_targets[:, 1:], dtype=torch.float32)
        return image, labels, boxes, (self.target_h, self.target_w)

    def close_mosaic(self):
        self.mosaic_prob = 0.0
        logger.info("Closing mosaic")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = Path(self.split.iloc[idx].values[0])
        if random.random() < self.mosaic_prob:
            image, labels, boxes, orig_size = self._load_mosaic(idx)
        else:
            image, targets, orig_size = self._get_data(idx)  # boxes in abs xyxy format

            if self.ignore_background and np.all(targets == 0) and self.mode == "train":
                return None

            box_heights = targets[:, 3] - targets[:, 1]
            box_widths = targets[:, 4] - targets[:, 2]
            targets = targets[np.minimum(box_heights, box_widths) > 0]

            # Apply transformations
            transformed = self.transform(
                image=image, bboxes=targets[:, 1:], class_labels=targets[:, 0]
            )
            image = transformed["image"]  # RGB, CHW
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)

        if self.debug_img_processing and idx <= self.cases_to_debug:
            self._debug_image(idx, image, boxes, labels, image_path)

        # return back to normalized format for model
        boxes = torch.tensor(
            abs_xyxy_to_norm_xywh(boxes, image.shape[1], image.shape[2]), dtype=torch.float32
        )
        return image, labels, boxes, image_path, orig_size

    def __len__(self):
        return len(self.split)


class Loader:
    def __init__(
        self,
        root_path: Path,
        img_size: Tuple[int, int],
        batch_size: int,
        num_workers: int,
        cfg: DictConfig,
        debug_img_processing: bool = False,
    ) -> None:
        self.root_path = root_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cfg = cfg
        self.use_one_class = cfg.train.use_one_class
        self.debug_img_processing = debug_img_processing
        self._get_splits()
        self.class_names = list(cfg.train.label_to_name.values())
        self.multiscale_prob = cfg.train.augs.multiscale_prob

    def _get_splits(self) -> None:
        self.splits = {"train": None, "val": None, "test": None}
        for split_name in self.splits.keys():
            if (self.root_path / f"{split_name}.csv").exists():
                self.splits[split_name] = pd.read_csv(
                    self.root_path / f"{split_name}.csv", header=None
                )
            else:
                self.splits[split_name] = []

    def _get_label_stats(self) -> Dict:
        if self.use_one_class:
            classes = {"target": 0}
        else:
            classes = {class_name: 0 for class_name in self.class_names}
        for split in self.splits.values():
            if not np.any(split):
                continue
            for image_path in split.iloc[:, 0]:
                labels_path = self.root_path / "labels" / f"{Path(image_path).stem}.txt"
                if not (labels_path.exists() and labels_path.stat().st_size > 1):
                    continue
                targets = np.loadtxt(labels_path)
                if targets.ndim == 1:
                    targets = targets.reshape(1, -1)
                labels = targets[:, 0]
                for class_id in labels:
                    if self.use_one_class:
                        classes["target"] += 1
                    else:
                        classes[self.class_names[int(class_id)]] += 1
        return classes

    def _get_amount_of_background(self):
        labels = set()
        for label_path in (self.root_path / "labels").iterdir():
            if not label_path.stat().st_size:
                label_path.unlink()  # remove empty txt files
            elif not (label_path.stem.startswith(".") and label_path.name == "labels.txt"):
                labels.add(label_path.stem)

        raw_split_images = set()
        for split in self.splits.values():
            if np.any(split):
                raw_split_images.update(split.iloc[:, 0].values)

        split_images = []
        for split_image in raw_split_images:
            split_images.append(Path(split_image).stem)

        images = {
            f.stem for f in (self.root_path / "images").iterdir() if not f.stem.startswith(".")
        }
        images = images.intersection(split_images)
        return len(images - labels)

    def _build_dataloader_impl(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        collate_fn = self.val_collate_fn
        if dataset.mode == "train":
            collate_fn = self.train_collate_fn

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            prefetch_factor=4,
            pin_memory=True,
        )
        return dataloader

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = CustomDataset(
            self.img_size,
            self.root_path,
            self.splits["train"],
            self.debug_img_processing,
            mode="train",
            cfg=self.cfg,
        )
        val_ds = CustomDataset(
            self.img_size,
            self.root_path,
            self.splits["val"],
            self.debug_img_processing,
            mode="val",
            cfg=self.cfg,
        )

        train_loader = self._build_dataloader_impl(train_ds, shuffle=True)
        val_loader = self._build_dataloader_impl(val_ds)

        test_loader = None
        test_ds = []
        if len(self.splits["test"]):
            test_ds = CustomDataset(
                self.img_size,
                self.root_path,
                self.splits["test"],
                self.debug_img_processing,
                mode="test",
                cfg=self.cfg,
            )
            test_loader = self._build_dataloader_impl(test_ds)

        logger.info(f"Images in train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
        obj_stats = self._get_label_stats()
        logger.info(
            f"Objects count: {', '.join(f'{key}: {value}' for key, value in obj_stats.items())}"
        )
        logger.info(f"Background images: {self._get_amount_of_background()}")
        return train_loader, val_loader, test_loader

    def _collate_fn(self, batch) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Input: List[Tuple[Tensor[channel, height, width], Tensor[labels], Tensor[boxes]], ...]
        where each tuple is a an item in a batch...]
        """
        if None in batch:
            return None, None, None
        images = []
        targets = []
        img_paths = []
        orig_sizes = []

        for item in batch:
            target_dict = {"boxes": item[2], "labels": item[1], "orig_size": item[4]}
            images.append(item[0])
            targets.append(target_dict)
            img_paths.append(item[3])
            orig_sizes.append(item[4])

        images = torch.stack(images, dim=0)
        return images, targets, img_paths

    def val_collate_fn(self, batch) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        return self._collate_fn(batch)

    def train_collate_fn(
        self, batch
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        During traing add multiscale augmentation to the batch
        """
        images, targets, img_paths = self._collate_fn(batch)

        if random.random() < self.multiscale_prob:
            offset = random.choice([-2, -1, 1, 2]) * 32
            new_h = images.shape[2] + offset
            new_w = images.shape[3] + offset

            # boxes are normalized, so only image should be resized
            images = torch.nn.functional.interpolate(
                images, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        return images, targets, img_paths
