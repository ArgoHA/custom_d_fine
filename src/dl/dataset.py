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
from torch.utils.data.distributed import DistributedSampler

from src.d_fine.dist_utils import is_main_process
from src.dl.utils import (
    LetterboxRect,
    abs_xyxy_to_norm_xywh,
    clip_polygon_to_rect,
    get_mosaic_coordinate,
    norm_poly_to_abs,
    norm_xywh_to_abs_xyxy,
    poly_abs_to_mask,
    random_affine,
    seed_worker,
    vis_one_box,
)


def parse_yolo_label_file(path: Path):
    """
    Supports both pure detection lines (5 cols) and YOLO-Seg lines (>=7 cols).
    Returns:
      boxes_norm: np.ndarray (N,5) -> [cls, xc, yc, w, h] in norm (float32)
      polys_norm: list[np.ndarray] -> each (K,2) normalized polygon (float32) or [] if none
    """
    boxes_norm = []
    polys_norm = []  # keep normalized here

    with open(path, "r") as f:
        for ln, raw in enumerate(f, 1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            cl = float(parts[0])

            nums = [float(x) for x in parts[1:]]  # variable length
            if len(nums) == 4:  # bbox annotations
                boxes_norm.append([cl, *nums[:4]])
                polys_norm.append(np.empty((0, 2), dtype=np.float32))  # no polygon
            elif len(nums) >= 6:  # segmentation annotations
                if len(nums) % 2 == 1:
                    nums = nums[:-1]
                    logger.warning(
                        f"Odd number of coordinates in segmentation annotation at {path}:{ln}: {s}. "
                        "Dropping the last value."
                    )
                poly = np.array(nums).reshape(-1, 2)  # (K, 2)
                polys_norm.append(poly)
                x_min, y_min = poly.min(axis=0)
                x_max, y_max = poly.max(axis=0)
                boxes_norm.append(
                    [cl, (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]
                )
            else:
                raise ValueError(f"Invalid label line (wrong number of values) {path}:{ln}: {s}")

    if len(boxes_norm) == 0:
        return np.zeros((0, 5), dtype=np.float32), []
    boxes_norm = np.asarray(boxes_norm, dtype=np.float32)
    return boxes_norm, polys_norm


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
        self.return_masks = str(cfg.task).lower() == "segment"

        self.mosaic_prob = cfg.train.mosaic_augs.mosaic_prob
        self.mosaic_scale = cfg.train.mosaic_augs.mosaic_scale
        self.degrees = cfg.train.mosaic_augs.degrees
        self.translate = cfg.train.mosaic_augs.translate
        self.shear = cfg.train.mosaic_augs.shear
        self.keep_ratio = cfg.train.keep_ratio
        self.use_one_class = cfg.train.use_one_class
        self.cases_to_debug = 100

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
        self,
        idx,
        image: torch.Tensor,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        img_path: Path,
        masks=None,
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

        if masks is not None and masks.numel() > 0:
            mnp = masks.cpu().numpy()
            for k in range(mnp.shape[0]):
                cnts, _ = cv2.findContours(
                    mnp[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(image_np, cnts, -1, (0, 255, 0), 2)

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
        targets = np.zeros((0, 5), dtype=np.float32)
        polys_abs = []  # list[(K, 2)] normalized; may be []

        if labels_path.exists() and labels_path.stat().st_size > 1:
            boxes_norm, polys_norm = parse_yolo_label_file(labels_path)

            if boxes_norm.shape[0] and self.use_one_class:
                boxes_norm[:, 0] = 0

            xyxy_abs = norm_xywh_to_abs_xyxy(boxes_norm[:, 1:5], height, width).astype(np.float32)
            targets = np.concatenate([boxes_norm[:, [0]], xyxy_abs], axis=1)  # [N,5]
            polys_abs = [norm_poly_to_abs(p, height, width) for p in polys_norm]
        return image, targets, orig_size, polys_abs

    def _load_mosaic(self, idx):
        mosaic_targets = []
        mosaic_segments = []
        yc = int(random.uniform(self.target_h * 0.6, self.target_h * 1.4))
        xc = int(random.uniform(self.target_w * 0.6, self.target_w * 1.4))
        indices = [idx] + [random.randint(0, self.__len__() - 1) for _ in range(3)]

        mosaic_img = None
        for i_mosaic, m_idx in enumerate(indices):
            img, targets, _, polys_abs = self._get_data(m_idx)
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

            if mosaic_img is None:
                mosaic_img = np.full((self.target_h * 2, self.target_w * 2, c), 114, dtype=np.uint8)

            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, self.target_h, self.target_w
            )

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            if targets.size > 0:
                targets = targets.copy()
                targets[:, 1] = scale_w * targets[:, 1] + padw
                targets[:, 2] = scale_h * targets[:, 2] + padh
                targets[:, 3] = scale_w * targets[:, 3] + padw
                targets[:, 4] = scale_h * targets[:, 4] + padh
            mosaic_targets.append(targets)

            # adjust polygons 1:1 with targets rows
            for p in polys_abs:
                if p.size == 0:
                    mosaic_segments.append(np.empty((0, 2), dtype=np.float32))
                    continue
                pp = p.astype(np.float32).copy()
                pp[:, 0] = pp[:, 0] * scale_w + padw
                pp[:, 1] = pp[:, 1] * scale_h + padh
                mosaic_segments.append(pp)

        if len(mosaic_targets):
            mosaic_targets = np.concatenate(mosaic_targets, 0)

            # Clip polygons to the mosaic canvas and update bboxes from clipped polygons
            canvas_w, canvas_h = 2 * self.target_w, 2 * self.target_h
            clipped_segments = []
            valid_indices = []
            for i, poly in enumerate(mosaic_segments):
                if poly.size == 0:
                    clipped_segments.append(np.empty((0, 2), dtype=np.float32))
                    valid_indices.append(i)
                    continue
                clipped = clip_polygon_to_rect(poly, canvas_w, canvas_h)
                if clipped.size >= 6:  # At least 3 points for a valid polygon
                    clipped_segments.append(clipped)
                    valid_indices.append(i)
                    # Update bbox from clipped polygon
                    x_min, y_min = clipped.min(axis=0)
                    x_max, y_max = clipped.max(axis=0)
                    mosaic_targets[i, 1:5] = [x_min, y_min, x_max, y_max]
                else:
                    # Polygon became invalid after clipping, mark for removal
                    clipped_segments.append(np.empty((0, 2), dtype=np.float32))
                    valid_indices.append(i)

            mosaic_segments = clipped_segments

            # Clip bboxes (for detection-only annotations that don't have polygons)
            np.clip(mosaic_targets[:, 1], 0, canvas_w, out=mosaic_targets[:, 1])
            np.clip(mosaic_targets[:, 2], 0, canvas_h, out=mosaic_targets[:, 2])
            np.clip(mosaic_targets[:, 3], 0, canvas_w, out=mosaic_targets[:, 3])
            np.clip(mosaic_targets[:, 4], 0, canvas_h, out=mosaic_targets[:, 4])

        mosaic_img, mosaic_targets, mosaic_segs = random_affine(
            mosaic_img,
            mosaic_targets if len(mosaic_targets) else np.zeros((0, 5), dtype=np.float32),
            mosaic_segments if len(mosaic_segments) else [],
            target_size=(self.target_w, self.target_h),
            degrees=self.degrees,
            translate=self.translate,
            scales=self.mosaic_scale,
            shear=self.shear,
        )

        # remove tiny boxes after affine
        if mosaic_targets.shape[0]:
            box_heights = mosaic_targets[:, 3] - mosaic_targets[:, 1]
            box_widths = mosaic_targets[:, 4] - mosaic_targets[:, 2]
            keep = np.minimum(box_heights, box_widths) > 1
            mosaic_targets = mosaic_targets[keep]
            mosaic_segs = [p for p, k in zip(mosaic_segs, keep) if k]
        else:
            mosaic_segs = []

        image = self.mosaic_transform(image=mosaic_img)["image"]
        labels = torch.tensor(mosaic_targets[:, 0], dtype=torch.int64)
        boxes = torch.tensor(mosaic_targets[:, 1:], dtype=torch.float32)

        # rasterize masks from transformed polygons
        if self.return_masks and len(mosaic_segs):
            H, W = self.target_h, self.target_w
            masks = [
                poly_abs_to_mask(p, H, W) if p.size else np.zeros((H, W), np.uint8)
                for p in mosaic_segs
            ]
            masks_t = torch.from_numpy(np.stack(masks, 0)).to(torch.uint8)
        else:
            masks_t = torch.zeros((0, self.target_h, self.target_w), dtype=torch.uint8)
        return image, labels, boxes, masks_t, (self.target_h, self.target_w)

    def close_mosaic(self):
        self.mosaic_prob = 0.0
        if is_main_process():
            logger.info("Closing mosaic")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns
            image: CHW tensor
            labels: (N,) long
            boxes: (N,4) normalized xywh
            masks_t: (N,H,W) uint8 (possibly N=0)
            image_path: Path
            orig_size: torch.tensor([H, W])
        """
        image_path = Path(self.split.iloc[idx].values[0])
        if random.random() < self.mosaic_prob:
            image, labels, boxes, masks_t, orig_size = self._load_mosaic(idx)
        else:
            image, targets, orig_size, polys_abs = self._get_data(idx)  # boxes in abs xyxy format

            if self.ignore_background and np.all(targets == 0) and self.mode == "train":
                return None

            # remove tiny objects
            if targets.shape[0]:
                box_heights = targets[:, 3] - targets[:, 1]
                box_widths = targets[:, 4] - targets[:, 2]
                keep = np.minimum(box_heights, box_widths) > 0
                targets = targets[keep]
                polys_abs = [p for p, k in zip(polys_abs, keep) if k]
            else:
                polys_abs = []

            masks_list = []
            if self.return_masks and len(polys_abs) > 0:
                H, W = image.shape[0], image.shape[1]
                masks_list = [poly_abs_to_mask(p, H, W) for p in polys_abs]

            # Apply transformations
            if self.return_masks:
                transformed = self.transform(
                    image=image, bboxes=targets[:, 1:], class_labels=targets[:, 0], masks=masks_list
                )
                masks = transformed.get("masks", [])
                if len(masks):
                    masks_t = torch.stack([m.squeeze().to(dtype=torch.uint8) for m in masks], dim=0)
                else:
                    masks_t = torch.zeros(
                        (0, transformed["image"].shape[1], transformed["image"].shape[2]),
                        dtype=torch.uint8,
                    )
            else:
                transformed = self.transform(
                    image=image, bboxes=targets[:, 1:], class_labels=targets[:, 0]
                )
                masks_t = torch.zeros(
                    (0, transformed["image"].shape[1], transformed["image"].shape[2]),
                    dtype=torch.uint8,
                )

            image = transformed["image"]  # RGB, CHW
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)  # abs xyxy
            labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)

        if self.debug_img_processing and idx <= self.cases_to_debug:
            self._debug_image(idx, image, boxes, labels, image_path, masks=masks_t)

        # return back to normalized format for model
        boxes = torch.tensor(
            abs_xyxy_to_norm_xywh(boxes, image.shape[1], image.shape[2]), dtype=torch.float32
        )
        return image, labels, boxes, masks_t, image_path, orig_size

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
        self.train_sampler = None

    def _get_splits(self) -> None:
        self.splits = {"train": None, "val": None, "test": None}
        for split_name in self.splits.keys():
            if (self.root_path / f"{split_name}.csv").exists():
                self.splits[split_name] = pd.read_csv(
                    self.root_path / f"{split_name}.csv", header=None
                )
            else:
                self.splits[split_name] = []
        assert len(self.splits["train"]) and len(self.splits["val"]), (
            "Train and Val splits must be present"
        )

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
                targets, _ = parse_yolo_label_file(labels_path)
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

    def _build_dataloader_impl(
        self, dataset: Dataset, shuffle: bool = False, distributed: bool = False
    ) -> DataLoader:
        collate_fn = self.val_collate_fn
        if dataset.mode == "train":
            collate_fn = self.train_collate_fn

        sampler = None
        shuffle_flag = shuffle

        if distributed:
            # Use DistributedSampler for both train and val/test in DDP mode
            # For val/test: shuffle=False, drop_last=False to ensure all samples are evaluated
            sampler = DistributedSampler(
                dataset, shuffle=(shuffle and dataset.mode == "train"), drop_last=False
            )
            shuffle_flag = False  # cannot use shuffle=True when sampler is set

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle_flag,
            sampler=sampler,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            prefetch_factor=4,
            pin_memory=True,
        )

        if dataset.mode == "train":
            self.train_sampler = sampler

        return dataloader

    def build_dataloaders(
        self, distributed: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

        train_loader = self._build_dataloader_impl(train_ds, shuffle=True, distributed=distributed)
        val_loader = self._build_dataloader_impl(val_ds, shuffle=False, distributed=distributed)

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
            test_loader = self._build_dataloader_impl(
                test_ds, shuffle=False, distributed=distributed
            )

        if is_main_process():
            logger.info(
                f"Images in train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
            )
            obj_stats = self._get_label_stats()
            sorted_obj_stats = dict(
                sorted(obj_stats.items(), key=lambda item: item[1], reverse=True)
            )
            logger.info(
                f"Objects count: {', '.join(f'{key}: {value}' for key, value in sorted_obj_stats.items())}"
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

        for item in batch:
            target_dict = {
                "labels": item[1],
                "boxes": item[2],
                "masks": item[3],
                "orig_size": item[5],
            }
            images.append(item[0])
            targets.append(target_dict)
            img_paths.append(item[4])

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

            for t in targets:
                m = t["masks"]
                if m.numel() == 0:
                    continue
                m = m.unsqueeze(1).float()  # (N,1,H,W)
                m = torch.nn.functional.interpolate(m, size=(new_h, new_w), mode="nearest")
                t["masks"] = (m.squeeze(1) > 0.5).to(torch.uint8)  # back to (N,H,W)
        return images, targets, img_paths
