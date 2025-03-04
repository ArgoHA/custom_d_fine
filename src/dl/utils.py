import math
import os
import random
import subprocess
import time
from pathlib import Path
from shutil import rmtree
from typing import Dict

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger
from tabulate import tabulate

from src.ptypes import label_to_name_mapping


def set_seeds(seed: int, cudnn_fixed: bool = False) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cudnn_fixed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):  # noqa
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def wandb_logger(loss, metrics: Dict[str, float], epoch, mode: str) -> None:
    log_data = {"epoch": epoch}
    if loss:
        log_data[f"{mode}/loss/"] = loss

    for metric_name, metric_value in metrics.items():
        log_data[f"{mode}/metrics/{metric_name}"] = metric_value

    wandb.log(log_data)


def log_metrics_locally(
    all_metrics: Dict[str, Dict[str, float]], path_to_save: Path, epoch: int
) -> None:
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    metrics_df = metrics_df.round(4)
    metrics_df = metrics_df[
        ["mAP_50", "f1", "precision", "recall", "iou", "mAP_50_95", "TPs", "FPs", "FNs"]
    ]

    tabulated_data = tabulate(metrics_df, headers="keys", tablefmt="pretty", showindex=True)
    if epoch:
        logger.info(f"Metrics on epoch {epoch}:\n{tabulated_data}\n")
    else:
        logger.info(f"Best epoch metrics:\n{tabulated_data}\n")

    if path_to_save:
        metrics_df.to_csv(path_to_save / "metrics.csv")


def save_metrics(train_metrics, metrics, loss, epoch, path_to_save) -> None:
    log_metrics_locally(
        all_metrics={"train": train_metrics, "val": metrics}, path_to_save=path_to_save, epoch=epoch
    )
    wandb_logger(loss, train_metrics, epoch, mode="train")
    wandb_logger(None, metrics, epoch, mode="val")


def calculate_remaining_time(
    one_epoch_time, epoch_start_time, epoch, epochs, cur_iter, all_iters
) -> str:
    if one_epoch_time is None:
        average_iter_time = (time.time() - epoch_start_time) / cur_iter
        remaining_iters = epochs * all_iters - cur_iter

        hours, remainder = divmod(average_iter_time * remaining_iters, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}"

    time_for_remaining_epochs = one_epoch_time * (epochs + 1 - epoch)
    current_epoch_progress = time.time() - epoch_start_time
    hours, remainder = divmod(time_for_remaining_epochs - current_epoch_progress, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}"


def get_vram_usage():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        used, total = map(float, output.strip().split(", "))
        return round((used / total) * 100)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        return 0


def norm_xywh_to_abs_xyxy(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    # Convert normalized centers to absolute pixel coordinates
    x_center = boxes[:, 0] * width
    y_center = boxes[:, 1] * height
    box_width = boxes[:, 2] * width
    box_height = boxes[:, 3] * height

    # Compute the top-left and bottom-right coordinates
    x_min = x_center - (box_width / 2)
    y_min = y_center - (box_height / 2)
    x_max = x_center + (box_width / 2)
    y_max = y_center + (box_height / 2)

    # Convert coordinates to integers
    if isinstance(boxes, np.ndarray):
        x_min = np.maximum(np.floor(x_min), 1)
        y_min = np.maximum(np.floor(y_min), 1)
        x_max = np.minimum(np.ceil(x_max), width - 1)
        y_max = np.minimum(np.ceil(y_max), height - 1)
        return np.stack([x_min, y_min, x_max, y_max], axis=1)
    elif isinstance(boxes, torch.Tensor):
        x_min = torch.clamp(torch.floor(x_min), min=1)
        y_min = torch.clamp(torch.floor(y_min), min=1)
        x_max = torch.clamp(torch.ceil(x_max), max=width - 1)
        y_max = torch.clamp(torch.ceil(y_max), max=height - 1)
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def abs_xyxy_to_norm_xywh(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    x_center = (boxes[:, 0] + boxes[:, 2]) / 2 / width
    y_center = (boxes[:, 1] + boxes[:, 3]) / 2 / height
    box_width = (boxes[:, 2] - boxes[:, 0]) / width
    box_height = (boxes[:, 3] - boxes[:, 1]) / height
    return np.stack([x_center, y_center, box_width, box_height], axis=1)


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(value)
        )


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T
        )  # segment xy
    return segments


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint,
    # i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    (x, y) = (x[inside], y[inside])
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def box_candidates(
    box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16
):  # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (
        (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)
    )  # candidates


def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    new_height, new_width = new_shape
    # Center
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = get_aug_params(scale, center=1.0)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * new_height
    )  # y transla ion (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    return M, s


def random_affine(img, targets, segments, target_size, degrees, translate, scales, shear):
    M, scale = get_transform_matrix(img.shape[:2], target_size, degrees, scales, shear, translate)

    if (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if (n and len(segments) == 0) or (len(segments) != len(targets)):
        new = np.zeros((n, 4))

        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, target_size[0])
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, target_size[1])

    else:
        segments = resample_segments(segments)  # upsample
        new = np.zeros((len(targets), 4))
        assert len(segments) <= len(targets)
        for i, segment in enumerate(segments):
            xy = np.ones((len(segment), 3))
            xy[:, :2] = segment
            xy = xy @ M.T  # transform
            xy = xy[:, :2]  # perspective rescale or affine
            # clip
            new[i] = segment2box(xy, target_size[0], target_size[1])

    # filter candidates
    i = box_candidates(box1=targets[:, 1:5].T * scale, box2=new.T, area_thr=0.1)
    targets = targets[i]
    targets[:, 1:5] = new[i]

    return img, targets


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, target_h, target_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, target_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(target_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, target_w * 2), min(target_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


def filter_preds(preds, conf_thresh):
    for pred in preds:
        keep_idxs = pred["scores"] >= conf_thresh
        pred["scores"] = pred["scores"][keep_idxs]
        pred["boxes"] = pred["boxes"][keep_idxs]
        pred["labels"] = pred["labels"][keep_idxs]
    return preds


def vis_one_box(img, box, label, mode, score=None):
    if mode == "gt":
        prefix = "GT: "
        color = (46, 153, 60)
        postfix = ""
    elif mode == "pred":
        prefix = ""
        color = (148, 70, 44)
        postfix = f" {score:.2f}"

    x1, y1, x2, y2 = map(int, box.tolist())
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color=color,
        thickness=2,
    )
    cv2.putText(
        img,
        f"{prefix}{label_to_name_mapping[int(label)]}{postfix}",
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        thickness=1,
    )


def visualize(img_paths, gt, preds, dataset_path, path_to_save):
    """
    Saves images with drawn bounding boxes.
      - Green bboxes for GT
      - Blue bboxes for preds
    """
    path_to_save.mkdir(parents=True, exist_ok=True)

    for i, (gt_dict, pred_dict, img_path) in enumerate(zip(gt, preds, img_paths)):
        img = cv2.imread(str(dataset_path / img_path))

        # Draw ground-truth boxes (green)
        for box, label in zip(gt_dict["boxes"], gt_dict["labels"]):
            # box: [x1, y1, x2, y2]
            vis_one_box(img, box, label, mode="gt")

        # Draw predicted boxes (blue)
        for box, label, score in zip(pred_dict["boxes"], pred_dict["labels"], pred_dict["scores"]):
            vis_one_box(img, box, label, mode="pred", score=score)

        # Construct a filename and save
        outpath = path_to_save / img_path.name
        cv2.imwrite(str(outpath), img)


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes_ratio_kept(boxes, img0_shape, img1_shape, ratio_pad=None, padding=True):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_boxes(boxes, orig_shape, resized_shape):
    """
    boxes in format: [x1, y1, x2, y2], absolute values
    orig_shape: [height, width]
    resized_shape: [height, width]
    """
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes


def process_boxes(boxes, processed_size, orig_sizes, keep_ratio, device):
    """
    Inputs:
        boxes: Torch.tensor[batch_size, num_boxes, 4]
        processed_size: Torch.tensor[2] h, w
        orig_sizes: Torch.tensor[batch_size, 2] h, w
        keep_ratio: bool
        device: Torch.device

    Outputs:
        Torch.tensor[batch_size, num_boxes, 4]

    """
    bs = orig_sizes.shape[0]
    processed_sizes = np.repeat(
        np.array([processed_size[0], processed_size[1]])[None, :], bs, axis=0
    )
    orig_sizes = orig_sizes.cpu().numpy()
    boxes = boxes.cpu().numpy()

    final_boxes = np.zeros_like(boxes)
    for idx, box in enumerate(boxes):
        final_boxes[idx] = norm_xywh_to_abs_xyxy(
            box, processed_sizes[idx][0], processed_sizes[idx][1]
        )

    for i in range(bs):
        if keep_ratio:
            final_boxes[i] = scale_boxes_ratio_kept(
                final_boxes[i],
                orig_sizes[i],
                processed_sizes[i],
            )
        else:
            final_boxes[i] = scale_boxes(
                final_boxes[i],
                orig_sizes[i],
                processed_sizes[i],
            )
    return torch.tensor(final_boxes).to(device)


class CustomPadIfNeeded(A.DualTransform):
    """
    Custom padding transform that, with probability `rect_resize_prob`, pads the image
    to the nearest dimensions divisible by `div` and otherwise pads to fixed square dimensions.
    Also adjusts bounding boxes accordingly.
    """

    def __init__(
        self,
        square_height: int,
        square_width: int,
        rect_resize_prob: float = 0.0,
        div: int = 32,
        border_mode=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.square_height = square_height
        self.square_width = square_width
        self.rect_resize_prob = rect_resize_prob
        self.div = div
        self.border_mode = border_mode
        self.value = value

    def get_params_dependent_on_targets(self, params):
        # We need the image size to compute padding
        image = params["image"]
        h, w = image.shape[:2]
        # Decide which padding mode to use
        if random.random() < self.rect_resize_prob:
            new_h = math.ceil(h / self.div) * self.div
            new_w = math.ceil(w / self.div) * self.div
        else:
            new_h, new_w = self.square_height, self.square_width

        pad_h = max(new_h - h, 0)
        pad_w = max(new_w - w, 0)

        # For center padding
        top = pad_h // 2
        left = pad_w // 2

        return {"top": top, "left": left, "new_h": new_h, "new_w": new_w}

    def apply(self, image, top=0, left=0, new_h=None, new_w=None, **params):
        h, w = image.shape[:2]
        bottom = new_h - h - top
        right = new_w - w - left
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, self.border_mode, value=self.value
        )
        return image

    def apply_to_bbox(self, bbox, top=0, left=0, **params):
        # bbox: [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox
        x_min += left
        x_max += left
        y_min += top
        y_max += top
        return [x_min, y_min, x_max, y_max]

    def get_transform_init_args_names(self):
        return (
            "square_height",
            "square_width",
            "rect_resize_prob",
            "div",
            "border_mode",
            "value",
        )
