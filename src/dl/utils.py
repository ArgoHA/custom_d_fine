import logging
import math
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import wandb
from albumentations.core.transforms_interface import DualTransform
from faster_coco_eval.core import mask as mask_utils
from loguru import logger
from tabulate import tabulate

logging.getLogger("faster_coco_eval").setLevel(logging.WARNING)


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
        if metric_name == "extended_metrics":
            for ext_metric_name, ext_metric_value in metric_value.items():
                log_data[f"{mode}_extended/{ext_metric_name}"] = ext_metric_value
        else:
            log_data[f"{mode}/metrics/{metric_name}"] = metric_value

    wandb.log(log_data)


def rename_metric_keys(d, label_to_name):
    """precision_1 -> precision_class_name"""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if "_" in k:
            base, tail = k.rsplit("_", 1)
            if tail.isdigit():
                name = label_to_name.get(int(tail), tail)
                k = f"{base}_{name}"
        out[k] = v
    return out


def log_metrics_locally(
    all_metrics: Dict[str, Dict[str, float]], path_to_save: Path, epoch: int, extended=False
) -> None:
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    metrics_df = metrics_df.round(4)
    if extended:
        extended_metrics = pd.DataFrame.from_records(
            metrics_df["extended_metrics"].tolist(), index=metrics_df.index
        ).round(4)

    metrics_list = ["mAP_50", "f1", "precision", "recall", "iou", "mAP_50_95", "TPs", "FPs", "FNs"]
    if "mAP_50_mask" in metrics_df.columns:
        metrics_list.insert(1, "mAP_50_mask")
        metrics_list.remove("mAP_50_95")
    metrics_df = metrics_df[metrics_list]

    tabulated_data = tabulate(metrics_df, headers="keys", tablefmt="pretty", showindex=True)
    if epoch:
        logger.info(f"Metrics on epoch {epoch}:\n{tabulated_data}\n")
    else:
        logger.info(f"Best epoch metrics:\n{tabulated_data}\n")

    if path_to_save:
        metrics_df.to_csv(path_to_save / "metrics.csv")

        if extended:
            extended_metrics.to_csv(path_to_save / "extended_metrics.csv")


def save_metrics(train_metrics, metrics, loss, epoch, path_to_save, use_wandb) -> None:
    log_metrics_locally(
        all_metrics={"train": train_metrics, "val": metrics}, path_to_save=path_to_save, epoch=epoch
    )
    if use_wandb:
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

    time_for_remaining_epochs = max(one_epoch_time * (epochs + 1 - epoch), 0)
    current_epoch_progress = time.time() - epoch_start_time
    hours, remainder = divmod(time_for_remaining_epochs - current_epoch_progress, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}"


def get_vram_usage():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )

        # Split lines to handle multiple GPUs correctly
        lines = output.strip().split("\n")
        total_usage = []

        for line in lines:
            try:
                used, total = map(float, line.split(", "))
                total_usage.append((used / total) * 100)
            except ValueError:
                print(f"Skipping malformed line: {line}")

        # If there are multiple GPUs, return the max usage percentage
        return round(max(total_usage)) if total_usage else 0

    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        return 0


def norm_xywh_to_abs_xyxy(boxes: np.ndarray, height: int, width: int, to_round=True) -> np.ndarray:
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
    if to_round:
        x_min = np.maximum(np.floor(x_min), 1)
        y_min = np.maximum(np.floor(y_min), 1)
        x_max = np.minimum(np.ceil(x_max), width - 1)
        y_max = np.minimum(np.ceil(y_max), height - 1)
        return np.stack([x_min, y_min, x_max, y_max], axis=1)
    else:
        x_min = np.maximum(x_min, 0)
        y_min = np.maximum(y_min, 0)
        x_max = np.minimum(x_max, width)
        y_max = np.minimum(y_max, height)
        return np.stack([x_min, y_min, x_max, y_max], axis=1)


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


def clip_polygon_to_rect(poly: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    Clip a polygon to a rectangle [0, width] x [0, height] using Sutherland-Hodgman algorithm.
    Returns the clipped polygon as (M, 2) array, or empty (0, 2) if fully outside.
    """
    if poly.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    def inside(p, edge):
        x, y = p
        if edge == "left":
            return x >= 0
        elif edge == "right":
            return x <= width
        elif edge == "top":
            return y >= 0
        elif edge == "bottom":
            return y <= height

    def intersection(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        if edge == "left":
            t = (0 - x1) / dx if dx != 0 else 0
            return np.array([0, y1 + t * dy])
        elif edge == "right":
            t = (width - x1) / dx if dx != 0 else 0
            return np.array([width, y1 + t * dy])
        elif edge == "top":
            t = (0 - y1) / dy if dy != 0 else 0
            return np.array([x1 + t * dx, 0])
        elif edge == "bottom":
            t = (height - y1) / dy if dy != 0 else 0
            return np.array([x1 + t * dx, height])

    output = poly.copy()
    for edge in ["left", "right", "top", "bottom"]:
        if len(output) == 0:
            return np.empty((0, 2), dtype=np.float32)
        input_list = output
        output = []
        for i in range(len(input_list)):
            current = input_list[i]
            prev = input_list[i - 1]
            if inside(current, edge):
                if not inside(prev, edge):
                    output.append(intersection(prev, current, edge))
                output.append(current)
            elif inside(prev, edge):
                output.append(intersection(prev, current, edge))
        output = np.array(output) if len(output) > 0 else np.empty((0, 2), dtype=np.float32)

    if len(output) < 3:
        return np.empty((0, 2), dtype=np.float32)
    return output.astype(np.float32)


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
    new_width, new_height = new_shape
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
    """
    Args:
      img: (Hbig, Wbig, 3)
      targets: (N, 5) -> [cls, x1, y1, x2, y2] ABS on the mosaic canvas
      segments: list[np.ndarray] of length N; each (K,2) ABS polygon on the mosaic canvas
                If an object comes from bbox-only annotation, pass an empty array for it.
    Returns:
      img_aff: final (target_h, target_w, 3)
      targets_aff: (M, 5) filtered + transformed
      segments_aff: list[np.ndarray] length=M, transformed polygons
    """
    M, scale = get_transform_matrix(img.shape[:2], target_size, degrees, scales, shear, translate)

    # warp image
    if (M != np.eye(3)).any():
        img = cv2.warpAffine(img, M[:2], dsize=target_size, borderValue=(114, 114, 114))

    n = len(targets)
    if n:
        # transform boxes by corners
        xy = np.ones((n * 4, 3), dtype=np.float32)
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)

        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], axis=1)

        # clip boxes into target frame
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, target_size[0])
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, target_size[1])

        # transform segments (if provided)
        segs_out = []
        if segments is None or len(segments) == 0:
            segs_out = [np.empty((0, 2), dtype=np.float32) for _ in range(n)]
        else:
            # keep 1:1 with targets
            for idx, s in enumerate(segments):
                if s.size == 0:
                    segs_out.append(np.empty((0, 2), dtype=np.float32))
                    continue
                pts = np.concatenate([s, np.ones((len(s), 1), dtype=np.float32)], axis=1)  # (K,3)
                pts = pts @ M.T
                pts = pts[:, :2]
                # Properly clip polygon to the target frame
                clipped = clip_polygon_to_rect(pts, target_size[0], target_size[1])
                if clipped.size >= 6:  # At least 3 points for a valid polygon
                    segs_out.append(clipped)
                    # Update bounding box from clipped polygon
                    x_min, y_min = clipped.min(axis=0)
                    x_max, y_max = clipped.max(axis=0)
                    new[idx] = [x_min, y_min, x_max, y_max]
                else:
                    segs_out.append(np.empty((0, 2), dtype=np.float32))

        # filter candidates and keep segments in sync
        i = box_candidates(box1=targets[:, 1:5].T * scale, box2=new.T, area_thr=0.1)
        targets = targets[i]
        targets[:, 1:5] = new[i]
        segs_out = [segs_out[k] for k, keep in enumerate(i) if keep]

    else:
        segs_out = []

    return img, targets, segs_out


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


def filter_preds(preds, conf_thresh, mask_source="mask_probs"):
    """
    Filters predictions by score AND keeps masks in-sync with the kept indices.
    - If mask_source == "mask_probs" and present, also populates pred["masks"] as uint8 via conf_thresh.
    """
    for pred in preds:
        keep = pred["scores"] >= conf_thresh
        pred["scores"] = pred["scores"][keep]
        pred["boxes"] = pred["boxes"][keep]
        pred["labels"] = pred["labels"][keep]

        # Keep mask tensors aligned with kept queries
        if (
            mask_source in pred
            and pred[mask_source] is not None
            and getattr(pred[mask_source], "numel", lambda: 0)() > 0
        ):
            m = pred[mask_source][keep]
            pred[mask_source] = m
            # Ensure binary mask view exists (uint8)
            if mask_source == "mask_probs":
                pred["masks"] = (m > conf_thresh).to(torch.uint8)
        elif (
            "masks" in pred
            and pred["masks"] is not None
            and getattr(pred["masks"], "numel", lambda: 0)() > 0
        ):
            pred["masks"] = pred["masks"][keep].to(torch.uint8)

    return preds


def filter_masks(preds, conf_thresh, mask_source="mask_probs"):
    for pred in preds:
        keep = pred["scores"] >= conf_thresh

        # Keep mask tensors aligned with kept queries
        if (
            mask_source in pred
            and pred[mask_source] is not None
            and getattr(pred[mask_source], "numel", lambda: 0)() > 0
        ):
            m = pred[mask_source][keep]
            pred[mask_source] = m
            # Ensure binary mask view exists (uint8)
            if mask_source == "mask_probs":
                pred["masks"] = (m > conf_thresh).to(torch.uint8)

    return preds


def label_color(label: int):
    # deterministic color per class (BGR for OpenCV)
    palette = [
        (255, 56, 56),
        (255, 159, 56),
        (255, 255, 56),
        (56, 255, 56),
        (56, 255, 255),
        (56, 56, 255),
        (255, 56, 255),
        (180, 130, 70),
        (204, 153, 255),
        (80, 175, 76),
        (42, 157, 143),
        (233, 196, 106),
        (244, 162, 97),
        (231, 111, 81),
        (69, 123, 157),
        (29, 53, 87),
    ]
    return palette[int(label) % len(palette)]


def draw_mask(
    img: np.ndarray, mask: np.ndarray, color=(148, 70, 44), alpha: float = 0.4, outline: bool = True
):
    """
    img: BGR uint8 [H,W,3]
    mask: uint8/bool [H,W] (1=mask)
    color: (B,G,R)
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if mask.ndim == 3:  # [1,H,W] -> [H,W]
        mask = mask.squeeze(0)

    if mask.max() == 0:
        return img

    # fast alpha blend on masked pixels
    m = mask.astype(bool)
    overlay = np.zeros_like(img, dtype=np.uint8)
    overlay[:] = color
    img[m] = cv2.addWeighted(img[m], 1 - alpha, overlay[m], alpha, 0)

    if outline:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 2)
    return img


def vis_one_box(img, box, label, mode, label_to_name, score=None):
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
    y = y1 - 16 if mode == "gt" else y1 - 4
    cv2.putText(
        img,
        f"{prefix}{label_to_name[int(label)]}{postfix}",
        (x1, max(0, y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        thickness=2,
    )


def visualize(
    img_paths,
    gt,
    preds,
    dataset_path,
    path_to_save,
    label_to_name,
    mask_alpha_gt: float = 0.35,
    mask_alpha_pred: float = 0.40,
):
    """
    Saves images with:
      - GT: green boxes + optional green masks
      - Preds: brown boxes + colored masks per class
    Expects pred dicts possibly containing "masks" (uint8)
    """
    path_to_save.mkdir(parents=True, exist_ok=True)

    draw_gt_masks = "masks" in gt[0]
    draw_pred_masks = "masks" in preds[0]

    for gt_dict, pred_dict, img_path in zip(gt, preds, img_paths):
        img = cv2.imread(str(dataset_path / img_path))
        if img is None:
            continue

        # Draw GT masks (green-ish)
        if (
            draw_gt_masks
            and "masks" in gt_dict
            and gt_dict["masks"] is not None
            and len(gt_dict["masks"]) > 0
            and gt_dict["masks"].shape[1] != 0
        ):
            for m in gt_dict["masks"]:
                img = draw_mask(
                    img, m.numpy(), color=(46, 153, 60), alpha=mask_alpha_gt, outline=True
                )

        # Draw GT boxes (green)
        for box, label in zip(gt_dict["boxes"], gt_dict["labels"]):
            vis_one_box(img, box, label, mode="gt", label_to_name=label_to_name)

        # Prepare predicted masks
        pred_masks_to_draw = None
        if draw_pred_masks:
            if (
                "masks" in pred_dict
                and pred_dict["masks"] is not None
                and len(pred_dict["masks"]) > 0
            ):
                pred_masks_to_draw = pred_dict["masks"]
        # Draw predicted masks (colored by class)
        if pred_masks_to_draw is not None:
            pm = pred_masks_to_draw.cpu().numpy()
            for m, lab in zip(pm, pred_dict["labels"]):
                color = label_color(int(lab))
                img = draw_mask(img, m, color=color, alpha=mask_alpha_pred, outline=True)

        # Draw predicted boxes (blue-ish)
        for box, label, score in zip(pred_dict["boxes"], pred_dict["labels"], pred_dict["scores"]):
            vis_one_box(
                img,
                box,
                label,
                mode="pred",
                label_to_name=label_to_name,
                score=score,
            )

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


def process_masks(
    pred_masks,  # Tensor [B, Q, Hm, Wm] or [Q, Hm, Wm]
    processed_size,  # (H, W) of network input (after your A.Compose)
    orig_sizes,  # Tensor [B, 2] (H, W)
    keep_ratio: bool,
) -> List[torch.Tensor]:
    """
    Returns list of length B with masks resized to original image sizes:
    Each item: Float Tensor [Q, H_orig, W_orig] in [0,1] (no thresholding here).
    - Handles letterbox padding removal if keep_ratio=True.
    - Works for both batched and single-image inputs.
    """
    single = pred_masks.dim() == 3  # [Q,Hm,Wm]
    if single:
        pred_masks = pred_masks.unsqueeze(0)  # -> [1,Q,Hm,Wm]

    if pred_masks.shape[1] == 0:
        return [torch.zeros((0, int(orig_sizes[0, 0]), int(orig_sizes[0, 1])))]

    B, Q, Hm, Wm = pred_masks.shape
    device = pred_masks.device
    dtype = pred_masks.dtype

    # 1) Upsample masks to processed (input) size
    proc_h, proc_w = int(processed_size[0]), int(processed_size[1])
    masks_proc = torch.nn.functional.interpolate(
        pred_masks, size=(proc_h, proc_w), mode="bilinear", align_corners=False
    )  # [B,Q,Hp,Wp] with Hp=proc_h, Wp=proc_w

    out = []
    for b in range(B):
        H0, W0 = int(orig_sizes[b, 0].item()), int(orig_sizes[b, 1].item())
        m = masks_proc[b]  # [Q, Hp, Wp]
        if keep_ratio:
            # Compute same gain/pad as in scale_boxes_ratio_kept
            gain = min(proc_h / H0, proc_w / W0)
            padw = round((proc_w - W0 * gain) / 2 - 0.1)
            padh = round((proc_h - H0 * gain) / 2 - 0.1)

            # Remove padding before final resize
            y1 = max(padh, 0)
            y2 = proc_h - max(padh, 0)
            x1 = max(padw, 0)
            x2 = proc_w - max(padw, 0)
            m = m[:, y1:y2, x1:x2]  # [Q, cropped_h, cropped_w]

        # 2) Resize to original size
        m = torch.nn.functional.interpolate(
            m.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False
        ).squeeze(0)  # [Q, H0, W0]
        out.append(m.clamp_(0, 1).to(device=device, dtype=dtype))

    if single:
        return [out[0]]
    return out


def cleanup_masks(masks, boxes):
    # clean up masks outside of the corresponding bbox
    N, H, W = masks.shape
    ys = torch.arange(H)[None, :, None]  # (1, H, 1)
    xs = torch.arange(W)[None, None, :]  # (1, 1, W)

    x1, y1, x2, y2 = boxes.T
    inside = (
        (xs >= x1[:, None, None])
        & (xs < x2[:, None, None])
        & (ys >= y1[:, None, None])
        & (ys < y2[:, None, None])
    )  # (N, H, W), bool
    masks = masks * inside.to(dtype=masks.dtype)
    return masks


def get_latest_experiment_name(exp: str, output_dir: str):
    output_dir = Path(output_dir)
    if output_dir.exists():
        return exp

    target_exp_name = Path(exp).name.rsplit("_", 1)[0]
    latest_exp = None

    for exp_path in output_dir.parent.iterdir():
        exp_name, exp_date = exp_path.name.rsplit("_", 1)
        if target_exp_name == exp_name:
            exp_date = datetime.strptime(exp_date, "%Y-%m-%d")
            if not latest_exp or exp_date > latest_exp:
                latest_exp = exp_date

    final_exp_name = f"{target_exp_name}_{latest_exp.strftime('%Y-%m-%d')}"
    logger.info(f"Latest experiment: {final_exp_name}")
    return final_exp_name


class LetterboxRect(DualTransform):
    def __init__(
        self,
        height: int,
        width: int,
        color=(114, 114, 114),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        stride: int = 32,
        always_apply: bool = True,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.height = int(height)
        self.width = int(width)
        self.color = tuple(color)
        self.auto = bool(auto)
        self.scale_fill = bool(scale_fill)
        self.scaleup = bool(scaleup)
        self.stride = int(stride)

    def get_transform_init_args_names(self):
        return ("height", "width", "color", "auto", "scale_fill", "scaleup", "stride")

    @property
    def targets_as_params(self):
        return ["image"]

    # Generate all deterministic params needed by apply/apply_to_bboxes
    # (computed once per call, then reused for image and bboxes)
    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        h, w = img.shape[:2]

        if self.scale_fill:
            # stretch to exact size
            new_unpad_w, new_unpad_h = self.width, self.height
            ratio_x = self.width / w
            ratio_y = self.height / h
            dw, dh = 0.0, 0.0
        else:
            # keep aspect ratio
            r = min(self.height / h, self.width / w)
            if not self.scaleup:
                r = min(r, 1.0)

            new_unpad_w = int(round(w * r))
            new_unpad_h = int(round(h * r))

            dw = self.width - new_unpad_w
            dh = self.height - new_unpad_h

            if self.auto:
                # pad to stride multiple, like inference `auto=True`
                dw = np.mod(dw, self.stride)
                dh = np.mod(dh, self.stride)

            ratio_x = r
            ratio_y = r

        # split padding equally to both sides
        dw *= 0.5
        dh *= 0.5

        # match inference border rounding
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))

        return {
            # original size
            "orig_h": h,
            "orig_w": w,
            # resized (pre-pad) size
            "new_w": new_unpad_w,
            "new_h": new_unpad_h,
            # scale ratios
            "ratio_x": float(ratio_x),
            "ratio_y": float(ratio_y),
            # padding to apply
            "pad_left": left,
            "pad_top": top,
            "pad_right": right,
            "pad_bottom": bottom,
            # final canvas target (sanity)
            "target_h": self.height,
            "target_w": self.width,
        }

    # Image transform
    def apply(
        self, img, new_w=0, new_h=0, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0, **kwargs
    ):
        # resize if needed
        if img.shape[1] != new_w or img.shape[0] != new_h:
            img = cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)

        # pad if needed
        if pad_top or pad_bottom or pad_left or pad_right:
            img = cv2.copyMakeBorder(
                img,
                int(pad_top),
                int(pad_bottom),
                int(pad_left),
                int(pad_right),
                cv2.BORDER_CONSTANT,
                value=self.color,
            )
        return img

    # Mask transform - pad with 0 (not object) instead of image color
    def apply_to_mask(
        self, mask, new_w=0, new_h=0, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0, **kwargs
    ):
        # resize if needed - use INTER_NEAREST to preserve binary mask values
        if mask.shape[1] != new_w or mask.shape[0] != new_h:
            mask = cv2.resize(mask, (int(new_w), int(new_h)), interpolation=cv2.INTER_NEAREST)

        # pad if needed - use 0 for masks (padding is not part of any object)
        if pad_top or pad_bottom or pad_left or pad_right:
            mask = cv2.copyMakeBorder(
                mask,
                int(pad_top),
                int(pad_bottom),
                int(pad_left),
                int(pad_right),
                cv2.BORDER_CONSTANT,
                value=0,
            )
        return mask

    # Bboxes transform (Pascal VOC: abs xyxy)
    def apply_to_bboxes(
        self,
        bboxes,
        ratio_x=1.0,
        ratio_y=1.0,
        pad_left=0,
        pad_top=0,
        orig_w=0,
        orig_h=0,
        target_w=0,
        target_h=0,
        **kwargs,
    ):
        # Albumentations passes bboxes in its INTERNAL NORMALIZED format [0..1]
        # We must return normalized bboxes for the transformed image.
        if bboxes is None or len(bboxes) == 0:
            return bboxes

        b = np.asarray(bboxes, dtype=np.float32)

        has_extra = b.shape[1] > 4
        extra = None
        if has_extra:
            extra = b[:, 4:].copy()
            b = b[:, :4]

        # to absolute coordinates (original image)
        b[:, [0, 2]] *= float(orig_w)
        b[:, [1, 3]] *= float(orig_h)

        # resize
        b[:, [0, 2]] *= float(ratio_x)
        b[:, [1, 3]] *= float(ratio_y)

        # pad
        b[:, [0, 2]] += float(pad_left)
        b[:, [1, 3]] += float(pad_top)

        # back to normalized (final canvas)
        b[:, [0, 2]] /= max(float(target_w), 1e-6)
        b[:, [1, 3]] /= max(float(target_h), 1e-6)

        # clip to [0,1] to avoid filtering
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, 1.0)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, 1.0)

        # ensure x2>=x1, y2>=y1 numerically
        b[:, 2] = np.maximum(b[:, 2], b[:, 0])
        b[:, 3] = np.maximum(b[:, 3], b[:, 1])

        if has_extra:
            b = np.concatenate([b, extra], axis=1)

        return b


def norm_poly_to_abs(poly_norm_flat: np.ndarray, H: int, W: int) -> np.ndarray:
    """poly_norm_flat: [x1,y1,x2,y2,...] normalized -> (K,2) absolute"""
    if poly_norm_flat.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = poly_norm_flat.reshape(-1, 2).copy()
    pts[:, 0] *= W
    pts[:, 1] *= H
    return pts.astype(np.float32)


def poly_abs_to_mask(poly_abs: np.ndarray, h: int, w: int) -> np.ndarray:
    pts = poly_abs.copy()
    pts = np.round(pts).astype(np.int32)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


# ============================================================================
# RLE Encoding/Decoding for memory-efficient mask storage
# ============================================================================


def masks_to_rle(masks: torch.Tensor) -> List[Dict]:
    """
    Encode binary masks to COCO RLE format for memory-efficient storage.

    Args:
        masks: [N, H, W] uint8 tensor with binary masks (0/1)

    Returns:
        List of RLE dicts, each with 'size' and 'counts' keys
    """
    if masks is None or masks.numel() == 0:
        return []

    # Ensure proper format: [N, H, W], uint8, on CPU
    if masks.dim() == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    masks = masks.to(torch.uint8).cpu().numpy()

    rles = []
    for m in masks:
        # pycocotools expects Fortran-order (column-major) array
        rle = mask_utils.encode(np.asfortranarray(m))
        # Convert bytes to string for JSON serialization compatibility
        rle["counts"] = (
            rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
        )
        rles.append(rle)

    return rles


def rle_to_masks(rles: List[Dict], device: str = "cpu") -> torch.Tensor:
    """
    Decode RLE-encoded masks back to dense tensor format.

    Args:
        rles: List of RLE dicts from masks_to_rle()
        device: Target device for output tensor

    Returns:
        [N, H, W] uint8 tensor with binary masks
    """
    if not rles:
        return torch.zeros((0, 1, 1), dtype=torch.uint8)

    # Ensure counts are bytes for pycocotools
    rles_bytes = []
    for rle in rles:
        rle_copy = rle.copy()
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
        rles_bytes.append(rle_copy)

    # Decode all masks at once (more efficient)
    masks_np = mask_utils.decode(rles_bytes)  # [H, W, N]
    if masks_np.ndim == 2:
        # Single mask case: [H, W] -> [1, H, W]
        masks_np = masks_np[np.newaxis, ...]
    else:
        masks_np = np.transpose(masks_np, (2, 0, 1))  # [N, H, W]

    return torch.from_numpy(masks_np.copy()).to(dtype=torch.uint8, device=device)


def encode_sample_masks_to_rle(sample: Dict) -> Dict:
    """
    Convert a prediction/GT sample dict to use RLE-encoded masks.
    Replaces 'masks' tensor with 'masks_rle' list and stores original size.

    Args:
        sample: Dict with 'masks' key containing [N, H, W] tensor

    Returns:
        Same dict with 'masks' replaced by 'masks_rle' and 'masks_size'
    """
    if "masks" not in sample or sample["masks"] is None:
        return sample

    masks = sample["masks"]
    if masks.numel() == 0:
        sample["masks_rle"] = []
        sample["masks_size"] = (0, 0)
        del sample["masks"]
        return sample

    # Store size for later decoding
    if masks.dim() == 3:
        _, H, W = masks.shape
    else:
        H, W = masks.shape[-2], masks.shape[-1]

    sample["masks_rle"] = masks_to_rle(masks)
    sample["masks_size"] = (H, W)
    del sample["masks"]

    return sample


def decode_sample_rle_to_masks(sample: Dict, device: str = "cpu") -> Dict:
    """
    Decode RLE masks back to dense tensor format in a sample dict.

    Args:
        sample: Dict with 'masks_rle' key
        device: Target device for output tensor

    Returns:
        Same dict with 'masks' restored from 'masks_rle'
    """
    if "masks_rle" not in sample:
        return sample

    rles = sample["masks_rle"]
    if not rles:
        size = sample.get("masks_size", (1, 1))
        sample["masks"] = torch.zeros((0, size[0], size[1]), dtype=torch.uint8)
    else:
        sample["masks"] = rle_to_masks(rles, device=device)

    return sample


def get_rle_memory_size(rles: List[Dict]) -> int:
    """
    Estimate memory usage of RLE-encoded masks in bytes.
    Useful for debugging/monitoring memory savings.
    """
    if not rles:
        return 0

    total = 0
    for rle in rles:
        # Size list + counts string
        total += 16  # overhead for dict
        total += len(str(rle.get("counts", "")))
        total += 16  # size tuple overhead
    return total


def get_dense_mask_memory_size(n_masks: int, h: int, w: int) -> int:
    """
    Calculate memory usage of dense masks in bytes.
    """
    return n_masks * h * w  # uint8 = 1 byte per pixel
