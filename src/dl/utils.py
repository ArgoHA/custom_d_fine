import math
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch
import wandb
from albumentations.core.transforms_interface import DualTransform
from loguru import logger
from tabulate import tabulate


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


def norm_xywh_to_abs_xyxy(boxes: np.ndarray, h: int, w: int, clip=True):
    """
    Normalised (x_c, y_c, w, h) -> absolute (x1, y1, x2, y2)
    Keeps full floating-point precision; no rounding.
    """
    x_c, y_c, bw, bh = boxes.T
    x_min = x_c * w - bw * w / 2
    y_min = y_c * h - bh * h / 2
    x_max = x_c * w + bw * w / 2
    y_max = y_c * h + bh * h / 2

    if clip:
        x_min = np.clip(x_min, 0, w - 1)
        y_min = np.clip(y_min, 0, h - 1)
        x_max = np.clip(x_max, 0, w - 1)
        y_max = np.clip(y_max, 0, h - 1)

    return np.stack([x_min, y_min, x_max, y_max], axis=1)


def abs_xyxy_to_norm_xywh(boxes: np.ndarray, h: int, w: int):
    """
    Absolute (x1, y1, x2, y2)  →  normalised (x_c, y_c, w, h)
    """
    x1, y1, x2, y2 = boxes.T
    x_c = (x1 + x2) / 2 / w
    y_c = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return np.stack([x_c, y_c, bw, bh], axis=1)


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


def visualize(img_paths, gt, preds, dataset_path, path_to_save, label_to_name):
    """
    Saves images with drawn bounding boxes.
      - Green bboxes for GT
      - Blue bboxes for preds
    """
    path_to_save.mkdir(parents=True, exist_ok=True)

    for gt_dict, pred_dict, img_path in zip(gt, preds, img_paths):
        img = cv2.imread(str(dataset_path / img_path))

        # Draw ground-truth boxes (green)
        for box, label in zip(gt_dict["boxes"], gt_dict["labels"]):
            # box: [x1, y1, x2, y2]
            vis_one_box(img, box, label, mode="gt", label_to_name=label_to_name)

        # Draw predicted boxes (blue)
        for box, label, score in zip(pred_dict["boxes"], pred_dict["labels"], pred_dict["scores"]):
            vis_one_box(
                img,
                box,
                label,
                mode="pred",
                label_to_name=label_to_name,
                score=score,
            )

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
