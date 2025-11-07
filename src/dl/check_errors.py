from pathlib import Path
from shutil import rmtree

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.ops import box_iou
from tqdm import tqdm

from src.dl.dataset import Loader
from src.dl.utils import (
    abs_xyxy_to_norm_xywh,
    get_latest_experiment_name,
    norm_xywh_to_abs_xyxy,
    vis_one_box,
)
from src.infer.torch_model import Torch_model


def norm_xywh_to_xyxy(norm_boxes: torch.Tensor) -> torch.Tensor:
    # boxes: [N,4] in (cx, cy, w, h), all in [0,1]
    cx, cy, w, h = norm_boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)


def save_case(
    case_type: str,
    img: np.ndarray,
    abs_box: torch.Tensor,  # [4] absolute xyxy
    label: int,
    score: float | None,
    img_path: Path,
    output_dir: Path,
    label_to_name: dict,
):
    # Draw and save image + single-line YOLO txt with the corresponding box
    out_img_dir = output_dir / case_type
    out_img_dir.mkdir(parents=True, exist_ok=True)

    mode = "pred" if case_type == "FPs" else "gt"

    draw = img.copy()
    vis_one_box(
        draw,
        abs_box,
        int(label),
        mode=mode,
        label_to_name=label_to_name,
        score=score if score is not None else None,
    )
    cv2.imwrite(str(out_img_dir / f"{Path(img_path).stem}.jpg"), draw)


def check_results(
    img,
    img_path,
    preds,
    targets,
    iou_thresh: float,
    conf_thresh: float,
    output_dir: Path,
    label_to_name: dict,
):
    """
    Match predictions to targets:
      - same class
      - IoU >= iou_thresh (computed on normalized xyxy)
      - pred score >= conf_thresh
    Save unmatched predictions as FP and unmatched targets as FN.
    """
    H, W = img.shape[:2]

    # Prepare tensors
    if len(preds) == 0:
        pred_boxes_xyxy_abs = torch.zeros((0, 4), dtype=torch.float32)
        pred_boxes_norm_xywh = torch.zeros((0, 4), dtype=torch.float32)
        pred_labels = torch.zeros((0,), dtype=torch.long)
        pred_scores = torch.zeros((0,), dtype=torch.float32)
    else:
        pred_boxes_xyxy_abs = torch.as_tensor(preds["boxes"], dtype=torch.float32)  # abs xyxy
        pred_boxes_norm_xywh = torch.as_tensor(
            preds["norm_boxes"], dtype=torch.float32
        )  # norm xywh
        pred_labels = torch.as_tensor(preds["labels"], dtype=torch.long)
        pred_scores = torch.as_tensor(preds["scores"], dtype=torch.float32)

    if len(targets["boxes"]) == 0:
        tgt_boxes_norm_xywh = torch.zeros((0, 4), dtype=torch.float32)
        tgt_labels = torch.zeros((0,), dtype=torch.long)
    else:
        tgt_boxes_norm_xywh = torch.as_tensor(targets["boxes"], dtype=torch.float32)  # norm xywh
        tgt_labels = torch.as_tensor(targets["labels"], dtype=torch.long)

    # filter by conf_thresh
    keep = pred_scores >= conf_thresh
    pred_boxes_xyxy_abs = pred_boxes_xyxy_abs[keep]
    pred_boxes_norm_xywh = pred_boxes_norm_xywh[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    # Early outs
    if pred_boxes_norm_xywh.numel() == 0 and tgt_boxes_norm_xywh.numel() == 0:
        return
    if tgt_boxes_norm_xywh.numel() == 0:
        # All kept preds become FP
        for i in range(len(pred_labels)):
            save_case(
                "FPs",
                img,
                pred_boxes_xyxy_abs[i],
                int(pred_labels[i]),
                float(pred_scores[i]),
                img_path,
                output_dir,
                label_to_name,
            )
        return
    if pred_boxes_norm_xywh.numel() == 0:
        # All targets become FN
        tgt_abs_xyxy = norm_xywh_to_abs_xyxy(tgt_boxes_norm_xywh, H, W)
        for j in range(len(tgt_labels)):
            save_case(
                "FNs",
                img,
                tgt_abs_xyxy[j],
                int(tgt_labels[j]),
                0,
                img_path,
                output_dir,
                label_to_name,
            )
        return

    # Compute IoU on normalized xyxy (scale doesn't matter)
    pred_xyxy_norm = norm_xywh_to_xyxy(pred_boxes_norm_xywh)  # [Np,4]
    tgt_xyxy_norm = norm_xywh_to_xyxy(tgt_boxes_norm_xywh)  # [Nt,4]
    ious = box_iou(pred_xyxy_norm, tgt_xyxy_norm)  # [Np,Nt]

    # Greedy one-to-one matching by pred score
    Np, Nt = ious.shape
    tgt_matched = torch.zeros(Nt, dtype=torch.bool)
    pred_matched = torch.zeros(Np, dtype=torch.bool)

    order = torch.argsort(pred_scores, descending=True)
    for pi in order.tolist():
        # mask by class and unmatched targets
        same_cls = tgt_labels == pred_labels[pi]
        cand = (~tgt_matched) & same_cls & (ious[pi] >= iou_thresh)
        if cand.any():
            # pick best IoU target among candidates
            ti = torch.argmax(ious[pi] * cand.float()).item()
            tgt_matched[ti] = True
            pred_matched[pi] = True

    # Save FPs and FNs
    for pi in torch.where(~pred_matched)[0].tolist():
        save_case(
            "FPs",
            img,
            pred_boxes_xyxy_abs[pi],
            int(pred_labels[pi]),
            float(pred_scores[pi]),
            img_path,
            output_dir,
            label_to_name,
        )

    tgt_abs_xyxy = norm_xywh_to_abs_xyxy(tgt_boxes_norm_xywh, H, W)
    for ti in torch.where(~tgt_matched)[0].tolist():
        save_case(
            "FNs",
            img,
            tgt_abs_xyxy[ti],
            int(tgt_labels[ti]),
            0,
            img_path,
            output_dir,
            label_to_name,
        )


def run(model, train_loader, val_loader, cfg: DictConfig) -> None:
    output_dir = Path(cfg.train.infer_path).parent / "check_errors"
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for loader in [train_loader, val_loader]:
        split_name = loader.dataset.mode

        for batch in tqdm(loader, desc=f"Processing {split_name}"):
            _, _, paths = batch

            img = cv2.imread(Path(cfg.train.data_path) / "images" / paths[0])

            label_path = Path(cfg.train.data_path) / "labels" / paths[0].with_suffix(".txt")
            targets = [{"boxes": [], "labels": []}]
            if label_path.exists() and label_path.stat().st_size > 1:
                targets_raw = np.loadtxt(label_path)
                if len(targets_raw.shape) == 1:
                    targets_raw = targets_raw[None, :]
                targets = [
                    {
                        "boxes": np.array(targets_raw[:, 1:5], dtype=np.float32),
                        "labels": np.array(targets_raw[:, 0], dtype=np.int64),
                    }
                ]

            preds = model(img)
            for pred in preds:
                pred["norm_boxes"] = abs_xyxy_to_norm_xywh(
                    pred["boxes"], img.shape[0], img.shape[1]
                )

            check_results(
                img=img,
                img_path=paths[0],
                preds=preds[0],
                targets=targets[0],
                iou_thresh=cfg.train.iou_thresh,
                conf_thresh=cfg.train.conf_thresh,
                output_dir=output_dir,
                label_to_name=cfg.train.label_to_name,
            )


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    model = Torch_model(
        model_name=cfg.model_name,
        model_path=Path(cfg.train.path_to_save) / "model.pt",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=cfg.train.conf_thresh,
        rect=cfg.export.dynamic_input,
        half=cfg.export.half,
        keep_ratio=cfg.train.keep_ratio,
    )
    base_loader = Loader(
        root_path=Path(cfg.train.data_path),
        img_size=tuple(cfg.train.img_size),
        batch_size=1,
        num_workers=cfg.train.num_workers,
        cfg=cfg,
        debug_img_processing=cfg.train.debug_img_processing,
    )
    train_loader, val_loader, _ = base_loader.build_dataloaders()

    run(model, train_loader, val_loader, cfg)


if __name__ == "__main__":
    main()
