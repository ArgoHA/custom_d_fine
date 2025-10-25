import time
from pathlib import Path
from shutil import rmtree
from typing import Dict, Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dl.dataset import CustomDataset, Loader
from src.dl.utils import get_latest_experiment_name, process_boxes, vis_one_box
from src.dl.validator import Validator
from src.infer.onnx_model import ONNX_model
from src.infer.ov_model import OV_model
from src.infer.torch_model import Torch_model
from src.infer.trt_model import TRT_model

torch.multiprocessing.set_sharing_strategy("file_system")


class BenchLoader(Loader):
    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        val_ds = CustomDataset(
            self.img_size,
            self.root_path,
            self.splits["val"],
            self.debug_img_processing,
            mode="bench",
            cfg=self.cfg,
        )

        test_loader = None
        if len(self.splits["test"]):
            test_ds = CustomDataset(
                self.img_size,
                self.root_path,
                self.splits["test"],
                self.debug_img_processing,
                mode="bench",
                cfg=self.cfg,
            )
            test_loader = self._build_dataloader_impl(test_ds)

        val_loader = self._build_dataloader_impl(val_ds)
        return val_loader, test_loader


def visualize(
    img,
    gt_boxes,
    pred_boxes,
    gt_labels,
    pred_labels,
    pred_scores,
    output_path,
    img_path,
    label_to_name,
):
    for gt_box, gt_label in zip(gt_boxes, gt_labels):
        vis_one_box(img, gt_box, gt_label, mode="gt", label_to_name=label_to_name)

    for pred_box, pred_label, score in zip(pred_boxes, pred_labels, pred_scores):
        vis_one_box(
            img,
            pred_box,
            pred_label,
            mode="pred",
            label_to_name=label_to_name,
            score=score,
        )

    cv2.imwrite((str(f"{output_path / Path(img_path).stem}.jpg")), img)


def test_model(
    test_loader: DataLoader,
    data_path: Path,
    output_path: Path,
    model,
    name: str,
    conf_thresh: float,
    iou_thresh: float,
    to_visualize: bool,
    processed_size: Tuple[int, int],
    keep_ratio: bool,
    device: str,
    label_to_name: Dict[int, str],
):
    logger.info(f"Testing {name} model")
    gt = []
    preds = []
    latency = []
    batch = 0

    output_path = output_path / name
    output_path.mkdir(exist_ok=True, parents=True)

    for _, targets, img_paths in tqdm(test_loader, total=len(test_loader)):
        for img_path, targets in zip(img_paths, targets):
            gt_boxes = process_boxes(
                targets["boxes"][None],
                processed_size,
                targets["orig_size"][None],
                keep_ratio,
                device,
            )[batch].cpu()
            gt_labels = targets["labels"]

            img = cv2.imread(str(data_path / "images" / img_path))
            t0 = time.perf_counter()
            model_preds = model(img)
            latency.append((time.perf_counter() - t0) * 1000)

            gt.append({"boxes": gt_boxes, "labels": gt_labels.int()})
            preds.append(
                {
                    "boxes": torch.tensor(model_preds[batch]["boxes"]),
                    "labels": torch.tensor(model_preds[batch]["labels"]),
                    "scores": torch.tensor(model_preds[batch]["scores"]),
                }
            )

            if to_visualize:
                visualize(
                    img=img,
                    gt_boxes=gt_boxes,
                    pred_boxes=model_preds[batch]["boxes"],
                    gt_labels=gt_labels,
                    pred_labels=model_preds[batch]["labels"],
                    pred_scores=model_preds[batch]["scores"],
                    output_path=output_path,
                    img_path=img_path,
                    label_to_name=label_to_name,
                )

    validator = Validator(
        gt,
        preds,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        label_to_name=label_to_name,
    )
    metrics = validator.compute_metrics(extended=False)

    # as inference done with a conf threshold, mAPs don't make much sense
    metrics.pop("mAP_50")
    metrics.pop("mAP_50_95")
    metrics["latency"] = round(np.mean(latency[1:]), 1)
    return metrics


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    conf_thresh = 0.5
    iou_thresh = 0.5

    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    torch_model = Torch_model(
        model_name=cfg.model_name,
        model_path=Path(cfg.train.path_to_save) / "model.pt",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=conf_thresh,
        rect=cfg.export.dynamic_input,
        half=cfg.export.half,
        keep_ratio=cfg.train.keep_ratio,
    )

    trt_model = TRT_model(
        model_path=Path(cfg.train.path_to_save) / "model.engine",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=conf_thresh,
        rect=False,
        half=cfg.export.half,
        keep_ratio=cfg.train.keep_ratio,
    )

    ov_model = OV_model(
        model_path=Path(cfg.train.path_to_save) / "model.xml",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=conf_thresh,
        rect=cfg.export.dynamic_input,
        half=cfg.export.half,
        keep_ratio=cfg.train.keep_ratio,
        max_batch_size=1,
    )

    onnx_model = ONNX_model(
        model_path=Path(cfg.train.path_to_save) / "model.onnx",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=conf_thresh,
        rect=cfg.export.dynamic_input,
        half=cfg.export.half,
        keep_ratio=cfg.train.keep_ratio,
    )

    data_path = Path(cfg.train.data_path)
    val_loader, test_loader = BenchLoader(
        root_path=data_path,
        img_size=tuple(cfg.train.img_size),
        batch_size=1,
        num_workers=1,
        cfg=cfg,
        debug_img_processing=False,
    ).build_dataloaders()

    output_path = Path(cfg.train.bench_img_path)
    if output_path.exists():
        rmtree(output_path)

    all_metrics = {}
    models = {
        "OpenVINO": ov_model,
        "Torch": torch_model,
        "TensorRT": trt_model,
        "ONNX": onnx_model,
    }
    for model_name, model in models.items():
        all_metrics[model_name] = test_model(
            val_loader,
            data_path,
            Path(cfg.train.bench_img_path),
            model,
            model_name,
            conf_thresh,
            iou_thresh,
            to_visualize=True,
            processed_size=tuple(cfg.train.img_size),
            keep_ratio=cfg.train.keep_ratio,
            device=cfg.train.device,
            label_to_name=cfg.train.label_to_name,
        )

    metrcs = pd.DataFrame.from_dict(all_metrics, orient="index")
    tabulated_data = tabulate(metrcs.round(4), headers="keys", tablefmt="pretty", showindex=True)
    print("\n" + tabulated_data)


if __name__ == "__main__":
    main()
