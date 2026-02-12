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
from src.dl.utils import get_latest_experiment_name, process_boxes, process_masks, visualize
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
    compute_maps: bool,
    to_draw_gt: bool,
):
    logger.info(f"Testing {name} model")
    latency = []
    batch = 0
    all_gt = []
    all_preds = []

    output_path = output_path / name
    output_path.mkdir(exist_ok=True, parents=True)

    # Warmup iterations
    first_batch = next(iter(test_loader))
    warmup_img = cv2.imread(str(data_path / "images" / first_batch[2][0]))
    for _ in range(10):
        _ = model(warmup_img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _, targets, img_paths in tqdm(test_loader, total=len(test_loader)):
        for img_path, targets in zip(img_paths, targets):
            img = cv2.imread(str(data_path / "images" / img_path))

            # laod GT
            gt_boxes = process_boxes(
                targets["boxes"][None],
                processed_size,
                targets["orig_size"][None],
                keep_ratio,
                device,
            )[batch].cpu()

            gt_labels = targets["labels"]

            if "masks" in targets:
                gt_masks = process_masks(
                    targets["masks"][None], processed_size, targets["orig_size"][None], keep_ratio
                )[batch].cpu()

            gt_dict = {"boxes": gt_boxes, "labels": gt_labels.int()}
            if "masks" in targets:
                gt_dict["masks"] = gt_masks
            all_gt.append(gt_dict)

            # inference with CUDA synchronization for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model_preds = model(img)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency.append((time.perf_counter() - t0) * 1000)

            # prepare preds
            pred_dict = {
                "boxes": model_preds[batch]["boxes"].cpu(),
                "labels": model_preds[batch]["labels"].cpu(),
                "scores": model_preds[batch]["scores"].cpu(),
            }
            if "masks" in model_preds[batch]:
                pred_dict["masks"] = model_preds[batch]["masks"].cpu()

            all_preds.append(pred_dict)

            gt_to_vis = [gt_dict]
            if not to_draw_gt:
                gt_to_vis = [{"boxes": [], "labels": []}]

            if to_visualize:
                visualize(
                    img_paths,
                    gt_to_vis,
                    [pred_dict],
                    dataset_path=data_path / "images",
                    path_to_save=output_path,
                    label_to_name=label_to_name,
                )

    validator = Validator(
        all_gt,
        all_preds,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        label_to_name=label_to_name,
        compute_maps=compute_maps,  # as inference done with a conf threshold, mAPs don't make much sense
    )

    metrics = validator.compute_metrics(extended=False)
    metrics["latency"] = round(np.mean(latency[1:]), 1)
    return metrics


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    conf_thresh = 0.5
    iou_thresh = 0.5
    compute_maps = False
    to_visualize = True
    to_draw_gt = True

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
        enable_mask_head=cfg.task == "segment",
    )

    trt_model = TRT_model(
        model_path=Path(cfg.train.path_to_save) / "model.engine",
        n_outputs=len(cfg.train.label_to_name),
        conf_thresh=conf_thresh,
        rect=False,
        half=cfg.export.half,
        keep_ratio=cfg.train.keep_ratio,
    )

    ov_model = OV_model(
        model_path=Path(cfg.train.path_to_save) / "model.xml",
        conf_thresh=conf_thresh,
        rect=cfg.export.dynamic_input,
        half=cfg.export.half,
        keep_ratio=cfg.train.keep_ratio,
        max_batch_size=1,
    )

    onnx_model = ONNX_model(
        model_path=Path(cfg.train.path_to_save) / "model.onnx",
        n_outputs=len(cfg.train.label_to_name),
        conf_thresh=conf_thresh,
        rect=False,
        half=False,
        keep_ratio=cfg.train.keep_ratio,
    )

    ov_int8_path = Path(cfg.train.path_to_save) / "model_int8.xml"
    if ov_int8_path.exists():
        ov_int8_model = OV_model(
            model_path=ov_int8_path,
            conf_thresh=conf_thresh,
            rect=cfg.export.dynamic_input,
            half=cfg.export.half,
            keep_ratio=cfg.train.keep_ratio,
            max_batch_size=1,
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
    if ov_int8_path.exists():
        models["OpenVINO INT8"] = ov_int8_model

    for model_name, model in models.items():
        all_metrics[model_name] = test_model(
            val_loader,
            data_path,
            Path(cfg.train.bench_img_path),
            model,
            model_name,
            conf_thresh,
            iou_thresh,
            to_visualize=to_visualize,
            processed_size=tuple(cfg.train.img_size),
            keep_ratio=cfg.train.keep_ratio,
            device=cfg.train.device,
            label_to_name=cfg.train.label_to_name,
            compute_maps=compute_maps,
            to_draw_gt=to_draw_gt,
        )

    metrics = pd.DataFrame.from_dict(all_metrics, orient="index").round(3)
    metrics.to_csv(Path(cfg.train.path_to_save) / "bench_metrics.csv")
    tabulated_data = tabulate(metrics, headers="keys", tablefmt="pretty", showindex=True)
    print("\n" + tabulated_data)


if __name__ == "__main__":
    main()
