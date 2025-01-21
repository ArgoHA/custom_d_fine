import time
from pathlib import Path
from typing import Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dl.dataset import CustomDataset, Loader
from src.dl.utils import norm_xywh_to_abs_xyxy
from src.dl.validator import Validator
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


def visualize(img, gt_boxes, pred_boxes, output_path, img_path):
    for gt_box in gt_boxes:
        cv2.rectangle(
            img,
            (int(gt_box[0]), int(gt_box[1])),
            (int(gt_box[2]), int(gt_box[3])),
            (0, 255, 0),
            2,
        )

    for pred_box in pred_boxes:
        cv2.rectangle(
            img,
            (int(pred_box[0]), int(pred_box[1])),
            (int(pred_box[2]), int(pred_box[3])),
            (255, 0, 0),
            2,
        )

    cv2.imwrite((str(f"{output_path / Path(img_path).stem}.jpg")), img)


def test_model(
    test_loader: DataLoader,
    data_path: Path,
    root_path: Path,
    model,
    name: str,
    conf_thresh: float,
    iou_thresh: float,
    single_class: bool,
    to_visualize: bool,
):
    logger.info(f"Testing {name} model")
    gt = []
    preds = []
    latency = []

    output_path = root_path / Path(f"output/bench_imgs/{name}")
    output_path.mkdir(exist_ok=True, parents=True)

    for _, targets, img_paths in tqdm(test_loader, total=len(test_loader)):
        for img_path, targets in zip(img_paths, targets):
            gt_boxes = torch.tensor(
                norm_xywh_to_abs_xyxy(
                    targets["boxes"],
                    targets["orig_size"][1],
                    targets["orig_size"][0],
                )
            )
            gt_labels = targets["labels"]

            t0 = time.perf_counter()
            img = cv2.imread(str(data_path / "images" / img_path))
            model_preds = model(img)
            latency.append((time.perf_counter() - t0) * 1000)

            gt.append({"boxes": gt_boxes, "labels": gt_labels.int()})
            preds.append(
                {
                    "boxes": torch.tensor(model_preds["boxes"]),
                    "labels": torch.tensor(model_preds["class_ids"]).int(),
                    "scores": torch.tensor(model_preds["scores"]),
                }
            )

            if to_visualize:
                visualize(img, gt_boxes, model_preds["boxes"], output_path, img_path)

    validator = Validator(
        gt,
        preds,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        single_class=single_class,
    )
    metrics = validator.compute_metrics(extended=False)
    metrics["latency"] = round(np.mean(latency[1:]), 1)
    return metrics


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    conf_thresh = 0.5
    iou_thresh = 0.5

    trt_model = TRT_model(
        model_path=Path(cfg.train.path_to_save) / "model.engine",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        rect=False,
        half=cfg.export.half,
        keep_ratio=cfg.train.keep_ratio,
    )

    torch_model = Torch_model(
        model_name=cfg.model_name,
        model_path=Path(cfg.train.path_to_save) / "model.pt",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        rect=cfg.export.dynamic_input,
        half=cfg.export.half,
        keep_ratio=cfg.train.keep_ratio,
        # device="cpu",
    )

    single_class = False
    data_path = Path(cfg.train.data_path)
    val_loader, test_loader = BenchLoader(
        root_path=data_path,
        img_size=tuple(cfg.train.img_size),
        batch_size=1,
        num_workers=1,
        cfg=cfg,
        debug_img_processing=False,
    ).build_dataloaders()

    all_metrics = {}
    models = {
        "trt_model": trt_model,
        "torch": torch_model,
    }
    for model_name, model in models.items():
        all_metrics[model_name] = test_model(
            val_loader,
            data_path,
            Path(cfg.train.root),
            model,
            model_name,
            conf_thresh,
            iou_thresh,
            single_class=single_class,
            to_visualize=True,
        )

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    print(metrics_df)


if __name__ == "__main__":
    main()
