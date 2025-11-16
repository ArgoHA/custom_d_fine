from pathlib import Path
from typing import Dict, List

import hydra
import nncf
import numpy as np
import openvino as ov
import torch
from loguru import logger
from omegaconf import DictConfig

from src.dl.dataset import Loader
from src.dl.train import Trainer
from src.dl.utils import get_latest_experiment_name
from src.dl.validator import Validator


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    """
    Run INT8 quantization with accuracy control on OpenVINO IR model using f1 score.
    Expects FP32 IR at <cfg.train.path_to_save>/model.xml.
    """
    # Resolve latest experiment (same as in train/export scripts)
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    save_dir = Path(cfg.train.path_to_save)
    fp32_xml_path = save_dir / "model.xml"
    assert fp32_xml_path.exists(), f"FP32 OpenVINO model not found: {fp32_xml_path}"

    logger.info(f"Using FP32 OpenVINO model: {fp32_xml_path}")

    #  Data: val loader (used both for calibration & validation)
    base_loader = Loader(
        root_path=Path(cfg.train.data_path),
        img_size=tuple(cfg.train.img_size),
        batch_size=1,
        num_workers=cfg.train.num_workers,
        cfg=cfg,
        debug_img_processing=False,
    )
    _, val_loader, _ = base_loader.build_dataloaders()
    logger.info(f"Val images: {len(val_loader.dataset)}")

    label_to_name = cfg.train.label_to_name
    num_labels = len(label_to_name)
    keep_ratio = cfg.train.keep_ratio
    conf_thresh = cfg.train.conf_thresh
    iou_thresh = cfg.train.iou_thresh

    # OpenVINO model
    core = ov.Core()
    model = core.read_model(fp32_xml_path)

    # NNCF datasets
    # DataLoader returns: (images, targets, img_paths)
    def transform_fn(data_item):
        images, _, _ = data_item
        # NNCF expects numpy inputs for OpenVINO model
        return images.numpy().astype(np.float32)

    calibration_dataset = nncf.Dataset(val_loader, transform_fn)
    validation_dataset = nncf.Dataset(val_loader, transform_fn)

    def validate(compiled_model: ov.CompiledModel, validation_loader) -> float:
        """
        compiled_model: openvino.CompiledModel
        validation_loader: torch.utils.data.DataLoader
        returns: mAP_50 (float)
        """
        output_logits = compiled_model.output("logits")
        output_boxes = compiled_model.output("boxes")

        all_preds: List[Dict[str, torch.Tensor]] = []
        all_gt: List[Dict[str, torch.Tensor]] = []

        for inputs, targets, _ in validation_loader:
            # inputs: [B, C, H, W], torch on CPU
            inputs_np = inputs.numpy().astype(np.float32)
            ov_res = compiled_model(inputs_np)

            logits_np = ov_res[output_logits]
            boxes_np = ov_res[output_boxes]

            logits = torch.from_numpy(logits_np)
            boxes = torch.from_numpy(boxes_np)

            outputs = {"pred_logits": logits, "pred_boxes": boxes}

            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).float()

            preds = Trainer.preds_postprocess(
                inputs,
                outputs,
                orig_sizes,
                num_labels=num_labels,
                keep_ratio=keep_ratio,
            )
            gt = Trainer.gt_postprocess(
                inputs,
                targets,
                orig_sizes,
                keep_ratio=keep_ratio,
            )

            all_preds.extend(preds)
            all_gt.extend(gt)

        validator = Validator(
            all_gt,
            all_preds,
            label_to_name=label_to_name,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
        metrics = validator.compute_metrics(extended=False)
        f1_score = metrics["f1"]
        logger.info(f"Validation F1-score: {f1_score:.4f}")
        return f1_score

    # Run quantization with accuracy control
    max_drop = getattr(cfg.export, "ov_int8_max_drop", 0.03)
    subset_size = getattr(cfg.export, "ov_int8_subset_size", 300)

    logger.info(
        f"Starting INT8 quantization with accuracy control: "
        f"max_drop={max_drop}, subset_size={subset_size}"
    )

    quantized_model = nncf.quantize_with_accuracy_control(
        model,
        calibration_dataset=calibration_dataset,
        validation_dataset=validation_dataset,
        validation_fn=validate,
        max_drop=max_drop,
        drop_type=nncf.DropType.ABSOLUTE,
        preset=nncf.QuantizationPreset.MIXED,  # better accuracy than PERFORMANCE
        subset_size=subset_size,
    )

    # Save INT8 model
    int8_xml_path = save_dir / "model_int8.xml"
    int8_xml_path.parent.mkdir(parents=True, exist_ok=True)

    # Save without additional FP16 compression
    ov.save_model(quantized_model, str(int8_xml_path), compress_to_fp16=False)
    logger.info(f"INT8 model with accuracy control saved to: {int8_xml_path}")


if __name__ == "__main__":
    main()
