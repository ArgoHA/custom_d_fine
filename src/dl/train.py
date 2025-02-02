import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.d_fine.dfine import build_loss, build_model, build_optimizer
from src.d_fine.postprocess import DFINEPostProcessor
from src.dl.dataset import Loader
from src.dl.utils import (
    calculate_remaining_time,
    filter_preds,
    log_metrics_locally,
    norm_xywh_to_abs_xyxy,
    save_metrics,
    set_seeds,
    visualize,
    wandb_logger,
)
from src.dl.validator import Validator
from src.ptypes import num_labels


class ema_model:
    def __init__(self, student, ema_momentum):
        self.model = deepcopy(student).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.ema_scheduler = lambda x: ema_momentum * (1 - math.exp(-x / 2000))

    def update(self, iters, student):
        student = student.state_dict()
        with torch.no_grad():
            momentum = self.ema_scheduler(iters)
            for name, param in self.model.state_dict().items():
                if param.dtype.is_floating_point:
                    param *= momentum
                    param += (1.0 - momentum) * student[name].detach()


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = cfg.train.device
        self.conf_thresh = cfg.train.conf_thresh
        self.iou_thresh = cfg.train.iou_thresh
        self.epochs = cfg.train.epochs
        self.no_mosaic_epochs = cfg.train.mosaic_augs.no_mosaic_epochs
        self.warmup_epochs = cfg.train.warmup_epochs
        self.path_to_save = Path(cfg.train.path_to_save)
        self.to_visualize_eval = cfg.train.to_visualize_eval
        self.amp_enabled = cfg.train.amp_enabled
        self.clip_max_norm = cfg.train.clip_max_norm
        self.b_accum_steps = max(cfg.train.b_accum_steps, 1)

        wandb.init(
            project=cfg.project_name,
            name=cfg.exp,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

        log_file = Path(cfg.train.path_to_save) / "train_log.txt"
        log_file.unlink(missing_ok=True)
        logger.add(
            log_file,
            format="{message}",
            level="INFO",
            rotation="10 MB",
        )

        set_seeds(cfg.train.seed, cfg.train.cudnn_fixed)

        base_loader = Loader(
            root_path=Path(cfg.train.data_path),
            img_size=tuple(cfg.train.img_size),
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            cfg=cfg,
            debug_img_processing=cfg.train.debug_img_processing,
        )
        self.train_loader, self.val_loader, self.test_loader = base_loader.build_dataloaders()
        if self.warmup_epochs:
            self.train_loader.dataset.ignore_background = True

        self.model = build_model(
            cfg.model_name,
            num_labels,
            cfg.train.device,
            pretrained_model_path=Path(cfg.train.pretrained_model_path),
        )

        self.ema_model = None
        if self.cfg.train.use_ema:
            logger.info("EMA model will be evaluated and saved")
            self.ema_model = ema_model(self.model, cfg.train.ema_momentum)

        self.loss_fn = build_loss(cfg.model_name, num_labels)

        self.optimizer = build_optimizer(
            self.model,
            lr=cfg.train.base_lr,
            betas=cfg.train.betas,
            weight_decay=cfg.train.weight_decay,
            base_lr=cfg.train.base_lr,
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=cfg.train.max_lr,
            epochs=cfg.train.epochs,
            steps_per_epoch=len(self.train_loader) // self.b_accum_steps,
            pct_start=cfg.train.cycler_pct_start,
        )

        if self.amp_enabled:
            self.scaler = GradScaler()

        self.postprocess = DFINEPostProcessor(num_classes=num_labels, use_focal_loss=True)

        wandb.watch(self.model)

    @torch.no_grad
    def get_preds_and_gt(
        self, val_loader: DataLoader
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        Outputs gt and preds. Each is a List of dicts. 1 dict = 1 image.

        """
        gt, preds = [], []
        model = self.model
        if self.ema_model:
            model = self.ema_model.model

        model.eval()
        for idx, (inputs, targets, img_paths) in enumerate(val_loader):
            inputs = inputs.to(self.device)
            raw_res = model(inputs)

            targets = [
                {
                    k: (v.to(self.device) if (v is not None and hasattr(v, "to")) else v)
                    for k, v in t.items()
                }
                for t in targets
            ]
            orig_target_sizes = (
                torch.stack([t["orig_size"] for t in targets], dim=0).float().to(self.device)
            )

            output = self.postprocess(raw_res, orig_target_sizes)

            for i in range(len(targets)):
                gt.append(
                    {
                        "boxes": torch.tensor(
                            norm_xywh_to_abs_xyxy(
                                targets[i]["boxes"].cpu(),
                                orig_target_sizes[i][1].cpu(),
                                orig_target_sizes[i][0].cpu(),
                            ),
                        ).to(self.device),
                        "labels": targets[i]["labels"].int(),  # shape [num_gt_i]
                    }
                )
                preds.append(
                    {
                        "boxes": output[i]["boxes"],
                        "labels": output[i]["labels"].int(),
                        "scores": output[i]["scores"],
                    }
                )

            if not idx and self.to_visualize_eval:
                batch_gt = gt[-len(targets) :]  # the last N items appended
                batch_preds = preds[-len(targets) :]  # the last N items appended

                visualize(
                    img_paths,
                    batch_gt,
                    filter_preds(batch_preds, self.conf_thresh),
                    dataset_path=Path(self.cfg.train.data_path) / "images",
                    path_to_save=Path(self.cfg.train.root) / "output" / "eval_preds",
                )

        return gt, preds

    @staticmethod
    def get_metrics(
        gt,
        preds,
        conf_thresh: float,
        iou_thresh: float,
        path_to_save=None,
        mode=None,
    ):
        validator = Validator(
            gt,
            preds,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
        metrics = validator.compute_metrics()
        if path_to_save:  # val and test
            validator.save_plots(path_to_save / "plots" / mode)
        return metrics

    def evaluate(
        self,
        val_loader: DataLoader,
        conf_thresh,
        iou_thresh,
        path_to_save: Path,
        mode: str = None,
    ) -> Dict[str, float]:
        gt, preds = self.get_preds_and_gt(val_loader=val_loader)
        metrics = self.get_metrics(
            gt, preds, conf_thresh, iou_thresh, path_to_save=path_to_save, mode=mode
        )
        return metrics

    def save_model(self, metrics, best_metric):
        model_to_save = self.model
        if self.ema_model:
            model_to_save = self.ema_model.model

        self.path_to_save.mkdir(parents=True, exist_ok=True)
        torch.save(model_to_save.state_dict(), self.path_to_save / "last.pt")

        decision_metric = (metrics["mAP_50"] + metrics["f1"]) / 2
        if decision_metric > best_metric:
            best_metric = decision_metric
            logger.info("Saving new best model🔥")
            torch.save(model_to_save.state_dict(), self.path_to_save / "model.pt")
        return best_metric

    def _pred_and_loss(self, inputs, targets):
        output = self.model(inputs, targets=targets)
        loss_dict = self.loss_fn(output, targets)
        loss = sum(loss_dict.values())
        return loss

    def train(self) -> None:
        best_metric = 0
        cur_iter = 0
        one_epoch_time = None
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.model.train()
            losses = []

            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch_idx, (inputs, targets, _) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}/{self.epochs}")
                    cur_iter += 1
                    if inputs is None:
                        continue

                    inputs = inputs.to(self.device)
                    targets = [
                        {
                            k: (v.to(self.device) if (v is not None and hasattr(v, "to")) else v)
                            for k, v in t.items()
                        }
                        for t in targets
                    ]

                    if batch_idx % self.b_accum_steps == 0:
                        self.optimizer.zero_grad(set_to_none=True)
                    lr = self.optimizer.param_groups[0]["lr"]

                    if self.amp_enabled:
                        with autocast(self.device, dtype=torch.bfloat16):
                            loss = self._pred_and_loss(inputs, targets)

                        self.scaler.scale(loss).backward()

                        if self.clip_max_norm:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.clip_max_norm
                            )

                        if (batch_idx + 1) % self.b_accum_steps == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.scheduler.step()

                    else:
                        loss = self._pred_and_loss(inputs, targets)
                        loss.backward()

                        if self.clip_max_norm:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.clip_max_norm
                            )

                        if (batch_idx + 1) % self.b_accum_steps == 0:
                            self.optimizer.step()
                            self.scheduler.step()

                    if self.ema_model and batch_idx % self.b_accum_steps == 0:
                        self.ema_model.update(cur_iter, self.model)

                    losses.append(loss.item())

                    tepoch.set_postfix(
                        loss=np.mean(losses),
                        eta=calculate_remaining_time(
                            one_epoch_time,
                            epoch_start_time,
                            epoch,
                            self.epochs,
                            cur_iter,
                            len(self.train_loader),
                        ),
                    )

            wandb.log({"lr": lr, "epoch": epoch})

            metrics = self.evaluate(
                val_loader=self.val_loader,
                conf_thresh=self.conf_thresh,
                iou_thresh=self.iou_thresh,
                path_to_save=None,
            )

            best_metric = self.save_model(metrics, best_metric)
            save_metrics({}, metrics, np.mean(losses), epoch, path_to_save=None)

            if (
                epoch >= self.epochs - self.no_mosaic_epochs
                and self.train_loader.dataset.mosaic_prob
            ):
                self.train_loader.dataset.close_mosaic()

            if epoch == self.warmup_epochs:
                self.train_loader.dataset.ignore_background = False
                logger.info("Including background images")

            one_epoch_time = time.time() - epoch_start_time


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)

    try:
        t_start = time.time()
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(e)
    finally:
        logger.info("Evaluating best model...")
        model = build_model(cfg.model_name, num_labels, cfg.train.device)
        model.load_state_dict(
            torch.load(Path(cfg.train.path_to_save) / "model.pt", weights_only=True)
        )
        if trainer.ema_model:
            trainer.ema_model.model = model
        else:
            trainer.model = model

        val_metrics = trainer.evaluate(
            val_loader=trainer.val_loader,
            conf_thresh=trainer.conf_thresh,
            iou_thresh=trainer.iou_thresh,
            path_to_save=Path(cfg.train.path_to_save),
            mode="val",
        )

        test_metrics = {}
        if trainer.test_loader:
            test_metrics = trainer.evaluate(
                val_loader=trainer.test_loader,
                conf_thresh=trainer.conf_thresh,
                iou_thresh=trainer.iou_thresh,
                path_to_save=Path(cfg.train.path_to_save),
                mode="test",
            )
            wandb_logger(None, test_metrics, epoch=-1, mode="test")

        log_metrics_locally(
            all_metrics={"val": val_metrics, "test": test_metrics},
            path_to_save=Path(cfg.train.path_to_save),
            epoch=0,
        )
        logger.info(f"Full training time: {(time.time() - t_start) / 60 / 60:.2f} hours")


if __name__ == "__main__":
    main()
