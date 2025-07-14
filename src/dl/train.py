
import math
import time
from datetime import timedelta
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import sys

project_root = "path_to_your_project"
if project_root not in sys.path:
    sys.path.append(project_root)

from src.d_fine.dfine import build_loss, build_model, build_optimizer
from src.dl.dataset import Loader
from src.dl.utils import (
    calculate_remaining_time,
    filter_preds,
    get_vram_usage,
    log_metrics_locally,
    process_boxes,
    save_metrics,
    set_seeds,
    visualize,
    wandb_logger,
)
from src.dl.validator import Validator


class ModelEMA:
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
        self.cycler_pct_start = cfg.train.cycler_pct_start
        self.no_mosaic_epochs = cfg.train.mosaic_augs.no_mosaic_epochs
        self.ignore_background_epochs = cfg.train.ignore_background_epochs
        self.path_to_save = Path(cfg.train.path_to_save)
        self.to_visualize_eval = cfg.train.to_visualize_eval
        self.amp_enabled = cfg.train.amp_enabled
        self.clip_max_norm = cfg.train.clip_max_norm
        self.b_accum_steps = max(cfg.train.b_accum_steps, 1)
        self.keep_ratio = cfg.train.keep_ratio
        self.early_stopping = cfg.train.early_stopping
        self.use_wandb = cfg.train.use_wandb
        self.label_to_name = cfg.train.label_to_name
        self.num_labels = len(cfg.train.label_to_name)
        self.optimizer_step_called = False

        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers

        self.font_path = cfg.font_path

        self.max_visualize_images = cfg.train.max_visualize_images
        self.debug_img_processing = cfg.train.debug_img_processing

        self.debug_img_path = Path(self.cfg.train.debug_img_path)
        self.eval_preds_path = Path(self.cfg.train.eval_preds_path)
        self.init_dirs()

        if self.use_wandb:
            try:
                wandb_key = "XXXXXX"
                os.environ["WANDB_API_KEY"] = wandb_key
                logger.info(f"Using wandb key: {wandb_key}")
                wandb.login()
                wandb.init(
                    project=cfg.project_name,
                    name=cfg.exp,
                    save_code=False,
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                )
            except:
                logger.error("Failed to login to wandb. Please check your wandb key.")

        log_file = Path(cfg.train.path_to_save) / "train_log.txt"
        log_file.unlink(missing_ok=True)
        logger.add(log_file, format="{message}", level="INFO", rotation="10 MB")

        set_seeds(cfg.train.seed, cfg.train.cudnn_fixed)

        # load dataloader
        base_loader = Loader(
            root_path=Path(cfg.train.data_path),
            img_size=tuple(cfg.train.img_size),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            cfg=cfg,
            debug_img_processing=self.debug_img_processing,
        )
        self.train_loader, self.train4metric_loader,\
            self.val_loader, self.test_loader = base_loader.build_dataloaders()
                
        if self.ignore_background_epochs:
            self.train_loader.dataset.ignore_background = True

        # load model
        self.model = build_model(
            cfg.model_name,
            self.num_labels,
            cfg.train.device,
            img_size=cfg.train.img_size,
            pretrained_model_path=cfg.train.pretrained_model_path,
        )

        # load EMA model if needed
        self.ema_model = None
        if self.cfg.train.use_ema:
            logger.info("EMAed model will be evaluated and saved.")
            self.ema_model = ModelEMA(self.model, cfg.train.ema_momentum)

        # load loss
        self.loss_fn = build_loss(
            cfg.model_name, self.num_labels, label_smoothing=cfg.train.label_smoothing
        )

        # load optimizer
        self.optimizer = build_optimizer(
            self.model,
            lr=cfg.train.base_lr,
            backbone_lr=cfg.train.backbone_lr,
            betas=cfg.train.betas,
            weight_decay=cfg.train.weight_decay,
            base_lr=cfg.train.base_lr,
        )

        # set max lr
        self.max_lr = cfg.train.base_lr * 2
        if cfg.model_name in ["l", "x"]:  # per group max lr for big models
            self.max_lr = [
                cfg.train.backbone_lr * 2,
                cfg.train.backbone_lr * 2,
                cfg.train.base_lr * 2,
                cfg.train.base_lr * 2,
            ]
        
        # set scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_loader) // self.b_accum_steps,
            pct_start=self.cycler_pct_start,
            cycle_momentum=False,
        )

        # set amp
        if self.amp_enabled:
            self.scaler = GradScaler()
        
        if self.use_wandb:
            wandb.watch(self.model)

    def init_dirs(self):
        for path in [self.debug_img_path, self.eval_preds_path]:
            if path.exists():
                # 递归删除指定路径下的整个目录树
                rmtree(path)
            path.mkdir(exist_ok=True, parents=True)

        self.path_to_save.mkdir(exist_ok=True, parents=True)
        with open(self.path_to_save / "config.yaml", "w") as f:
            OmegaConf.save(config=self.cfg, f=f)

    def preds_postprocess(
        self,
        inputs,
        outputs,
        orig_sizes,
        num_top_queries=300,
        use_focal_loss=True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        boxes = process_boxes(
            boxes, inputs.shape[2:], orig_sizes, self.keep_ratio, inputs.device
        )  # B x TopQ x 4

        if use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
            labels = index - index // self.num_labels * self.num_labels
            index = index // self.num_labels
            boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > num_top_queries:
                scores, index = torch.topk(scores, num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(
                labels=lab.detach().cpu(), boxes=box.detach().cpu(), scores=sco.detach().cpu()
            )
            results.append(result)
        return results

    def gt_postprocess(self, inputs, targets, orig_sizes):
        """
        后处理输入的ground truth。
        """
        results = []
        for idx, target in enumerate(targets):
            lab = target["labels"]
            box = process_boxes(
                target["boxes"][None],
                inputs[idx].shape[1:],
                orig_sizes[idx][None],
                self.keep_ratio,
                inputs.device,
            )
            result = dict(labels=lab.detach().cpu(), boxes=box.squeeze(0).detach().cpu())
            results.append(result)
        return results

    @torch.no_grad()
    def get_preds_and_gt(
        self, val_loader: DataLoader
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        Outputs gt and preds. Each is a List of dicts. 1 dict = 1 image.
        """
        all_gt, all_preds = [], []
        model = self.model
        if self.ema_model:
            model = self.ema_model.model

        visualized_count = 0
        model.eval()
        for batchidx, (inputs, targets, img_paths) in enumerate(val_loader):
            inputs = inputs.to(self.device)
            if self.amp_enabled:
                with autocast(self.device, cache_enabled=True):
                    raw_res = model(inputs)
            else:
                raw_res = model(inputs)

            targets = [
                {
                    k: (v.to(self.device) if (v is not None and hasattr(v, "to")) else v)
                    for k, v in t.items()
                }
                for t in targets
            ]
            orig_sizes = (
                torch.stack([t["orig_size"] for t in targets], dim=0).float().to(self.device)
            )

            preds = self.preds_postprocess(inputs, raw_res, orig_sizes)
            gt = self.gt_postprocess(inputs, targets, orig_sizes)

            for pred_instance, gt_instance in zip(preds, gt):
                all_preds.append(pred_instance)
                all_gt.append(gt_instance)

            if self.to_visualize_eval and visualized_count < self.max_visualize_images:
                batch_visualize = min(len(img_paths), self.max_visualize_images - visualized_count)
                visualize(
                    img_paths[:batch_visualize],
                    gt,
                    filter_preds(preds, self.conf_thresh),
                    dataset_path=Path(self.cfg.train.data_path) / "images",
                    path_to_save=self.eval_preds_path,
                    font_path=self.font_path,
                    label_to_name=self.label_to_name,
                )
                visualized_count += batch_visualize

        return all_gt, all_preds

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
        if path_to_save:  # save plots of val and test
            validator.save_plots(path_to_save / "plots" / mode)
            logger.info(f"plots of validator were saved in {path_to_save / 'plots' / mode}.")
        return metrics

    def evaluate(
        self,
        val_loader: DataLoader,
        conf_thresh: float,
        iou_thresh: float,
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
            logger.info(f"Saving new best model to {self.path_to_save / 'model.pt'} 🔥")
            torch.save(model_to_save.state_dict(), self.path_to_save / "model.pt")
            self.early_stopping_steps = 0
        else:
            self.early_stopping_steps += 1
        
        return best_metric

    def train(self) -> None:
        best_val_metric = 0
        cur_iter = 0
        ema_iter = 0
        self.early_stopping_steps = 0
        one_epoch_time = None

        def optimizer_step(step_scheduler: bool):
            """
            Clip grads, optimizer step, scheduler step, zero grad, EMA model update
            """
            # 在嵌套函数中声明 ema_iter 不是当前作用域的变量，也不是全局变量，
            # 而是外层函数作用域中的变量，允许在此函数中修改外层函数的 ema_iter 变量。
            nonlocal ema_iter
            if self.amp_enabled:
                if self.clip_max_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.scaler.step(self.optimizer) # 优化器更新
                self.scaler.update()
            else:
                if self.clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.optimizer.step() # 优化器更新

            if step_scheduler:
                self.scheduler.step()
            
            self.optimizer.zero_grad()

            if self.ema_model:
                ema_iter += 1
                self.ema_model.update(ema_iter, self.model)

        # main training loop
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.model.train()
            self.loss_fn.train()
            losses = []

            # one epoch training
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch_idx, (inputs, targets, _) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}/{self.epochs}")
                    if inputs is None:
                        continue
                    cur_iter += 1

                    inputs = inputs.to(self.device)
                    targets = [
                        {
                            k: (v.to(self.device) if (v is not None and hasattr(v, "to")) else v)
                            for k, v in t.items()
                        }
                        for t in targets
                    ]

                    lr = self.optimizer.param_groups[-1]["lr"]

                    if self.amp_enabled:
                        with autocast(self.device, cache_enabled=True):
                            output = self.model(inputs, targets=targets)
                        with autocast(self.device, enabled=False):
                            loss_dict = self.loss_fn(output, targets)
                        loss = sum(loss_dict.values()) / self.b_accum_steps
                        self.scaler.scale(loss).backward()
                    else:
                        output = self.model(inputs, targets=targets)
                        loss_dict = self.loss_fn(output, targets)
                        loss = sum(loss_dict.values()) / self.b_accum_steps
                        loss.backward()

                    if (batch_idx + 1) % self.b_accum_steps == 0:
                        optimizer_step(step_scheduler=True)

                    losses.append(loss.item())

                    tepoch.set_postfix(
                        # 计算平均损失并乘以累积步数，反映当前训练损失值
                        loss=np.mean(losses) * self.b_accum_steps,
                        # 调用函数计算并显示预计剩余时间
                        eta=calculate_remaining_time(
                            one_epoch_time,
                            epoch_start_time,
                            epoch,
                            self.epochs,
                            cur_iter,
                            len(self.train_loader),
                        ),
                        # 显示当前显存使用百分比
                        GPURamUsage=f"{get_vram_usage()}%",
                    )

            # Final update for any leftover gradients from an incomplete accumulation step
            if (batch_idx + 1) % self.b_accum_steps != 0:
                optimizer_step(step_scheduler=False)

            if self.use_wandb:
                wandb.log({"lr": lr, "epoch": epoch})

            train_metrics = {}
            if self.train4metric_loader:
                train_metrics = self.evaluate(
                    val_loader=self.train4metric_loader,
                    conf_thresh=self.conf_thresh,
                    iou_thresh=self.iou_thresh,
                    path_to_save=None,
                )

            val_metrics = {}
            if self.val_loader:
                val_metrics = self.evaluate(
                    val_loader=self.val_loader,
                    conf_thresh=self.conf_thresh,
                    iou_thresh=self.iou_thresh,
                    path_to_save=None,
                )

            if val_metrics:
                best_val_metric = self.save_model(val_metrics, best_val_metric)
            elif train_metrics:
                best_val_metric = self.save_model(train_metrics, best_val_metric)
            else:
                logger.info("No validation metrics to save model, using the loss decision metric instead.")
                decision_metrics = 1. / (np.mean(losses) * self.b_accum_steps+1.)
                best_val_metric = self.save_model(decision_metrics, best_val_metric)
            
            # wandb_logger
            save_metrics(
                train_metrics,
                val_metrics,
                np.mean(losses) * self.b_accum_steps,
                epoch,
                path_to_save=None,
                use_wandb=self.use_wandb,
            )

            # 当训练轮数（epoch）接近最后 no_mosaic_epochs 轮时，
            # 且当前数据集启用了马赛克增强（mosaic_prob > 0），
            # 则关闭马赛克增强操作。
            if (
                epoch >= self.epochs - self.no_mosaic_epochs
                and self.train_loader.dataset.mosaic_prob
            ):
                self.train_loader.dataset.close_mosaic()

            # 在指定训练轮次后开启背景图像的使用
            if epoch == self.ignore_background_epochs:
                self.train_loader.dataset.ignore_background = False
                logger.info("Including background images")

            one_epoch_time = time.time() - epoch_start_time

            if self.early_stopping and self.early_stopping_steps >= self.early_stopping:
                logger.info("Early stopping")
                break


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
        logger.info("Evaluating the best model...")
        model = build_model(
            cfg.model_name,
            len(cfg.train.label_to_name),
            cfg.train.device,
            img_size=cfg.train.img_size,
        )
        model.load_state_dict(
            torch.load(Path(cfg.train.path_to_save) / "model.pt", weights_only=True)
        )

        if trainer.ema_model:
            trainer.ema_model.model = model
        else:
            trainer.model = model

        val_metrics = {}
        if trainer.val_loader:
            val_metrics = trainer.evaluate(
                val_loader=trainer.val_loader,
                conf_thresh=trainer.conf_thresh,
                iou_thresh=trainer.iou_thresh,
                path_to_save=Path(cfg.train.path_to_save),
                mode="val",
            )

        if cfg.train.use_wandb:
            wandb_logger(None, val_metrics, epoch=cfg.train.epochs + 1, mode="val")

        test_metrics = {}
        if trainer.test_loader:
            test_metrics = trainer.evaluate(
                val_loader=trainer.test_loader,
                conf_thresh=trainer.conf_thresh,
                iou_thresh=trainer.iou_thresh,
                path_to_save=Path(cfg.train.path_to_save),
                mode="test",
            )
            if cfg.train.use_wandb:
                wandb_logger(None, test_metrics, epoch=-1, mode="test")

        if val_metrics or test_metrics:
            log_metrics_locally(
                all_metrics={"val": val_metrics, "test": test_metrics},
                path_to_save=Path(cfg.train.path_to_save),
                epoch=0,
            )
        else:
            logger.info("No validation or test metrics to log and save.")

        training_seconds = time.time() - t_start
        training_time = timedelta(seconds=int(training_seconds))
        logger.info(f"Full training time: {training_time} / {training_seconds:.2f} seconds")

        if cfg.train.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
