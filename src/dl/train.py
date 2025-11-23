import math
import time
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.d_fine.dfine import build_loss, build_model, build_optimizer
from src.dl.dataset import Loader
from src.dl.utils import (
    calculate_remaining_time,
    filter_preds,
    get_latest_experiment_name,
    get_vram_usage,
    log_metrics_locally,
    process_boxes,
    save_metrics,
    set_seeds,
    visualize,
    wandb_logger,
    setup_ddp,
    is_main_process,
    cleanup_ddp,
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
        self.device = torch.device(cfg.train.device)
        self.conf_thresh = cfg.train.conf_thresh
        self.iou_thresh = cfg.train.iou_thresh
        self.epochs = cfg.train.epochs
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

        self.debug_img_path = Path(self.cfg.train.debug_img_path)
        self.eval_preds_path = Path(self.cfg.train.eval_preds_path)
        self.init_dirs()

        self.decision_metrics = cfg.train.decision_metrics

        if self.use_wandb:
            wandb.init(
                project=cfg.project_name,
                name=cfg.exp,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )

        log_file = Path(cfg.train.path_to_save) / "train_log.txt"
        log_file.unlink(missing_ok=True)
        logger.add(log_file, format="{message}", level="INFO", rotation="10 MB")

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
        if self.ignore_background_epochs:
            self.train_loader.dataset.ignore_background = True

        self.model = build_model(
            cfg.model_name,
            self.num_labels,
            cfg.train.device,
            img_size=cfg.train.img_size,
            pretrained_model_path=cfg.train.pretrained_model_path,
        )

        self.ema_model = None
        if self.cfg.train.use_ema:
            logger.info("EMA model will be evaluated and saved")
            self.ema_model = ModelEMA(self.model, cfg.train.ema_momentum)

        self.loss_fn = build_loss(
            cfg.model_name, self.num_labels, label_smoothing=cfg.train.label_smoothing
        )

        self.optimizer = build_optimizer(
            self.model,
            lr=cfg.train.base_lr,
            backbone_lr=cfg.train.backbone_lr,
            betas=cfg.train.betas,
            weight_decay=cfg.train.weight_decay,
            base_lr=cfg.train.base_lr,
        )

        max_lr = cfg.train.base_lr * 2
        if cfg.model_name in ["l", "x"]:  # per group max lr for big models
            max_lr = [
                cfg.train.backbone_lr * 2,
                cfg.train.backbone_lr * 2,
                cfg.train.base_lr * 2,
                cfg.train.base_lr * 2,
            ]
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=cfg.train.epochs,
            steps_per_epoch=len(self.train_loader) // self.b_accum_steps,
            pct_start=cfg.train.cycler_pct_start,
            cycle_momentum=False,
        )

        if self.amp_enabled:
            self.scaler = GradScaler()

        if self.use_wandb:
            wandb.watch(self.model)

    def init_dirs(self):
        for path in [self.debug_img_path, self.eval_preds_path]:
            if path.exists():
                rmtree(path)
            path.mkdir(exist_ok=True, parents=True)

        self.path_to_save.mkdir(exist_ok=True, parents=True)
        with open(self.path_to_save / "config.yaml", "w") as f:
            OmegaConf.save(config=self.cfg, f=f)

    @staticmethod
    def preds_postprocess(
        inputs: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        orig_sizes: torch.Tensor,
        num_labels: int,
        keep_ratio: bool,
        num_top_queries: int = 300,
        use_focal_loss: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        boxes = process_boxes(
            boxes, inputs.shape[2:], orig_sizes, keep_ratio, inputs.device
        )  # B x TopQ x 4

        if use_focal_loss:
            scores = torch.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
            labels = index - index // num_labels * num_labels
            index = index // num_labels
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

        results: List[Dict[str, torch.Tensor]] = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(
                labels=lab.detach().cpu(), boxes=box.detach().cpu(), scores=sco.detach().cpu()
            )
            results.append(result)
        return results

    @staticmethod
    def gt_postprocess(inputs, targets, orig_sizes, keep_ratio):
        results = []
        for idx, target in enumerate(targets):
            lab = target["labels"]
            box = process_boxes(
                target["boxes"][None],
                inputs[idx].shape[1:],
                orig_sizes[idx][None],
                keep_ratio,
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

        model.eval()
        for idx, (inputs, targets, img_paths) in enumerate(val_loader):
            inputs = inputs.to(self.device)
            if self.amp_enabled:
                with autocast(self.device.type, cache_enabled=True):
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

            preds = self.preds_postprocess(
                inputs, raw_res, orig_sizes, self.num_labels, self.keep_ratio
            )
            gt = self.gt_postprocess(inputs, targets, orig_sizes, self.keep_ratio)

            for pred_instance, gt_instance in zip(preds, gt):
                all_preds.append(pred_instance)
                all_gt.append(gt_instance)

            if self.to_visualize_eval and idx <= 5:
                visualize(
                    img_paths,
                    gt,
                    filter_preds(preds, self.conf_thresh),
                    dataset_path=Path(self.cfg.train.data_path) / "images",
                    path_to_save=self.eval_preds_path,
                    label_to_name=self.label_to_name,
                )
        return all_gt, all_preds

    @staticmethod
    def get_metrics(
        gt,
        preds,
        conf_thresh: float,
        iou_thresh: float,
        extended: bool,
        label_to_name: Dict[int, str],
        path_to_save=None,
        mode=None,
    ):
        validator = Validator(
            gt,
            preds,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            label_to_name=label_to_name,
        )
        metrics = validator.compute_metrics(extended=extended)
        if path_to_save:  # val and test
            validator.save_plots(path_to_save / "plots" / mode)
        return metrics

    def evaluate(
        self,
        val_loader: DataLoader,
        conf_thresh: float,
        iou_thresh: float,
        path_to_save: Path,
        extended: bool,
        mode: str = None,
    ) -> Dict[str, float]:
        gt, preds = self.get_preds_and_gt(val_loader=val_loader)
        metrics = self.get_metrics(
            gt,
            preds,
            conf_thresh,
            iou_thresh,
            extended=extended,
            label_to_name=self.label_to_name,
            path_to_save=path_to_save,
            mode=mode,
        )
        return metrics

    def save_model(self, metrics, best_metric):
        model_to_save = self.model
        if self.ema_model:
            model_to_save = self.ema_model.model

        self.path_to_save.mkdir(parents=True, exist_ok=True)
        torch.save(model_to_save.state_dict(), self.path_to_save / "last.pt")

        # mean from chosen metrics
        decision_metric = np.mean([metrics[metric_name] for metric_name in self.decision_metrics])

        if decision_metric > best_metric:
            best_metric = decision_metric
            logger.info("Saving new best model🔥")
            torch.save(model_to_save.state_dict(), self.path_to_save / "model.pt")
            self.early_stopping_steps = 0
        else:
            self.early_stopping_steps += 1
        return best_metric

    def train(self) -> None:
        best_metric = 0
        cur_iter = 0
        ema_iter = 0
        self.early_stopping_steps = 0
        one_epoch_time = None

        def optimizer_step(step_scheduler: bool):
            """
            Clip grads, optimizer step, scheduler step, zero grad, EMA model update
            """
            nonlocal ema_iter
            if self.amp_enabled:
                if self.clip_max_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                if self.clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.optimizer.step()

            if step_scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

            if self.ema_model:
                ema_iter += 1
                self.ema_model.update(ema_iter, self.model)

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.model.train()
            self.loss_fn.train()
            losses = []

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
                        with autocast(self.device.type, cache_enabled=True):
                            output = self.model(inputs, targets=targets)
                        with autocast(self.device.type, enabled=False):
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
                        loss=np.mean(losses) * self.b_accum_steps,
                        eta=calculate_remaining_time(
                            one_epoch_time,
                            epoch_start_time,
                            epoch,
                            self.epochs,
                            cur_iter,
                            len(self.train_loader),
                        ),
                        vram=f"{get_vram_usage()}%",
                    )

            # Final update for any leftover gradients from an incomplete accumulation step
            if (batch_idx + 1) % self.b_accum_steps != 0:
                optimizer_step(step_scheduler=False)

            if self.use_wandb:
                wandb.log({"lr": lr, "epoch": epoch})

            metrics = self.evaluate(
                val_loader=self.val_loader,
                conf_thresh=self.conf_thresh,
                iou_thresh=self.iou_thresh,
                extended=False,
                path_to_save=None,
            )

            best_metric = self.save_model(metrics, best_metric)
            save_metrics(
                {},
                metrics,
                np.mean(losses) * self.b_accum_steps,
                epoch,
                path_to_save=None,
                use_wandb=self.use_wandb,
            )

            if (
                epoch >= self.epochs - self.no_mosaic_epochs
                and self.train_loader.dataset.mosaic_prob
            ):
                self.train_loader.dataset.close_mosaic()

            if epoch == self.ignore_background_epochs:
                self.train_loader.dataset.ignore_background = False
                logger.info("Including background images")

            one_epoch_time = time.time() - epoch_start_time

            if self.early_stopping and self.early_stopping_steps >= self.early_stopping:
                logger.info("Early stopping")
                break

class Trainer_DDP:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        requested_device = cfg.train.device
        self.conf_thresh = cfg.train.conf_thresh
        self.iou_thresh = cfg.train.iou_thresh
        self.epochs = cfg.train.epochs
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

        self.debug_img_path = Path(self.cfg.train.debug_img_path)
        self.eval_preds_path = Path(self.cfg.train.eval_preds_path)

        self.decision_metrics = cfg.train.decision_metrics

        self.rank, self.local_rank, self.world_size = setup_ddp()
        logger.info(f"World size {self.world_size}")
        if torch.cuda.is_available() and requested_device.startswith("cuda"):
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(requested_device)

        # DDP-aware
        if is_main_process():
            self.init_dirs()

            if self.use_wandb:
                wandb.init(
                    project=cfg.project_name,
                    name=cfg.exp,
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                )

            log_file = Path(cfg.train.path_to_save) / "train_log.txt"
            log_file.unlink(missing_ok=True)
            logger.add(log_file, format="{message}", level="INFO", rotation="10 MB")

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
        if self.ignore_background_epochs:
            self.train_loader.dataset.ignore_background = True

        self.model = build_model(
            cfg.model_name,
            self.num_labels,
            self.device,
            img_size=cfg.train.img_size,
            pretrained_model_path=cfg.train.pretrained_model_path,
        )

        # wrap the model for DDP
        if is_main_process():
            logger.info("Converting model to SyncBatchNorm and wrapping with DDP...")
        self.model = SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device) # self.device as all 
        self.model = DDP(self.model, device_ids=[self.local_rank])

        # Initialize EMA model
        self.ema_model = None
        if self.cfg.train.use_ema:
            if is_main_process():
                logger.info("EMA model will be evaluated and saved")
            # Use the underlying module for EMA, not the DDP wrapper
            self.ema_model = ModelEMA(self.model.module, cfg.train.ema_momentum)

        self.loss_fn = build_loss(
            cfg.model_name, self.num_labels, label_smoothing=cfg.train.label_smoothing
        )

        self.optimizer = build_optimizer(
            self.model,
            lr=cfg.train.base_lr,
            backbone_lr=cfg.train.backbone_lr,
            betas=cfg.train.betas,
            weight_decay=cfg.train.weight_decay,
            base_lr=cfg.train.base_lr,
        )

        max_lr = cfg.train.base_lr * 2
        if cfg.model_name in ["l", "x"]:  # per group max lr for big models
            max_lr = [
                cfg.train.backbone_lr * 2,
                cfg.train.backbone_lr * 2,
                cfg.train.base_lr * 2,
                cfg.train.base_lr * 2,
            ]
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=cfg.train.epochs,
            steps_per_epoch=len(self.train_loader) // self.b_accum_steps,
            pct_start=cfg.train.cycler_pct_start,
            cycle_momentum=False,
        )

        if self.amp_enabled:
            self.scaler = GradScaler()

        if self.use_wandb and is_main_process():
            wandb.watch(self.model)

    def init_dirs(self):
        for path in [self.debug_img_path, self.eval_preds_path]:
            if path.exists():
                rmtree(path)
            path.mkdir(exist_ok=True, parents=True)

        self.path_to_save.mkdir(exist_ok=True, parents=True)
        with open(self.path_to_save / "config.yaml", "w") as f:
            OmegaConf.save(config=self.cfg, f=f)

    @staticmethod
    def preds_postprocess(
        inputs: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        orig_sizes: torch.Tensor,
        num_labels: int,
        keep_ratio: bool,
        num_top_queries: int = 300,
        use_focal_loss: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        boxes = process_boxes(
            boxes, inputs.shape[2:], orig_sizes, keep_ratio, inputs.device
        )  # B x TopQ x 4

        if use_focal_loss:
            scores = torch.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
            labels = index - index // num_labels * num_labels
            index = index // num_labels
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

        results: List[Dict[str, torch.Tensor]] = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(
                labels=lab.detach().cpu(), boxes=box.detach().cpu(), scores=sco.detach().cpu()
            )
            results.append(result)
        return results

    @staticmethod
    def gt_postprocess(inputs, targets, orig_sizes, keep_ratio):
        results = []
        for idx, target in enumerate(targets):
            lab = target["labels"]
            box = process_boxes(
                target["boxes"][None],
                inputs[idx].shape[1:],
                orig_sizes[idx][None],
                keep_ratio,
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

        model.eval()
        for idx, (inputs, targets, img_paths) in enumerate(val_loader):
            inputs = inputs.to(self.device)
            if self.amp_enabled:
                with autocast(self.device.type, cache_enabled=True):
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

            preds = self.preds_postprocess(
                inputs, raw_res, orig_sizes, self.num_labels, self.keep_ratio
            )
            gt = self.gt_postprocess(inputs, targets, orig_sizes, self.keep_ratio)

            for pred_instance, gt_instance in zip(preds, gt):
                all_preds.append(pred_instance)
                all_gt.append(gt_instance)

            if self.to_visualize_eval and idx <= 5 and is_main_process():
                visualize(
                    img_paths,
                    gt,
                    filter_preds(preds, self.conf_thresh),
                    dataset_path=Path(self.cfg.train.data_path) / "images",
                    path_to_save=self.eval_preds_path,
                    label_to_name=self.label_to_name,
                )
        return all_gt, all_preds

    @staticmethod
    def get_metrics(
        gt,
        preds,
        conf_thresh: float,
        iou_thresh: float,
        extended: bool,
        label_to_name: Dict[int, str],
        path_to_save=None,
        mode=None,
    ):
        validator = Validator(
            gt,
            preds,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            label_to_name=label_to_name,
        )
        metrics = validator.compute_metrics(extended=extended)
        if path_to_save and is_main_process:  # val and test
            validator.save_plots(path_to_save / "plots" / mode)
        return metrics

    def evaluate(
        self,
        val_loader: DataLoader,
        conf_thresh: float,
        iou_thresh: float,
        path_to_save: Path,
        extended: bool,
        mode: str = None,
    ) -> Dict[str, float]:
        gt, preds = self.get_preds_and_gt(val_loader=val_loader)
        metrics = self.get_metrics(
            gt,
            preds,
            conf_thresh,
            iou_thresh,
            extended=extended,
            label_to_name=self.label_to_name,
            path_to_save=path_to_save,
            mode=mode,
        )
        return metrics

    def save_model(self, metrics, best_metric):
        if not is_main_process():
            return best_metric

        model_to_save = self.model
        if self.ema_model:
            model_to_save = self.ema_model.model

        # Unwrap for rank0
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        self.path_to_save.mkdir(parents=True, exist_ok=True)
        torch.save(model_to_save.state_dict(), self.path_to_save / "last.pt")

        # mean from chosen metrics
        decision_metric = np.mean([metrics[metric_name] for metric_name in self.decision_metrics])

        if decision_metric > best_metric:
            best_metric = decision_metric
            logger.info("Saving new best model🔥")
            torch.save(model_to_save.state_dict(), self.path_to_save / "model.pt")
            self.early_stopping_steps = 0
        else:
            self.early_stopping_steps += 1
        return best_metric

    def train(self) -> None:
        best_metric = 0
        cur_iter = 0
        ema_iter = 0
        self.early_stopping_steps = 0
        one_epoch_time = None

        def optimizer_step(step_scheduler: bool):
            """
            Clip grads, optimizer step, scheduler step, zero grad, EMA model update
            """
            nonlocal ema_iter
            if self.amp_enabled:
                if self.clip_max_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                if self.clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.optimizer.step()

            if step_scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

            if self.ema_model:
                ema_iter += 1
                # if DDP, get underlying model
                underlying_model = self.model.module if hasattr(self.model, 'module') else self.model
                self.ema_model.update(ema_iter, underlying_model)

        for epoch in range(1, self.epochs + 1):
            # DDP Specific: set sampler epoch
            if self.train_loader.sampler and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            # End    
            epoch_start_time = time.time()
            self.model.train()
            self.loss_fn.train()
            losses = []

            # only rank0 should use tqdm
            with tqdm(self.train_loader, unit="batch", disable=not is_main_process()) as tepoch:
                for batch_idx, (inputs, targets, _) in enumerate(tepoch):
                    try:
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
                            with autocast(self.device.type, cache_enabled=True):
                                output = self.model(inputs, targets=targets)
                            with autocast(self.device.type, enabled=False):
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

                        if is_main_process():
                            tepoch.set_postfix(
                                loss=np.mean(losses) * self.b_accum_steps,
                                eta=calculate_remaining_time(
                                    one_epoch_time,
                                    epoch_start_time,
                                    epoch,
                                    self.epochs,
                                    cur_iter,
                                    len(self.train_loader),
                                ),
                                vram=f"{get_vram_usage()}%",
                            )
                    except Exception as e:
                        # Log error on all ranks
                        logger.error(f"Rank {self.rank} error in batch {batch_idx}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        # Re-raise to trigger cleanup in main()'s "finally" block
                        raise

            # Final update for any leftover gradients from an incomplete accumulation step
            if (batch_idx + 1) % self.b_accum_steps != 0:
                optimizer_step(step_scheduler=False)

            
            if self.use_wandb and is_main_process():
                wandb.log({"lr": lr, "epoch": epoch})

            # all processes must evaluate independently to stay in sync
            metrics = self.evaluate(
                val_loader=self.val_loader,
                conf_thresh=self.conf_thresh,
                iou_thresh=self.iou_thresh,
                extended=False,
                path_to_save=None,
            )

            # all steps below should happen on main process
            # only main process should log and save
            if is_main_process(): 
                best_metric = self.save_model(metrics, best_metric)
                save_metrics(
                    {},
                    metrics,
                    np.mean(losses) * self.b_accum_steps,
                    epoch,
                    path_to_save=None,
                    use_wandb=self.use_wandb,
                )

            # all ranks run independently
            if (
                epoch >= self.epochs - self.no_mosaic_epochs
                and self.train_loader.dataset.mosaic_prob
            ):
                self.train_loader.dataset.close_mosaic()

            if epoch == self.ignore_background_epochs:
                self.train_loader.dataset.ignore_background = False
                if is_main_process(): 
                    logger.info("Including background images")

            one_epoch_time = time.time() - epoch_start_time

            # Synchronize early stopping
            stop_training = torch.tensor(0, device=self.device)
            if is_main_process() and self.early_stopping and self.early_stopping_steps >= self.early_stopping:
                logger.info("Early stopping")
                stop_training = torch.tensor(1, device=self.device)

            if self.world_size > 1:
                dist.broadcast(stop_training, 0)

            if stop_training.item() == 1:
                break

        # let all ranks finish
        if self.world_size > 1:
            dist.barrier()
        

@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    
    if cfg.train.ddp:
        trainer = Trainer_DDP(cfg)
    else:
        trainer = Trainer(cfg)

    # Store original rank for each process before any DDP cleanup for post-training evaluation
    # After cleanup_ddp(), is_main_process() becomes unreliable because
    # dist.is_initialized() returns False, causing get_rank() to return 0 for ALL processes
    original_rank = trainer.rank if cfg.train.ddp else 0
    
    try:
        t_start = time.time()
        trainer.train()
        
        # Clean up DDP after training completes, before finally block
        # This ensures all ranks are synchronized before cleanup
        if cfg.train.ddp and dist.is_initialized():
            if is_main_process():
                logger.info("Training completed, cleaning up DDP...")
            cleanup_ddp()
            
    except KeyboardInterrupt:
        if is_main_process(): # Use DDP-aware check
            logger.warning("Interrupted by user")
        # Ensure all ranks exit cleanly
        if cfg.train.ddp and dist.is_initialized():
            cleanup_ddp()
        return
    except Exception as e:
        # Log error on all ranks for debugging
        import traceback
        logger.error(f"Rank {dist.get_rank() if dist.is_initialized() else 0} error: {e}")
        logger.error(traceback.format_exc())
        # Ensure all ranks are aware of the failure
        if cfg.train.ddp and dist.is_initialized():
            try:
                # Try to synchronize before cleanup (without timeout for compatibility)
                dist.barrier()
            except Exception as barrier_error:
                logger.error(f"Barrier failed during error cleanup: {barrier_error}")
            finally:
                cleanup_ddp()
    finally:
        # Only original rank 0 should run evaluation
        # After cleanup_ddp(), is_main_process() returns True for ALL processes
        # because dist.is_initialized() is False, so we use the captured original_rank
        if original_rank == 0: 
            logger.info("Evaluating best model...")
            cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

            # Free GPU memory from training before loading evaluation model
            # This prevents CUDA OOM when loading the evaluation model
            if cfg.train.ddp:
                # Delete trainer's models to free GPU memory
                if hasattr(trainer, 'model'):
                    del trainer.model
                if hasattr(trainer, 'ema_model'):
                    del trainer.ema_model
                # Clear CUDA cache
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # For DDP, ensure we use cuda:0 for main process evaluation
            eval_device = torch.device('cuda:0') if cfg.train.ddp else (trainer.device if hasattr(trainer, 'device') else cfg.train.device)
            
            model = build_model(
                cfg.model_name,
                len(cfg.train.label_to_name),
                eval_device,
                img_size=cfg.train.img_size,
            )
            model.load_state_dict(
                torch.load(Path(cfg.train.path_to_save) / "model.pt", weights_only=True)
            )
            # Ensure model is on the correct device
            model = model.to(eval_device)
            
            # For DDP: need to create a fresh non-DDP dataloader for evaluation
            # since DDP has been cleaned up
            if cfg.train.ddp:
                # Create a new dataloader without DDP sampler
                from src.dl.dataset import Loader
                base_loader = Loader(
                    root_path=Path(cfg.train.data_path),
                    img_size=tuple(cfg.train.img_size),
                    batch_size=cfg.train.batch_size,
                    num_workers=cfg.train.num_workers,
                    cfg=cfg,
                    debug_img_processing=cfg.train.debug_img_processing,
                )
                _, eval_val_loader, eval_test_loader = base_loader.build_dataloaders()
                
                # Create a minimal evaluation wrapper without initializing a full Trainer
                # This avoids device conflicts from Trainer.__init__ building its own model
                class EvalWrapper:
                    def __init__(self, model, device, cfg, trainer_cfg):
                        self.model = model
                        self.device = device
                        self.ema_model = None  # No EMA for final eval
                        self.num_labels = len(cfg.train.label_to_name)
                        self.keep_ratio = trainer_cfg.keep_ratio
                        self.amp_enabled = cfg.train.amp_enabled
                        self.to_visualize_eval = trainer_cfg.to_visualize_eval
                        self.label_to_name = cfg.train.label_to_name
                        self.cfg = cfg
                        self.conf_thresh = cfg.train.conf_thresh
                        self.eval_preds_path = Path(cfg.train.eval_preds_path)
                    
                    # Copy the methods from Trainer_DDP
                    preds_postprocess = staticmethod(Trainer_DDP.preds_postprocess)
                    gt_postprocess = staticmethod(Trainer_DDP.gt_postprocess)
                    get_preds_and_gt = Trainer_DDP.get_preds_and_gt
                    get_metrics = staticmethod(Trainer_DDP.get_metrics)
                    evaluate = Trainer_DDP.evaluate
                
                eval_wrapper = EvalWrapper(model, eval_device, cfg, trainer)
                
                val_metrics = eval_wrapper.evaluate(
                    val_loader=eval_val_loader,
                    conf_thresh=trainer.conf_thresh,
                    iou_thresh=trainer.iou_thresh,
                    path_to_save=Path(cfg.train.path_to_save),
                    extended=True,
                    mode="val",
                )
            else:
                # Single GPU: can use trainer's evaluate method
                if trainer.ema_model:
                    trainer.ema_model.model = model.to(eval_device)
                else:
                    trainer.model = model.to(eval_device)

                val_metrics = trainer.evaluate(
                    val_loader=trainer.val_loader,
                    conf_thresh=trainer.conf_thresh,
                    iou_thresh=trainer.iou_thresh,
                    path_to_save=Path(cfg.train.path_to_save),
                    extended=True,
                    mode="val",
                )
            if cfg.train.use_wandb:
                wandb_logger(None, val_metrics, epoch=cfg.train.epochs + 1, mode="val")

            test_metrics = {}
            if cfg.train.ddp:
                if eval_test_loader:
                    test_metrics = eval_wrapper.evaluate(
                        val_loader=eval_test_loader,
                        conf_thresh=trainer.conf_thresh,
                        iou_thresh=trainer.iou_thresh,
                        path_to_save=Path(cfg.train.path_to_save),
                        extended=True,
                        mode="test",
                    )
            else:
                if trainer.test_loader:
                    test_metrics = trainer.evaluate(
                        val_loader=trainer.test_loader,
                        conf_thresh=trainer.conf_thresh,
                        iou_thresh=trainer.iou_thresh,
                        path_to_save=Path(cfg.train.path_to_save),
                        extended=True,
                        mode="test",
                    )
                if cfg.train.use_wandb:
                    wandb_logger(None, test_metrics, epoch=-1, mode="test")

            log_metrics_locally(
                all_metrics={"val": val_metrics, "test": test_metrics},
                path_to_save=Path(cfg.train.path_to_save),
                epoch=0,
                extended=True,
            )
            logger.info(f"Full training time: {(time.time() - t_start) / 60 / 60:.2f} hours")

if __name__ == "__main__":
    main()
