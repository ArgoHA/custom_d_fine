import math
import time
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.d_fine.dfine import build_loss, build_model, build_optimizer
from src.d_fine.dist_utils import (
    broadcast_scalar,
    cleanup_distributed,
    gather_predictions,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_main_process,
    synchronize,
)
from src.dl.dataset import Loader
from src.dl.utils import (
    calculate_remaining_time,
    cleanup_masks,
    encode_sample_masks_to_rle,
    get_latest_experiment_name,
    get_vram_usage,
    log_metrics_locally,
    process_boxes,
    process_masks,
    save_metrics,
    set_seeds,
    visualize,
    wandb_logger,
)
from src.dl.validator import Validator


class ModelEMA:
    def __init__(self, student, ema_momentum):
        # unwrap DDP if needed
        if isinstance(student, DDP):
            student = student.module
        self.model = deepcopy(student).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.ema_scheduler = lambda x: ema_momentum * (1 - math.exp(-x / 2000))

    def update(self, iters, student):
        # unwrap DDP if needed
        if isinstance(student, DDP):
            student = student.module

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

        self.distributed = hasattr(cfg.train, "ddp") and cfg.train.ddp.enabled
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main = self.rank == 0
        if self.distributed and torch.cuda.is_available():
            self.local_rank = get_local_rank()
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.local_rank = 0
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
        self.task = cfg.task  # detect/segment
        self.mask_batch_size = cfg.train.mask_batch_size
        enable_mask_head = self.task == "segment"

        self.debug_img_path = Path(self.cfg.train.debug_img_path)
        self.eval_preds_path = Path(self.cfg.train.eval_preds_path)
        self.decision_metrics = cfg.train.decision_metrics

        if self.is_main:
            self.init_dirs()

        if enable_mask_head and "iou" not in self.decision_metrics:
            self.decision_metrics.append("iou")
        if self.use_wandb and self.is_main:
            wandb.init(
                project=cfg.project_name,
                name=cfg.exp,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )

        log_file = Path(cfg.train.path_to_save) / "train_log.txt"
        if (not self.distributed) or self.is_main:
            log_file.unlink(missing_ok=True)
            logger.add(log_file, format="{message}", level="INFO", rotation="10 MB")

        logger.info(f"Experiment: {cfg.exp}, Task: {self.task}")
        seed = cfg.train.seed + self.rank if self.distributed else cfg.train.seed
        set_seeds(seed, cfg.train.cudnn_fixed)

        base_loader = Loader(
            root_path=Path(cfg.train.data_path),
            img_size=tuple(cfg.train.img_size),
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            cfg=cfg,
            debug_img_processing=cfg.train.debug_img_processing,
        )
        self.train_loader, self.val_loader, self.test_loader = base_loader.build_dataloaders(
            distributed=self.distributed
        )
        self.train_sampler = getattr(base_loader, "train_sampler", None)
        if self.ignore_background_epochs:
            self.train_loader.dataset.ignore_background = True

        self.model = build_model(
            cfg.model_name,
            self.num_labels,
            enable_mask_head,
            str(self.device),
            img_size=cfg.train.img_size,
            pretrained_model_path=cfg.train.pretrained_model_path,
        )
        if self.distributed:
            if torch.cuda.is_available():
                if cfg.train.batch_size < 4:  # SyncBatch is useful for small batches
                    self.model = SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                )
            else:
                # CPU DDP fallback (unlikely, but safe)
                self.model = DDP(self.model)

        self.ema_model = None
        if self.cfg.train.use_ema:
            self.ema_model = ModelEMA(self.model, cfg.train.ema_momentum)
            if self.is_main:
                logger.info("EMA model will be evaluated and saved")

        self.loss_fn = build_loss(
            cfg.model_name,
            self.num_labels,
            label_smoothing=cfg.train.label_smoothing,
            enable_mask_head=enable_mask_head,
        )

        self.optimizer = build_optimizer(
            self.model,
            lr=cfg.train.base_lr,
            backbone_lr=cfg.train.backbone_lr,
            betas=cfg.train.betas,
            weight_decay=cfg.train.weight_decay,
            base_lr=cfg.train.base_lr,
        )

        self.scheduler = None
        if cfg.train.use_scheduler:
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

        if self.use_wandb and self.is_main:
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
        conf_thresh: float,
        num_top_queries: int = 300,
        use_focal_loss: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        has_masks = ("pred_masks" in outputs) and (outputs["pred_masks"] is not None)
        pred_masks = outputs["pred_masks"] if has_masks else None  # [B,Q,Hm,Wm]
        B, Q = logits.shape[:2]

        # Map boxes back to original size
        boxes = process_boxes(
            boxes, inputs.shape[2:], orig_sizes, keep_ratio, inputs.device
        )  # B x TopQ x 4

        # scores/labels and preliminary topK over all Q*C
        if use_focal_loss:
            scores_all = torch.sigmoid(logits)  # [B,Q,C]
            flat = scores_all.flatten(1)  # [B, Q*C]
            # pre-topk to avoid scanning all queries later
            K = min(num_top_queries, flat.shape[1])
            topk_scores, topk_idx = torch.topk(flat, K, dim=-1)  # [B,K]
            topk_labels = topk_idx - (topk_idx // num_labels) * num_labels  # [B,K]
            topk_qidx = topk_idx // num_labels  # [B,K]
        else:
            probs = torch.softmax(logits, dim=-1)[:, :, :-1]  # [B,Q,C-1]
            topk_scores, topk_labels = probs.max(dim=-1)  # [B,Q]
            # keep at most K queries per image by score
            K = min(num_top_queries, Q)
            topk_scores, order = torch.topk(topk_scores, K, dim=-1)  # [B,K]
            topk_labels = topk_labels.gather(1, order)  # [B,K]
            topk_qidx = order

        results = []
        for b in range(B):
            # Filter by conf thresh, but keep boxes for mAP calc
            sb = topk_scores[b]
            lb = topk_labels[b]
            qb = topk_qidx[b]
            keep = sb >= conf_thresh

            sb = sb[keep]
            lb = lb[keep]
            qb = qb[keep]
            # gather boxes once
            bb = boxes[b].gather(0, qb.unsqueeze(-1).repeat(1, 4))

            # gather all_boxes using topk query indices to match all_scores/all_labels ordering
            all_bb = boxes[b].gather(0, topk_qidx[b].unsqueeze(-1).repeat(1, 4))

            out = {
                "labels": lb.detach().cpu(),
                "boxes": bb.detach().cpu(),
                "scores": sb.detach().cpu(),
                "all_boxes": all_bb.detach().cpu(),
                "all_scores": topk_scores[b].detach().cpu(),
                "all_labels": topk_labels[b].detach().cpu(),
            }

            if has_masks and qb.numel() > 0:
                # gather only kept masks, cast to half to save mem during resizing
                mb = pred_masks[b, qb]
                mb = mb.to(dtype=torch.float16)  # reduce VRAM and RAM during resize
                # resize to original size (list with length 1)
                masks_list = process_masks(
                    mb.unsqueeze(0),
                    processed_size=inputs.shape[2:],
                    orig_sizes=orig_sizes[b].unsqueeze(0),
                    keep_ratio=keep_ratio,
                )
                out["mask_probs"] = (
                    masks_list[0].to(dtype=torch.float32).detach().cpu()
                )  # [K, H0, W0]

                # binarize masks
                out["masks"] = (
                    (masks_list[0].clamp(0, 1) >= conf_thresh).to(torch.uint8).detach().cpu()
                )  # [N, H, W]

                # clean up masks outside of the corresponding bbox
                out["masks"] = cleanup_masks(out["masks"], out["boxes"])
                del out["mask_probs"]

            results.append(out)
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

            # GT masks come from dataset already rasterized at network size; map to original size
            if (
                "masks" in targets[idx]
                and targets[idx]["masks"] is not None
                and targets[idx]["masks"].numel() > 0
            ):
                gt_m = targets[idx]["masks"].to(
                    dtype=inputs.dtype, device=inputs.device
                )  # [Ni,Hnet,Wnet]
                gt_m = gt_m.unsqueeze(0)  # [1,Ni,Hnet,Wnet] to match helper API
                # Helper expects [B,Q,Hm,Wm]; we pass B=1, Q=Ni
                masks_list = process_masks(
                    gt_m,
                    processed_size=inputs[idx].shape[1:],  # (Hnet, Wnet)
                    orig_sizes=orig_sizes[idx].unsqueeze(0),  # [1,2]
                    keep_ratio=keep_ratio,
                )
                # back to [Ni,H0,W0] and uint8
                result["masks"] = (masks_list[0].clamp(0, 1) >= 0.5).to(torch.uint8).detach().cpu()
            else:
                result["masks"] = torch.zeros(
                    (0, int(orig_sizes[idx, 0].item()), int(orig_sizes[idx, 1].item())),
                    dtype=torch.uint8,
                )

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
        with torch.inference_mode():
            for idx, (inputs, targets, img_paths) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                if self.amp_enabled:
                    with autocast(str(self.device), cache_enabled=True):
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

                gt = self.gt_postprocess(inputs, targets, orig_sizes, self.keep_ratio)
                preds = self.preds_postprocess(
                    inputs, raw_res, orig_sizes, self.num_labels, self.keep_ratio, self.conf_thresh
                )

                if self.to_visualize_eval and idx <= 5:
                    visualize(
                        img_paths,
                        gt,
                        preds,
                        dataset_path=Path(self.cfg.train.data_path) / "images",
                        path_to_save=self.eval_preds_path,
                        label_to_name=self.label_to_name,
                    )

                # collect all preds and gt for metrics
                for gt_instance, pred_instance in zip(gt, preds):
                    # Encode masks to RLE to save memory during validation
                    all_preds.append(encode_sample_masks_to_rle(pred_instance))
                    all_gt.append(encode_sample_masks_to_rle(gt_instance))

        return all_gt, all_preds

    def evaluate(
        self,
        val_loader: DataLoader,
        conf_thresh: float,
        iou_thresh: float,
        path_to_save: Path,
        extended: bool,
        mode: str = None,
    ) -> Dict[str, float]:
        # All ranks perform inference on their portion of the data
        local_gt, local_preds = self.get_preds_and_gt(val_loader=val_loader)

        # Gather predictions from all ranks to rank 0
        if self.distributed:
            all_preds, all_gt = gather_predictions(local_preds, local_gt)
            synchronize()  # Ensure all ranks are done before continuing
        else:
            all_gt, all_preds = local_gt, local_preds

        # Only rank 0 computes metrics
        metrics = None
        if self.is_main and all_preds is not None and all_gt is not None:
            validator = Validator(
                all_gt,
                all_preds,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                label_to_name=self.label_to_name,
                mask_batch_size=self.mask_batch_size,
            )
            metrics = validator.compute_metrics(extended=extended)
            if path_to_save:  # val and test
                validator.save_plots(path_to_save / "plots" / mode)

        # Synchronize before returning so all ranks wait for metrics computation
        if self.distributed:
            synchronize()
        return metrics

    def save_model(self, metrics, best_metric):
        model_to_save = self.model
        if self.ema_model:
            model_to_save = self.ema_model.model

        if isinstance(model_to_save, DDP):
            model_to_save = model_to_save.module

        self.path_to_save.mkdir(parents=True, exist_ok=True)
        torch.save(model_to_save.state_dict(), self.path_to_save / "last.pt")

        # mean from chosen metrics
        decision_metric = np.mean(
            [
                metrics[metric_name]
                for metric_name in self.decision_metrics
                if metric_name in metrics
            ]
        )

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

            if step_scheduler and self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

            if self.ema_model:
                ema_iter += 1
                self.ema_model.update(ema_iter, self.model)

        for epoch in range(1, self.epochs + 1):
            if self.distributed and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            epoch_start_time = time.time()
            self.model.train()
            self.loss_fn.train()
            losses = []

            data_iter = self.train_loader
            if self.is_main:
                data_iter = tqdm(self.train_loader, unit="batch")

            for batch_idx, (inputs, targets, _) in enumerate(data_iter):
                if self.is_main:
                    data_iter.set_description(f"Epoch {epoch}/{self.epochs}")

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
                    with autocast(str(self.device), cache_enabled=True):
                        output = self.model(inputs, targets=targets)
                    with autocast(str(self.device), enabled=False):
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

                if self.is_main:
                    data_iter.set_postfix(
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

            if self.use_wandb and self.is_main:
                wandb.log({"lr": lr, "epoch": epoch})

            # All ranks run evaluation (inference is distributed, metrics computed on rank 0)
            metrics = self.evaluate(
                val_loader=self.val_loader,
                conf_thresh=self.conf_thresh,
                iou_thresh=self.iou_thresh,
                extended=False,
                path_to_save=None,
            )

            # Only rank 0 saves and logs
            if self.is_main:
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

            local_stop = False
            if (
                self.is_main
                and self.early_stopping
                and self.early_stopping_steps >= self.early_stopping
            ):
                local_stop = True

            if self.distributed:
                stop_flag = bool(int(broadcast_scalar(int(local_stop), src=0)))
            else:
                stop_flag = local_stop

            if stop_flag:
                if self.is_main:
                    logger.info("Early stopping")
                break


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    ddp_enabled = hasattr(cfg.train, "ddp") and cfg.train.ddp.enabled
    if ddp_enabled:
        init_distributed_mode()

    trainer = Trainer(cfg)

    try:
        t_start = time.time()
        trainer.train()
    except KeyboardInterrupt:
        if is_main_process():
            logger.warning("Interrupted by user")
    except Exception as e:
        if is_main_process():
            logger.error(e)
    finally:
        if is_main_process():
            logger.info("Evaluating best model...")
            cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

            model = build_model(
                cfg.model_name,
                len(cfg.train.label_to_name),
                cfg.task == "segment",
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

            # rebuild val and test loaders without DDP for evaluation
            if ddp_enabled:
                base_loader = Loader(
                    root_path=Path(cfg.train.data_path),
                    img_size=tuple(cfg.train.img_size),
                    batch_size=cfg.train.batch_size,
                    num_workers=cfg.train.num_workers,
                    cfg=cfg,
                    debug_img_processing=cfg.train.debug_img_processing,
                )
                _, val_loader_eval, test_loader_eval = base_loader.build_dataloaders(
                    distributed=False
                )
                trainer.val_loader = val_loader_eval
                trainer.test_loader = test_loader_eval
                trainer.distributed = False  # turn off DDP inside evaluate

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

        if ddp_enabled:
            cleanup_distributed()


if __name__ == "__main__":
    main()
