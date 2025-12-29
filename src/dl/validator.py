import copy
import gc
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from src.dl.utils import filter_preds, rle_to_masks

# Suppress verbose output from faster_coco_eval
logging.getLogger("faster_coco_eval").setLevel(logging.WARNING)


class Validator:
    def __init__(
        self,
        gt: List[Dict[str, torch.Tensor]],
        preds: List[Dict[str, torch.Tensor]],
        label_to_name: Dict[int, str],
        conf_thresh=0.5,
        iou_thresh=0.5,
        mask_batch_size=1000,
    ) -> None:
        """
        Format example:
        gt = [{'labels': tensor([0]), 'boxes': tensor([[561.0, 297.0, 661.0, 359.0]])}, ...]
        len(gt) is the number of images
        bboxes are in format [x1, y1, x2, y2], absolute values

        mask_batch_size - Number of images to process at once when computing mask metrics.
            Lower values use less RAM but may be slower. Default 500.
        """
        self.gt = gt
        self.preds = preds
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.thresholds = np.arange(0.2, 1.0, 0.05)
        self.label_to_name = label_to_name
        self.conf_matrix = None
        self.mask_batch_size = mask_batch_size

        # Use faster_coco_eval backend for numpy 2.x compatibility
        self.torch_metric = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", sync_on_compute=False, backend="faster_coco_eval"
        )
        self.torch_metric.warn_on_many_detections = False

        # get raw preds for torchmetrics
        self.torchmetrics_preds = copy.deepcopy(preds)

        if len(self.torchmetrics_preds) > 0 and "all_boxes" in self.torchmetrics_preds[0]:
            for torchmetrics_pred in self.torchmetrics_preds:
                for key in ["boxes", "labels", "scores"]:
                    torchmetrics_pred[key] = torchmetrics_pred[f"all_{key}"]
                    del torchmetrics_pred[f"all_{key}"]

        self.torch_metric.update(self.torchmetrics_preds, gt)

        # Check if masks available (either dense or RLE-encoded)
        def _has_masks(sample):
            if "masks" in sample and sample["masks"] is not None:
                if hasattr(sample["masks"], "numel"):
                    return sample["masks"].numel() > 0
                return True
            if "masks_rle" in sample and sample["masks_rle"]:
                return True
            return False

        self.use_masks = any(_has_masks(p) for p in preds) and any(_has_masks(g) for g in gt)
        if self.use_masks:
            self.torch_metric_mask = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="segm",
                sync_on_compute=False,
                backend="faster_coco_eval",
            )
            self.torch_metric_mask.warn_on_many_detections = False
            # Decode RLE masks for torchmetrics in batches to avoid OOM
            # torchmetrics supports incremental .update() calls
            batch_size = self.mask_batch_size
            n_samples = len(preds)
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                # Deep copy only this batch
                preds_batch = [copy.deepcopy(preds[i]) for i in range(batch_start, batch_end)]
                gt_batch = [copy.deepcopy(gt[i]) for i in range(batch_start, batch_end)]

                # Decode RLE to dense for this batch only
                preds_batch = self._prepare_masks_for_torchmetrics(preds_batch)
                gt_batch = self._prepare_masks_for_torchmetrics(gt_batch)

                # Update metrics incrementally
                self.torch_metric_mask.update(preds_batch, gt_batch)

                # Explicitly free memory
                del preds_batch, gt_batch

    def compute_metrics(self, extended=False, ignore_masks=False, cleanup=True) -> Dict[str, float]:
        self.torch_metrics = self.torch_metric.compute()

        metrics = self._compute_main_metrics(self.preds, ignore_masks=ignore_masks)
        metrics["mAP_50"] = self.torch_metrics["map_50"].item()
        metrics["mAP_50_95"] = self.torch_metrics["map"].item()
        if self.use_masks and not ignore_masks:
            tm_mask = self.torch_metric_mask.compute()
            metrics["mAP_50_mask"] = tm_mask["map_50"].item()
            metrics["mAP_50_95_mask"] = tm_mask["map"].item()
            metrics["extended_metrics"].update(
                {
                    "mAP_50_95_mask": metrics["mAP_50_95_mask"],
                    "mAP_50_95": metrics["mAP_50_95"],
                }
            )
            del tm_mask

        if not extended:
            metrics.pop("extended_metrics", None)
        # Clean up large data structures to free RAM
        if cleanup:
            self._cleanup_torchmetrics()
        return metrics

    def _cleanup_torchmetrics(self):
        """Reset torchmetrics internal state to free memory."""

        # Reset torchmetrics - this clears their internal detection/groundtruth lists
        if hasattr(self, "torch_metric"):
            self.torch_metric.reset()
        if hasattr(self, "torch_metric_mask"):
            self.torch_metric_mask.reset()

        # Clear the deep copy of predictions used for torchmetrics
        if hasattr(self, "torchmetrics_preds"):
            del self.torchmetrics_preds
            self.torchmetrics_preds = None

        # Clear stored torch_metrics results
        if hasattr(self, "torch_metrics"):
            del self.torch_metrics
            self.torch_metrics = None

        # Force garbage collection
        gc.collect()

    @staticmethod
    def _decode_masks_if_rle(sample: Dict, device: str = "cpu") -> torch.Tensor:
        """
        Get masks from sample, decoding RLE if necessary.
        Returns [N, H, W] uint8 tensor.
        """
        # If already has dense masks, return them
        if "masks" in sample and sample["masks"] is not None and sample["masks"].numel() > 0:
            m = sample["masks"]
            if m.dim() == 4 and m.shape[1] == 1:
                m = m[:, 0]
            return m.to(torch.uint8)

        # If has RLE-encoded masks, decode them
        if "masks_rle" in sample and sample["masks_rle"]:
            return rle_to_masks(sample["masks_rle"], device=device)

        # No masks available
        return torch.zeros((0, 1, 1), dtype=torch.uint8)

    def _prepare_masks_for_torchmetrics(self, samples: List[Dict]) -> List[Dict]:
        """
        Prepare samples for torchmetrics by ensuring dense binary masks.
        Decodes RLE if present, ensures uint8 format.
        """
        for s in samples:
            # Decode RLE if present
            if "masks_rle" in s and s["masks_rle"]:
                s["masks"] = self._decode_masks_if_rle(s)
                # Clean up RLE keys
                if "masks_rle" in s:
                    del s["masks_rle"]
                if "masks_size" in s:
                    del s["masks_size"]

            # Ensure binary uint8 masks
            if "masks" in s and s["masks"] is not None:
                s["masks"] = self._binarize_masks(s["masks"])
            elif "mask_probs" in s and s["mask_probs"] is not None:
                s["masks"] = self._binarize_masks(s["mask_probs"])
            else:
                s["masks"] = torch.zeros((0, 1, 1), dtype=torch.uint8)

        return samples

    def _binarize_masks(self, m: torch.Tensor) -> torch.Tensor:
        """Ensure binary uint8 masks using self.conf_thresh if needed."""
        if m is None or m.numel() == 0:
            return torch.zeros((0, 1, 1), dtype=torch.uint8)  # harmless empty
        if m.dtype == torch.uint8:
            return m
        return (m > float(self.conf_thresh)).to(torch.uint8)

    def _get_pred_masks_bin(self, pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return [Np,H,W] uint8; prefer 'masks'/'masks_rle' else binarize 'mask_probs'."""
        # First try RLE-encoded masks
        if "masks_rle" in pred and pred["masks_rle"]:
            return self._decode_masks_if_rle(pred)
        if "masks" in pred and pred["masks"] is not None:
            return self._binarize_masks(pred["masks"])
        if "mask_probs" in pred and pred["mask_probs"] is not None:
            return self._binarize_masks(pred["mask_probs"])
        return torch.zeros((0, 1, 1), dtype=torch.uint8)

    def _ensure_binary_pred_masks(self, preds: List[Dict[str, torch.Tensor]]):
        """Make sure each pred dict has a 'masks' (uint8) key for segm mAP."""
        for p in preds:
            if ("masks" not in p) or (p["masks"] is None) or (p["masks"].numel() == 0):
                mb = self._get_pred_masks_bin(p)
                # only set if non-empty; segm mAP tolerates empty as zero-len tensor too
                p["masks"] = mb
        return preds

    def _to_nhw_uint8(self, m: torch.Tensor) -> torch.Tensor:
        """
        Return binary uint8 masks with shape [N,H,W].
        Accepts [N,H,W] or [N,1,H,W] or probabilities.
        """
        if m is None or m.numel() == 0:
            return torch.zeros((0, 1, 1), dtype=torch.uint8)

        # binarize if not uint8
        if m.dtype != torch.uint8:
            m = (m > float(self.conf_thresh)).to(torch.uint8)

        # squeeze channel if present
        if m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0]
        elif m.ndim == 3:
            pass
        else:
            # fallback for unexpected shapes
            m = m.reshape(m.shape[0], -1, m.shape[-2], m.shape[-1])
            m = m[:, 0].to(torch.uint8)

        return m

    def _get_gt_masks_bin(self, gt: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return [Ng,H,W] uint8 for GT. Handles both dense and RLE-encoded masks.
        """
        # Try RLE first
        if "masks_rle" in gt and gt["masks_rle"]:
            return self._decode_masks_if_rle(gt)
        if "masks" not in gt or gt["masks"] is None or gt["masks"].numel() == 0:
            return torch.zeros((0, 1, 1), dtype=torch.uint8)
        return self._to_nhw_uint8(gt["masks"])

    def _get_pred_masks_bin_nhw(self, pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return [Np,H,W] uint8 for preds using 'masks', 'masks_rle', or 'mask_probs'.
        """
        # Try RLE first
        if "masks_rle" in pred and pred["masks_rle"]:
            return self._decode_masks_if_rle(pred)
        if "masks" in pred and pred["masks"] is not None and pred["masks"].numel() > 0:
            return self._to_nhw_uint8(pred["masks"])
        if (
            "mask_probs" in pred
            and pred["mask_probs"] is not None
            and pred["mask_probs"].numel() > 0
        ):
            return self._to_nhw_uint8(pred["mask_probs"])
        return torch.zeros((0, 1, 1), dtype=torch.uint8)

    def _pairwise_mask_iou(self, pm: torch.Tensor, gm: torch.Tensor) -> torch.Tensor:
        # pm: [Np,H,W] uint8, gm: [Ng,H,W] uint8
        if pm.numel() == 0 or gm.numel() == 0:
            return torch.zeros((pm.shape[0], gm.shape[0]))
        pmf = pm.to(dtype=torch.float32).flatten(1)  # [Np,HW]
        gmf = gm.to(dtype=torch.float32).flatten(1)  # [Ng,HW]
        inter = pmf @ gmf.T  # [Np,Ng]
        area_p = pmf.sum(dim=1, keepdim=True)  # [Np,1]
        area_g = gmf.sum(dim=1, keepdim=True).T  # [1,Ng]
        union = area_p + area_g - inter
        return torch.where(union > 0, inter / union, torch.zeros_like(union))

    def _compute_main_metrics(self, preds, ignore_masks=False):
        (
            self.metrics_per_class,
            self.conf_matrix,
            self.class_to_idx,
        ) = self._compute_metrics_and_confusion_matrix(preds, ignore_masks=ignore_masks)
        tps, fps, fns = 0, 0, 0
        ious = []
        extended_metrics = {}
        for key, value in self.metrics_per_class.items():
            tps += value["TPs"]
            fps += value["FPs"]
            fns += value["FNs"]
            ious.extend(value["IoUs"])

            extended_metrics[f"precision_{self.label_to_name[key]}"] = (
                value["TPs"] / (value["TPs"] + value["FPs"])
                if value["TPs"] + value["FPs"] > 0
                else 0
            )
            extended_metrics[f"recall_{self.label_to_name[key]}"] = (
                value["TPs"] / (value["TPs"] + value["FNs"])
                if value["TPs"] + value["FNs"] > 0
                else 0
            )

            extended_metrics[f"iou_{self.label_to_name[key]}"] = np.mean(value["IoUs"])
            extended_metrics[f"f1_{self.label_to_name[key]}"] = (
                2
                * (
                    extended_metrics[f"precision_{self.label_to_name[key]}"]
                    * extended_metrics[f"recall_{self.label_to_name[key]}"]
                )
                / (
                    extended_metrics[f"precision_{self.label_to_name[key]}"]
                    + extended_metrics[f"recall_{self.label_to_name[key]}"]
                )
                if (
                    extended_metrics[f"precision_{self.label_to_name[key]}"]
                    + extended_metrics[f"recall_{self.label_to_name[key]}"]
                )
                > 0
                else 0
            )

        precision = tps / (tps + fps) if (tps + fps) > 0 else 0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "iou": np.mean(ious) if ious else 0,
            "TPs": tps,
            "FPs": fps,
            "FNs": fns,
            "extended_metrics": extended_metrics,
        }

    def _compute_metrics_and_confusion_matrix(self, preds, ignore_masks):
        if self.use_masks and not ignore_masks:
            return self._compute_metrics_and_confusion_matrix_masks(preds)
        # Initialize per-class metrics
        metrics_per_class = defaultdict(lambda: {"TPs": 0, "FPs": 0, "FNs": 0, "IoUs": []})

        # Collect all class IDs
        all_classes = set()
        for pred in preds:
            all_classes.update(pred["labels"].tolist())
        for gt in self.gt:
            all_classes.update(gt["labels"].tolist())
        all_classes = sorted(list(all_classes))
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(all_classes)}
        n_classes = len(all_classes)
        conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)  # +1 for background class

        for pred, gt in zip(preds, self.gt):
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            n_preds = len(pred_boxes)
            n_gts = len(gt_boxes)

            if n_preds == 0 and n_gts == 0:
                continue

            ious = box_iou(pred_boxes, gt_boxes) if n_preds > 0 and n_gts > 0 else torch.tensor([])
            # Assign matches between preds and gts
            matched_pred_indices = set()
            matched_gt_indices = set()

            if ious.numel() > 0:
                # For each pred box, find the gt box with highest IoU
                ious_mask = ious >= self.iou_thresh
                pred_indices, gt_indices = torch.nonzero(ious_mask, as_tuple=True)
                iou_values = ious[pred_indices, gt_indices]

                # Sorting by IoU to match highest scores first
                sorted_indices = torch.argsort(-iou_values)
                pred_indices = pred_indices[sorted_indices]
                gt_indices = gt_indices[sorted_indices]
                iou_values = iou_values[sorted_indices]

                for pred_idx, gt_idx, iou in zip(pred_indices, gt_indices, iou_values):
                    if (
                        pred_idx.item() in matched_pred_indices
                        or gt_idx.item() in matched_gt_indices
                    ):
                        continue
                    matched_pred_indices.add(pred_idx.item())
                    matched_gt_indices.add(gt_idx.item())

                    pred_label = pred_labels[pred_idx].item()
                    gt_label = gt_labels[gt_idx].item()

                    pred_cls_idx = class_to_idx[pred_label]
                    gt_cls_idx = class_to_idx[gt_label]

                    # Update confusion matrix
                    conf_matrix[gt_cls_idx, pred_cls_idx] += 1

                    # Update per-class metrics
                    if pred_label == gt_label:
                        metrics_per_class[gt_label]["TPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(iou.item())
                    else:
                        # Misclassification
                        metrics_per_class[gt_label]["FNs"] += 1
                        metrics_per_class[pred_label]["FPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(0)
                        metrics_per_class[pred_label]["IoUs"].append(0)

            # Unmatched predictions (False Positives)
            unmatched_pred_indices = set(range(n_preds)) - matched_pred_indices
            for pred_idx in unmatched_pred_indices:
                pred_label = pred_labels[pred_idx].item()
                pred_cls_idx = class_to_idx[pred_label]
                # Update confusion matrix: background row
                conf_matrix[n_classes, pred_cls_idx] += 1
                # Update per-class metrics
                metrics_per_class[pred_label]["FPs"] += 1
                metrics_per_class[pred_label]["IoUs"].append(0)

            # Unmatched ground truths (False Negatives)
            unmatched_gt_indices = set(range(n_gts)) - matched_gt_indices
            for gt_idx in unmatched_gt_indices:
                gt_label = gt_labels[gt_idx].item()
                gt_cls_idx = class_to_idx[gt_label]
                # Update confusion matrix: background column
                conf_matrix[gt_cls_idx, n_classes] += 1
                # Update per-class metrics
                metrics_per_class[gt_label]["FNs"] += 1
                metrics_per_class[gt_label]["IoUs"].append(0)

        return metrics_per_class, conf_matrix, class_to_idx

    def _compute_metrics_and_confusion_matrix_masks(self, preds):
        """
        Instance-level IoU via binary mask matching (greedy by IoU),
        aggregated per class across the dataset.
        """
        metrics_per_class = defaultdict(lambda: {"TPs": 0, "FPs": 0, "FNs": 0, "IoUs": []})

        # Collect all classes
        all_classes = set()
        for pred in preds:
            all_classes.update(pred["labels"].tolist())
        for gt in self.gt:
            all_classes.update(gt["labels"].tolist())
        all_classes = sorted(list(all_classes))
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(all_classes)}
        n_classes = len(all_classes)
        conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

        for pred, gt in zip(preds, self.gt):
            pred_labels = pred["labels"]
            gt_labels = gt["labels"]

            pm = self._get_pred_masks_bin_nhw(pred)  # [Np,H?,W?]
            gm = self._get_gt_masks_bin(gt)  # [Ng,H,W]

            Np = pm.shape[0]
            Ng = gm.shape[0]

            # Early-outs
            if Np == 0 and Ng == 0:
                continue

            # If spatial sizes differ, resize preds to GT resolution (nearest)
            if Np > 0 and Ng > 0 and pm.shape[-2:] != gm.shape[-2:]:
                pm = pm.unsqueeze(1).float()  # [Np,1,Hp,Wp]
                pm = torch.nn.functional.interpolate(pm, size=gm.shape[-2:], mode="nearest")
                pm = (pm > 0.5).to(torch.uint8)[:, 0]  # back to [Np,H,W]

            if Np > 0 and Ng > 0:
                ious = self._pairwise_mask_iou(pm, gm)  # [Np,Ng]
                ious_mask = ious >= self.iou_thresh
                pred_idx, gt_idx = torch.nonzero(ious_mask, as_tuple=True)

                if pred_idx.numel() > 0:
                    iou_vals = ious[pred_idx, gt_idx]
                    order = torch.argsort(-iou_vals)  # greedy by IoU
                    pred_idx = pred_idx[order]
                    gt_idx = gt_idx[order]
                    iou_vals = iou_vals[order]
                else:
                    iou_vals = torch.tensor([])

                matched_preds = set()
                matched_gts = set()

                for pi, gi, iv in zip(pred_idx, gt_idx, iou_vals):
                    pi = pi.item()
                    gi = gi.item()
                    if pi in matched_preds or gi in matched_gts:
                        continue
                    matched_preds.add(pi)
                    matched_gts.add(gi)

                    pl = pred_labels[pi].item()
                    gl = gt_labels[gi].item()

                    pci = class_to_idx[pl]
                    gci = class_to_idx[gl]
                    conf_matrix[gci, pci] += 1

                    if pl == gl:
                        metrics_per_class[gl]["TPs"] += 1
                        metrics_per_class[gl]["IoUs"].append(float(iv))
                    else:
                        # misclassification
                        metrics_per_class[gl]["FNs"] += 1
                        metrics_per_class[pl]["FPs"] += 1
                        metrics_per_class[gl]["IoUs"].append(0.0)
                        metrics_per_class[pl]["IoUs"].append(0.0)

                # Unmatched predictions -> FP
                for pi in set(range(Np)) - matched_preds:
                    pl = pred_labels[pi].item()
                    pci = class_to_idx[pl]
                    conf_matrix[n_classes, pci] += 1
                    metrics_per_class[pl]["FPs"] += 1
                    metrics_per_class[pl]["IoUs"].append(0.0)

                # Unmatched GTs -> FN
                for gi in set(range(Ng)) - matched_gts:
                    gl = gt_labels[gi].item()
                    gci = class_to_idx[gl]
                    conf_matrix[gci, n_classes] += 1
                    metrics_per_class[gl]["FNs"] += 1
                    metrics_per_class[gl]["IoUs"].append(0.0)

            elif Ng == 0:
                # only predictions => all FP
                for pi in range(Np):
                    pl = pred_labels[pi].item()
                    pci = class_to_idx[pl]
                    conf_matrix[n_classes, pci] += 1
                    metrics_per_class[pl]["FPs"] += 1
                    metrics_per_class[pl]["IoUs"].append(0.0)

            else:  # Np == 0 and Ng > 0
                # only GT => all FN
                for gi in range(Ng):
                    gl = gt_labels[gi].item()
                    gci = class_to_idx[gl]
                    conf_matrix[gci, n_classes] += 1
                    metrics_per_class[gl]["FNs"] += 1
                    metrics_per_class[gl]["IoUs"].append(0.0)

        return metrics_per_class, conf_matrix, class_to_idx

    def save_plots(self, path_to_save) -> None:
        path_to_save = Path(path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)

        if self.conf_matrix is not None:
            class_labels = [str(cls_id) for cls_id in self.class_to_idx.keys()] + ["background"]

            plt.figure(figsize=(10, 8))
            plt.imshow(self.conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(class_labels))
            plt.xticks(tick_marks, class_labels, rotation=45)
            plt.yticks(tick_marks, class_labels)

            # Add labels to each cell
            thresh = self.conf_matrix.max() / 2.0
            for i in range(self.conf_matrix.shape[0]):
                for j in range(self.conf_matrix.shape[1]):
                    plt.text(
                        j,
                        i,
                        format(self.conf_matrix[i, j], "d"),
                        horizontalalignment="center",
                        color="white" if self.conf_matrix[i, j] > thresh else "black",
                    )

            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.savefig(path_to_save / "confusion_matrix.png")
            plt.close()

        thresholds = self.thresholds
        precisions, recalls, f1_scores = [], [], []

        # Store the original predictions to reset after each threshold
        for threshold in thresholds:
            torchmetrics_preds = copy.deepcopy(self.torchmetrics_preds)
            # remove masks as they are already filtered and we wll get a shape mismatch
            if not torchmetrics_preds:
                return
            for torchmetrics_pred in torchmetrics_preds:
                if "masks" in torchmetrics_pred:
                    del torchmetrics_pred["masks"]

            # Filter predictions based on the current threshold
            filtered_preds = filter_preds(torchmetrics_preds, threshold, mask_source="masks")
            # Compute metrics with the filtered predictions
            metrics = self._compute_main_metrics(filtered_preds, ignore_masks=True)
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1_scores.append(metrics["f1"])

        # Plot Precision and Recall vs Threshold
        plt.figure()
        plt.plot(thresholds, precisions, label="Precision", marker="o")
        plt.plot(thresholds, recalls, label="Recall", marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("Value")
        plt.title("Precision and Recall vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.savefig(path_to_save / "precision_recall_vs_threshold.png")
        plt.close()

        # Plot F1 Score vs Threshold
        plt.figure()
        plt.plot(thresholds, f1_scores, label="F1 Score", marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.title("F1 Score vs Threshold")
        plt.grid(True)
        plt.savefig(path_to_save / "f1_score_vs_threshold.png")
        plt.close()

        # Find the best threshold based on F1 Score (last occurence)
        best_idx = len(f1_scores) - np.argmax(f1_scores[::-1]) - 1
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        logger.info(
            f"Best Threshold for object detection: {round(best_threshold, 2)} with F1 Score: {round(best_f1, 3)}"
        )


def to_uint8_bool(arr):
    return torch.tensor(arr, dtype=torch.uint8)


def make_box_from_mask(mask):
    """
    mask: torch.uint8 [H,W] with 0/1
    returns [x1,y1,x2,y2] in absolute pixels (xyxy)
    """
    ys, xs = torch.where(mask > 0)
    if ys.numel() == 0:
        return torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    y1, y2 = ys.min().item(), ys.max().item()
    x1, x2 = xs.min().item(), xs.max().item()
    # +1 on x2/y2 to make xyxy inclusive->exclusive box consistent with pixels
    return torch.tensor([x1, y1, x2 + 1, y2 + 1], dtype=torch.float32)


def pack_sample(masks_uint8, labels, scores=None):
    """
    masks_uint8: list of [H,W] uint8 tensors (0/1)
    labels: list[int]
    scores: list[float] or None (required for preds)
    Produces a dict compatible with your Validator expectations.
    """
    if len(masks_uint8) == 0:
        ms = torch.zeros((0, 1, 1), dtype=torch.uint8)  # your class handles this
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros((0,), dtype=torch.int64)
        out = {"boxes": boxes, "labels": labels_t, "masks": ms}
        if scores is not None:
            out["scores"] = torch.tensor(scores, dtype=torch.float32)
        return out

    ms = torch.stack(masks_uint8, dim=0)  # [N,H,W]
    boxes = torch.stack([make_box_from_mask(m) for m in masks_uint8], dim=0).to(torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.int64)
    out = {"boxes": boxes, "labels": labels_t, "masks": ms}
    if scores is not None:
        out["scores"] = torch.tensor(scores, dtype=torch.float32)
    return out


def assert_close(name, got, exp, tol=1e-6):
    if not (abs(got - exp) <= tol):
        raise AssertionError(f"{name}: expected {exp}, got {got}")


def run_single_case(gt_list, pred_list, iou_thr=0.5, msg=""):
    # Collect all unique labels from gt and preds
    all_labels = set()
    for g in gt_list:
        all_labels.update(g["labels"].tolist())
    for p in pred_list:
        all_labels.update(p["labels"].tolist())
    label_to_name = {lbl: f"class_{lbl}" for lbl in all_labels}

    val = Validator(
        gt_list, pred_list, label_to_name=label_to_name, conf_thresh=0.5, iou_thresh=iou_thr
    )
    m = val.compute_metrics(extended=False)
    print(f"\nCase: {msg}\nMetrics: {m}")
    return m


def main():
    # -------- Case 1: Perfect match (TP=1, FP=0, FN=0; IoU=1.0) --------
    gt1_mask = to_uint8_bool(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    pred1_mask = gt1_mask.clone()

    gt1 = [pack_sample([gt1_mask], labels=[0])]
    pred1 = [pack_sample([pred1_mask], labels=[0], scores=[1.0])]

    m1 = run_single_case(gt1, pred1, iou_thr=0.5, msg="Perfect match")
    assert_close("precision", m1["precision"], 1.0)
    assert_close("recall", m1["recall"], 1.0)
    assert_close("iou", m1["iou"], 1.0)

    # -------- Case 2: Partial match above threshold (IoU=0.75) --------
    # Use 4x4 full-ones GT vs pred missing one 1x4 strip -> intersection=12, union=16 -> IoU=0.75
    gt2_mask = to_uint8_bool(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    pred2_mask = to_uint8_bool(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],  # missing last row
        ]
    )
    gt2 = [pack_sample([gt2_mask], labels=[0])]
    pred2 = [pack_sample([pred2_mask], labels=[0], scores=[1.0])]

    m2 = run_single_case(gt2, pred2, iou_thr=0.5, msg="Partial > thr (IoU=0.75)")
    # One matched pair (same class), so TP=1, Precision=Recall=1 for this dataset of size 1
    assert_close("precision", m2["precision"], 1.0)
    assert_close("recall", m2["recall"], 1.0)
    assert_close("iou", m2["iou"], 0.75)

    # -------- Case 3: Misclassification (same mask, wrong class) --------
    # Same geometry, but predicted label != GT label -> should count as FP (pred class) and FN (gt class).
    gt3 = [pack_sample([gt1_mask], labels=[0])]
    pred3 = [pack_sample([gt1_mask], labels=[1], scores=[1.0])]

    m3 = run_single_case(gt3, pred3, iou_thr=0.5, msg="Misclassification")
    # With 1 GT and 1 Pred but wrong class: TP=0, FP=1, FN=1
    assert_close("precision", m3["precision"], 0.0)
    assert_close("recall", m3["recall"], 0.0)
    assert_close("iou", m3["iou"], 0.0)

    # -------- Case 4: Pure FP (prediction present, no GT) --------
    gt4 = [pack_sample([], labels=[])]
    pred4 = [pack_sample([gt1_mask], labels=[0], scores=[1.0])]

    m4 = run_single_case(gt4, pred4, iou_thr=0.5, msg="Pure FP")
    # TP=0, FP=1, FN=0
    assert_close("precision", m4["precision"], 0.0)
    assert_close("recall", m4["recall"], 0.0)
    assert_close("iou", m4["iou"], 0.0)

    # -------- Case 5: Different resolutions (pred 5x5 all-ones, GT 10x10 all-ones) --------
    # Your code upsamples preds to GT size with nearest neighbor; both are full ones -> IoU=1.
    gt5_mask = to_uint8_bool(np.ones((10, 10), dtype=np.uint8))
    pred5_mask_small = to_uint8_bool(np.ones((5, 5), dtype=np.uint8))
    # Pack pred with masks under "masks" at 5x5; your code will upsample internally.
    gt5 = [pack_sample([gt5_mask], labels=[2])]
    pred5 = [pack_sample([pred5_mask_small], labels=[2], scores=[1.0])]
    m5 = run_single_case(gt5, pred5, iou_thr=0.5, msg="Different resolutions upsample check")
    assert_close("precision", m5["precision"], 1.0)
    assert_close("recall", m5["recall"], 1.0)
    assert_close("iou", m5["iou"], 1.0)

    # -------- Case 6: Big GT mask and several smaller preds inside it --------
    gt6_mask = to_uint8_bool(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    pred6_mask1 = to_uint8_bool(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
        ]
    )

    pred6_mask2 = to_uint8_bool(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
        ]
    )

    m6 = run_single_case(
        [pack_sample([gt6_mask], labels=[0])],
        [pack_sample([pred6_mask1, pred6_mask2], labels=[0, 0], scores=[1.0, 1.0])],
        iou_thr=0.5,
        msg="One big GT, one smaller pred inside",
    )
    assert_close("iou", m6["iou"], 0.28125)
    assert_close("recall", m6["recall"], 1.0)
    assert_close("precision", m6["precision"], 0.5)

    print("\nAll synthetic instance-segmentation tests PASSED ✅")

    # -------- RLE Encoding Tests --------
    print("\n" + "=" * 50)
    print("Testing RLE Encoding/Decoding...")
    print("=" * 50)

    from src.dl.utils import encode_sample_masks_to_rle, masks_to_rle, rle_to_masks

    # Test 1: RLE encode/decode roundtrip
    test_masks = torch.stack([gt1_mask, gt2_mask], dim=0)  # [2, H, W]
    rles = masks_to_rle(test_masks)
    decoded = rle_to_masks(rles)
    assert torch.equal(test_masks, decoded), "RLE roundtrip failed!"
    print("✓ RLE encode/decode roundtrip works")

    # Test 2: Memory savings estimation
    from src.dl.utils import get_dense_mask_memory_size, get_rle_memory_size

    dense_size = get_dense_mask_memory_size(2, test_masks.shape[1], test_masks.shape[2])
    rle_size = get_rle_memory_size(rles)
    print(
        f"✓ Memory: dense={dense_size} bytes, RLE={rle_size} bytes, savings={100 * (1 - rle_size / dense_size):.1f}%"
    )

    # Test 3: Validator with RLE-encoded masks
    gt_rle = [pack_sample([gt1_mask], labels=[0])]
    pred_rle = [pack_sample([gt1_mask.clone()], labels=[0], scores=[1.0])]

    # Encode to RLE
    encode_sample_masks_to_rle(gt_rle[0])
    encode_sample_masks_to_rle(pred_rle[0])

    # Verify masks were replaced with RLE
    assert "masks_rle" in gt_rle[0] and "masks" not in gt_rle[0], "GT should have masks_rle"
    assert "masks_rle" in pred_rle[0] and "masks" not in pred_rle[0], "Pred should have masks_rle"

    # Run validator with RLE samples
    m_rle = run_single_case(gt_rle, pred_rle, iou_thr=0.5, msg="RLE-encoded perfect match")
    assert_close("precision", m_rle["precision"], 1.0)
    assert_close("recall", m_rle["recall"], 1.0)
    assert_close("iou", m_rle["iou"], 1.0)
    print("✓ Validator works with RLE-encoded masks")

    print("\nAll RLE tests PASSED ✅")


if __name__ == "__main__":
    main()
