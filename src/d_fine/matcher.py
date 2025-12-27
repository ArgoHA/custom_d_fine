"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .arch.utils import box_cxcywh_to_xyxy, generalized_box_iou


def dice_cost(pred_masks: torch.Tensor, gt_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute pairwise Dice cost between predicted and GT masks.

    Args:
        pred_masks: [num_queries, H, W] sigmoid probabilities
        gt_masks: [num_targets, H, W] binary masks

    Returns:
        cost: [num_queries, num_targets] Dice cost (1 - Dice)
    """
    pred_masks = pred_masks.flatten(1).float()  # [Q, H*W] - ensure float32
    gt_masks = gt_masks.flatten(1).float()  # [T, H*W] - ensure float32

    # Compute pairwise intersection and union
    # pred: [Q, HW], gt: [T, HW] -> need [Q, T]
    numerator = 2 * torch.einsum("qp,tp->qt", pred_masks, gt_masks)  # [Q, T]
    denominator = pred_masks.sum(dim=1, keepdim=True) + gt_masks.sum(dim=1, keepdim=False)  # [Q, T]

    dice = (numerator + eps) / (denominator + eps)  # [Q, T]
    return 1 - dice


def sigmoid_focal_cost(
    pred_logits: torch.Tensor, gt_labels: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """
    Compute sigmoid focal cost for masks (pixel-wise).

    Args:
        pred_logits: [num_queries, H*W] mask logits
        gt_labels: [num_targets, H*W] binary ground truth

    Returns:
        cost: [num_queries, num_targets]
    """
    # Ensure float32 for numerical stability and dtype consistency
    pred_logits = pred_logits.float()
    gt_labels = gt_labels.float()

    pred_prob = pred_logits.sigmoid()

    # Focal weights
    neg_cost = (1 - alpha) * (pred_prob**gamma) * (-(1 - pred_prob + 1e-8).log())
    pos_cost = alpha * ((1 - pred_prob) ** gamma) * (-(pred_prob + 1e-8).log())

    # Cost per pair: mean over pixels
    # pos_cost: [Q, HW], gt: [T, HW]
    cost = torch.einsum("qp,tp->qt", pos_cost, gt_labels) + torch.einsum(
        "qp,tp->qt", neg_cost, (1 - gt_labels)
    )

    return cost / pred_logits.shape[1]  # normalize by number of pixels


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = [
        "use_focal_loss",
    ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            cost_mask: (optional) weight for mask dice cost in matching
        """
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.cost_mask = weight_dict.get("cost_mask", 0)  # Optional mask cost
        self.cost_mask_dice = weight_dict.get("cost_mask_dice", 0)  # Optional dice cost

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, (
            "all costs cant be 0"
        )

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix: [bs*num_queries, total_targets]
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        # Reshape to [bs, num_queries, total_targets] before adding mask cost
        C = C.view(bs, num_queries, -1)

        # Add mask cost if masks are available and cost weight > 0
        if (self.cost_mask > 0 or self.cost_mask_dice > 0) and "pred_masks" in outputs:
            pred_masks = outputs["pred_masks"]  # [B, Q, Hm, Wm]
            if pred_masks is not None:
                # Check if any target has masks
                has_any_masks = any(
                    "masks" in t and t["masks"] is not None and t["masks"].numel() > 0
                    for t in targets
                )
                if has_any_masks:
                    B_mask, Q_mask, Hm, Wm = pred_masks.shape
                    sizes_local = [len(v["boxes"]) for v in targets]

                    # Only apply mask cost if Q matches (handles denoising queries mismatch)
                    if Q_mask != num_queries:
                        # pred_masks includes denoising queries, need to extract only the matching queries
                        # The regular queries are after the denoising queries
                        dn_num = Q_mask - num_queries
                        if dn_num > 0:
                            # Skip denoising queries - take only the regular queries
                            pred_masks = pred_masks[:, dn_num:, :, :]

                    # Process per batch since targets have varying counts
                    offset = 0
                    for b in range(bs):
                        n_tgt = sizes_local[b]
                        if n_tgt == 0:
                            continue

                        t = targets[b]
                        if "masks" not in t or t["masks"] is None or t["masks"].numel() == 0:
                            offset += n_tgt
                            continue

                        tgt_m = t["masks"].float().to(pred_masks.device)  # [Nb, H, W]
                        if tgt_m.shape[-2:] != (Hm, Wm):
                            tgt_m = F.interpolate(
                                tgt_m.unsqueeze(1), size=(Hm, Wm), mode="nearest"
                            ).squeeze(1)

                        pred_m = pred_masks[b].sigmoid()  # [Q, Hm, Wm]

                        # Dice cost: [Q, Nb]
                        cost_mc = torch.zeros(num_queries, n_tgt, device=pred_m.device)
                        if self.cost_mask_dice > 0:
                            cost_mc = cost_mc + self.cost_mask_dice * dice_cost(pred_m, tgt_m)

                        # Focal cost: [Q, Nb]
                        if self.cost_mask > 0:
                            cost_mc = cost_mc + self.cost_mask * sigmoid_focal_cost(
                                pred_masks[b].flatten(1),  # [Q, Hm*Wm] logits
                                tgt_m.flatten(1),  # [Nb, Hm*Wm]
                                alpha=self.alpha,
                                gamma=self.gamma,
                            )

                        C[b, :, offset : offset + n_tgt] = (
                            C[b, :, offset : offset + n_tgt] + cost_mc
                        )
                        offset += n_tgt

        C = C.cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = (
                [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                if i > 0
                else initial_indices
            )
            indices_list.append(
                [
                    (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices_k
                ]
            )
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]
        # C.copy_(C_original)
        return indices_list
