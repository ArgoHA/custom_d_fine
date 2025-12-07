from copy import deepcopy
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from src.d_fine.dfine_criterion import DFINECriterion

from .arch.dfine_decoder import DFINETransformer
from .arch.hgnetv2 import HGNetv2
from .arch.hybrid_encoder import HybridEncoder
from .configs import models
from .matcher import HungarianMatcher
from .utils import load_tuning_state

__all__ = ["DFINE"]


class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


def build_model(
    model_name, num_classes, enable_mask_head, device, img_size=None, pretrained_model_path=None
):
    model_cfg = deepcopy(models[model_name])

    model_cfg["HybridEncoder"]["eval_spatial_size"] = img_size
    model_cfg["DFINETransformer"]["eval_spatial_size"] = img_size
    model_cfg["DFINETransformer"]["enable_mask_head"] = enable_mask_head

    backbone = HGNetv2(**model_cfg["HGNetv2"])
    encoder = HybridEncoder(**model_cfg["HybridEncoder"])
    decoder = DFINETransformer(num_classes=num_classes, **model_cfg["DFINETransformer"])

    model = DFINE(backbone, encoder, decoder)

    if pretrained_model_path:
        if not Path(pretrained_model_path).exists():
            raise FileNotFoundError(f"{pretrained_model_path} does not exist")
        model = load_tuning_state(model, str(pretrained_model_path))
    return model.to(device)


def build_loss(model_name, num_classes, label_smoothing, enable_mask_head):
    model_cfg = models[model_name]
    if enable_mask_head:
        model_cfg["DFINECriterion"]["losses"].append("masks")
    matcher = HungarianMatcher(**model_cfg["matcher"])
    loss_fn = DFINECriterion(
        matcher,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        **model_cfg["DFINECriterion"],
    )
    return loss_fn


def build_optimizer(model, lr, backbone_lr, betas, weight_decay, base_lr):
    backbone_exclude_norm = []
    backbone_norm = []
    encdec_norm_bias = []
    rest = []

    for name, param in model.named_parameters():
        # Group 1 and 2: "backbone" in name
        if "backbone" in name:
            if "norm" in name or "bn" in name:
                # Group 2: backbone + norm/bn
                backbone_norm.append(param)
            else:
                # Group 1: backbone but not norm/bn
                backbone_exclude_norm.append(param)

        # Group 3: "encoder" or "decoder" plus "norm"/"bn"/"bias"
        elif ("encoder" in name or "decoder" in name) and (
            "norm" in name or "bn" in name or "bias" in name
        ):
            encdec_norm_bias.append(param)

        else:
            rest.append(param)

    group1 = {"params": backbone_exclude_norm, "lr": backbone_lr, "initial_lr": backbone_lr}
    group2 = {
        "params": backbone_norm,
        "lr": backbone_lr,
        "weight_decay": 0.0,
        "initial_lr": backbone_lr,
    }
    group3 = {"params": encdec_norm_bias, "weight_decay": 0.0, "lr": base_lr, "initial_lr": base_lr}
    group4 = {"params": rest, "lr": base_lr, "initial_lr": base_lr}

    param_groups = [group1, group2, group3, group4]

    return optim.AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)
