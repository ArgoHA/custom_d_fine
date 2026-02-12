from pathlib import Path

import hydra
import onnx
import onnxsim
import openvino as ov
import tensorrt as trt
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from onnxconverter_common import float16
from torch import nn

from src.d_fine.configs import base_cfg
from src.d_fine.dfine import build_model
from src.dl.utils import get_latest_experiment_name


class DFINEPostProcessor(nn.Module):
    """Fused detection postprocessor baked into the exported graph.

    Performs: sigmoid -> topK -> cxcywh -> xyxy (in input-size pixels).
    Outputs: labels [B,K], boxes [B,K,4], scores [B,K].
    Masks (if present) are passed through with sigmoid applied.
    """

    def __init__(self, num_classes: int, num_top_queries: int = 300, use_focal_loss: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.use_focal_loss = use_focal_loss

    @staticmethod
    def norm_xywh_to_abs_xyxy(
        boxes: torch.Tensor, height: int, width: int, to_round=True
    ) -> torch.Tensor:
        """Converts boxes: [N, 4] normalized xywh -> [N, 4] absolute xyxy"""
        x_center = boxes[:, 0] * width
        y_center = boxes[:, 1] * height
        box_width = boxes[:, 2] * width
        box_height = boxes[:, 3] * height

        x_min = x_center - (box_width / 2)
        y_min = y_center - (box_height / 2)
        x_max = x_center + (box_width / 2)
        y_max = y_center + (box_height / 2)

        if to_round:
            x_min = torch.clamp(torch.floor(x_min), min=1)
            y_min = torch.clamp(torch.floor(y_min), min=1)
            x_max = torch.clamp(torch.ceil(x_max), max=width - 1)
            y_max = torch.clamp(torch.ceil(y_max), max=height - 1)
        else:
            x_min = torch.clamp(x_min, min=0)
            y_min = torch.clamp(y_min, min=0)
            x_max = torch.clamp(x_max, max=width)
            y_max = torch.clamp(y_max, max=height)
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    def forward(self, outputs: dict, input_h: int, input_w: int):
        logits = outputs["pred_logits"]  # [B, Q, C]
        boxes = outputs["pred_boxes"]  # [B, Q, 4]  normalised cxcywh
        pred_masks = outputs.get("pred_masks", None)  # [B, Q, Hm, Wm] or None

        # box conversion: normalised cxcywh -> absolute xyxy in input-size space
        abs_boxes = self.norm_xywh_to_abs_xyxy(boxes.flatten(0, 1), input_h, input_w).view(
            boxes.shape[0], boxes.shape[1], 4
        )

        # score extraction & topK
        if self.use_focal_loss:
            scores_all = torch.sigmoid(logits)  # [B, Q, C]
            flat = scores_all.flatten(1)  # [B, Q*C]
            K = min(self.num_top_queries, flat.shape[1])
            topk_scores, topk_idx = torch.topk(flat, K, dim=-1)  # [B, K]
            topk_labels = topk_idx % self.num_classes  # [B, K]
            topk_qidx = topk_idx // self.num_classes  # [B, K]
        else:
            probs = F.softmax(logits, dim=-1)[:, :, :-1]  # [B, Q, C-1]
            topk_scores, topk_labels = probs.max(dim=-1)  # [B, Q]
            K = min(self.num_top_queries, topk_scores.shape[1])
            topk_scores, order = torch.topk(topk_scores, K, dim=-1)
            topk_labels = topk_labels.gather(1, order)
            topk_qidx = order

        # gather boxes for top-K queries
        topk_boxes = abs_boxes.gather(1, topk_qidx.unsqueeze(-1).expand(-1, -1, 4))  # [B, K, 4]

        result = (topk_labels, topk_boxes, topk_scores)

        if pred_masks is not None:
            # Gather masks for top-K queries: [B, Q, Hm, Wm] -> [B, K, Hm, Wm]
            Hm, Wm = pred_masks.shape[2], pred_masks.shape[3]
            topk_masks = pred_masks.gather(
                1, topk_qidx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Hm, Wm)
            )
            result = result + (topk_masks,)

        return result


class ExportWrapper(nn.Module):
    """Wraps backbone model + postprocessor for ONNX/TRT export."""

    def __init__(self, model: nn.Module, postprocessor: DFINEPostProcessor, input_size):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
        self.input_h = input_size[0]
        self.input_w = input_size[1]

    def forward(self, x):
        outputs = self.model(x)
        return self.postprocessor(outputs, self.input_h, self.input_w)


def prepare_model(cfg, device):
    model = build_model(
        cfg.model_name,
        len(cfg.train.label_to_name),
        enable_mask_head=cfg.task == "segment",
        device=device,
        img_size=cfg.train.img_size,
    )
    model.load_state_dict(torch.load(Path(cfg.train.path_to_save) / "model.pt", weights_only=True))
    model.eval()
    return model


def export_to_onnx(
    model: nn.Module,
    model_path: Path,
    x_test: torch.Tensor,
    max_batch_size: int,
    half: bool,
    dynamic_input: bool,
    input_name: str,
    output_names: list[str],
) -> None:
    dynamic_axes = {}
    if max_batch_size > 1:
        for name in [input_name] + output_names:
            dynamic_axes[name] = {0: "batch_size"}
    if dynamic_input:
        if input_name not in dynamic_axes:
            dynamic_axes[input_name] = {}
        dynamic_axes[input_name].update({2: "height", 3: "width"})

    output_path = model_path.with_suffix(".onnx")
    torch.onnx.export(
        model,
        x_test,
        opset_version=19,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        dynamo=True,
    ).save(output_path)

    onnx_model = onnx.load(output_path)
    if half:
        onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)

    try:
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check
        logger.info("ONNX simplified and exported")
    except Exception as e:
        logger.info(f"Simplification failed: {e}")
    finally:
        onnx.save(onnx_model, output_path)
        return output_path


def export_to_openvino(onnx_path: Path, x_test, dynamic_input: bool, max_batch_size: int) -> None:
    if not dynamic_input and max_batch_size <= 1:
        inp = None
    elif max_batch_size > 1 and dynamic_input:
        inp = [-1, 3, -1, -1]
    elif max_batch_size > 1:
        inp = [-1, *x_test.shape[1:]]
    elif dynamic_input:
        inp = [1, 3, -1, -1]

    model = ov.convert_model(input_model=str(onnx_path), input=inp, example_input=x_test)

    ov.serialize(model, str(onnx_path.with_suffix(".xml")), str(onnx_path.with_suffix(".bin")))
    logger.info("OpenVINO model exported")


def export_to_tensorrt(
    onnx_file_path: Path,
    half: bool,
    max_batch_size: int,
) -> None:
    tr_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(tr_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, tr_logger)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    # Increase workspace memory to help with larger batch sizes
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    if max_batch_size > 1:
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        # Load ONNX model to get the actual input shape information
        onnx_model = onnx.load(str(onnx_file_path))

        # Find the input by name to ensure we get the correct one
        input_shape_proto = None
        for inp in onnx_model.graph.input:
            if inp.name == input_name:
                input_shape_proto = inp.type.tensor_type.shape
                break

        if input_shape_proto is None:
            raise ValueError(
                f"Could not find input '{input_name}' in ONNX model. "
                f"Available inputs: {[inp.name for inp in onnx_model.graph.input]}"
            )

        # Extract static dimensions from ONNX model
        # The first dimension (batch) should be dynamic, others should be static
        static_dims = []
        for i, dim in enumerate(input_shape_proto.dim[1:], start=1):  # Skip batch dimension
            if dim.dim_value:
                # Static dimension
                static_dims.append(int(dim.dim_value))
            elif dim.dim_param:
                # Dynamic dimension (not allowed for non-batch dims in this case)
                raise ValueError(
                    f"Cannot create TensorRT optimization profile: input shape has dynamic "
                    f"dimension at index {i} (beyond batch). Only batch dimension can be dynamic."
                )
            else:
                raise ValueError(
                    f"Cannot create TensorRT optimization profile: input shape dimension at "
                    f"index {i} is undefined."
                )

        # Set the minimum and optimal batch size to 1, and allow the maximum batch size as provided.
        min_shape = (1, *static_dims)
        opt_shape = (1, *static_dims)
        max_shape = (max_batch_size, *static_dims)

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(
            "Failed to build TensorRT engine. This can happen due to:\n"
            "1. Insufficient GPU memory\n"
            "2. Unsupported operations in the ONNX model\n"
            "3. Issues with dynamic batch size configuration\n"
            "Check the TensorRT logs above for more details."
        )

    with open(onnx_file_path.with_suffix(".engine"), "wb") as f:
        f.write(engine)
    logger.info("TensorRT model exported")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    input_name = "input"
    output_names = ["labels", "boxes", "scores"]
    enable_mask_head = cfg.task == "segment"
    if enable_mask_head:
        output_names.append("masks")

    device = cfg.train.device
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    model_path = Path(cfg.train.path_to_save) / "model.pt"

    raw_model = prepare_model(cfg, device)

    # Wrap model with fused postprocessor
    postprocessor = DFINEPostProcessor(
        num_classes=len(cfg.train.label_to_name),
        num_top_queries=base_cfg["DFINETransformer"]["num_queries"],
        use_focal_loss=base_cfg["matcher"]["use_focal_loss"],
    )
    model = ExportWrapper(raw_model, postprocessor, input_size=cfg.train.img_size)
    model.eval()
    raw_model.eval()

    x_test = torch.randn(cfg.export.max_batch_size, 3, *cfg.train.img_size).to(device)
    _ = model(x_test)

    # Openvino currently doesn't supprort some operations in postprocessor
    raw_output_names = ["logits", "boxes"]
    if enable_mask_head:
        raw_output_names.append("masks")
    raw_onnx_path = export_to_onnx(
        raw_model,
        model_path,
        x_test,
        cfg.export.max_batch_size,
        half=False,
        dynamic_input=False,
        input_name=input_name,
        output_names=raw_output_names,
    )
    export_to_openvino(raw_onnx_path, x_test, cfg.export.dynamic_input, max_batch_size=1)

    full_onnx_path = export_to_onnx(
        model,
        model_path,
        x_test,
        cfg.export.max_batch_size,
        half=False,
        dynamic_input=False,
        input_name=input_name,
        output_names=output_names,
    )
    export_to_tensorrt(full_onnx_path, cfg.export.half, cfg.export.max_batch_size)

    logger.info(f"Exports saved to: {model_path.parent}")


if __name__ == "__main__":
    main()
