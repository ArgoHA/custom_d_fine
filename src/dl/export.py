from pathlib import Path

import hydra
import onnx
import onnxsim
import openvino as ov
import tensorrt as trt
import torch
from loguru import logger
from omegaconf import DictConfig
from onnxconverter_common import float16
from torch import nn

from src.d_fine.dfine import build_model
from src.dl.utils import get_latest_experiment_name

INPUT_NAME = "input"
OUTPUT_NAMES = ["logits", "boxes"]


def prepare_model(cfg, device):
    model = build_model(
        cfg.model_name, len(cfg.train.label_to_name), device, img_size=cfg.train.img_size
    )
    model.load_state_dict(torch.load(Path(cfg.train.path_to_save) / "model.pt", weights_only=True))
    model.eval()
    return model


def add_suffix(output_path, dynamic_input: bool, half: bool):
    if dynamic_input:
        output_path = f"{output_path}_dynamic"
    if half:
        output_path = f"{output_path}_half"
    return Path(output_path).with_suffix(".onnx")


def export_to_onnx(
    model: nn.Module,
    model_path: Path,
    x_test: torch.Tensor,
    max_batch_size: int,
    half: bool,
    dynamic_input: bool,
) -> None:
    dynamic_axes = {}
    if max_batch_size > 1:
        dynamic_axes = {
            INPUT_NAME: {0: "batch_size"},
            OUTPUT_NAMES[0]: {0: "batch_size"},
            OUTPUT_NAMES[1]: {0: "batch_size"},
        }
    if dynamic_input:
        if INPUT_NAME not in dynamic_axes:
            dynamic_axes[INPUT_NAME] = {}
        dynamic_axes[INPUT_NAME].update({2: "height", 3: "width"})

    output_path = add_suffix(
        model_path.parent / model_path.stem, dynamic_input=dynamic_input, half=half
    )
    torch.onnx.export(
        model,
        x_test,
        opset_version=19,
        input_names=[INPUT_NAME],
        output_names=OUTPUT_NAMES,
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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    if max_batch_size > 1:
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        input_shape = network.get_input(0).shape  # e.g., [batch, channels, height, width]

        # Set the minimum and optimal batch size to 1, and allow the maximum batch size as provided.
        min_shape = (1, *input_shape[1:])
        opt_shape = (1, *input_shape[1:])
        max_shape = (max_batch_size, *input_shape[1:])

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    with open(onnx_file_path.with_suffix(".engine"), "wb") as f:
        f.write(engine)
    logger.info("TensorRT model exported")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    device = cfg.train.device
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    model_path = Path(cfg.train.path_to_save) / "model.pt"

    model = prepare_model(cfg, device)
    x_test = torch.randn(cfg.export.max_batch_size, 3, *cfg.train.img_size).to(device)
    _ = model(x_test)

    # onnx version
    onnx_path = export_to_onnx(
        model,
        model_path,
        x_test,
        cfg.export.max_batch_size,
        half=cfg.export.half,
        dynamic_input=cfg.export.dynamic_input,
    )
    export_to_openvino(onnx_path, x_test, cfg.export.dynamic_input, max_batch_size=1)

    # static onnx for tensorrt
    if cfg.export.half or cfg.export.dynamic_input:
        onnx_path = export_to_onnx(
            model,
            model_path,
            x_test,
            cfg.export.max_batch_size,
            False,
            False,
        )
    export_to_tensorrt(
        onnx_path,
        cfg.export.half,
        cfg.export.max_batch_size,
    )

    logger.info(f"Exports saved to: {model_path.parent}")


if __name__ == "__main__":
    main()
