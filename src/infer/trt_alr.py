from collections import OrderedDict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


class TRT_model:
    def __init__(
        self,
        model_path: str,
        n_outputs: int,
        input_width: int = 640,
        input_height: int = 640,
        conf_thresh: float = 0.5,
        rect: bool = False,  # No need for rectangular inference with fixed size
        half: bool = False,
        keep_ratio: bool = False,
        device: str = None,
        max_batch_size: int = 1,
    ) -> None:
        self.input_size = (input_height, input_width)
        self.n_outputs = n_outputs
        self.model_path = model_path
        self.rect = rect
        self.half = half
        self.keep_ratio = keep_ratio
        self.channels = 3
        self.max_batch_size = max_batch_size

        if isinstance(conf_thresh, float):
            self.conf_threshs = [conf_thresh] * self.n_outputs
        elif isinstance(conf_thresh, list):
            self.conf_threshs = conf_thresh

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.np_dtype = np.float32

        self._load_model()
        self._test_pred()

    def _load_model(self):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(self.model_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream

        #  discover tensors
        self.input_name = None
        self.output_info = OrderedDict()  # name → (dtype, shape)

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                assert self.input_name is None, "multiple inputs not handled"
                self.input_name = name

            elif mode == trt.TensorIOMode.OUTPUT:
                o_dtype = self._torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
                self.output_info[name] = (o_dtype, tuple(self.engine.get_tensor_shape(name)))

        assert self.input_name and self.output_info, "failed to find I/O tensors"

        # pre-allocate output buffers
        self.out_bufs = {}
        for name, (dtype, shape) in self.output_info.items():
            # replace dynamic batch dim (-1) with max_batch (usually 1)
            shape = tuple((self.max_batch_size if s < 0 else s) for s in shape)
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self.out_bufs[name] = buf
            self.context.set_tensor_address(name, int(buf.data_ptr()))

        # static part of the input shape
        # batch dim will change each call, H&W are fixed (640×640 here)
        static_shape = (1, self.channels, self.input_size[0], self.input_size[1])
        self.context.set_input_shape(self.input_name, static_shape)

    @staticmethod
    def _torch_dtype_from_trt(trt_dtype):
        if trt_dtype == trt.float32:
            return torch.float32
        elif trt_dtype == trt.float16:
            return torch.float16
        elif trt_dtype == trt.int32:
            return torch.int32
        elif trt_dtype == trt.int8:
            return torch.int8
        else:
            raise TypeError(f"Unsupported TensorRT data type: {trt_dtype}")

    def _test_pred(self) -> None:
        random_image = np.random.randint(0, 255, size=(1100, 1000, self.channels), dtype=np.uint8)
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(random_image)
        preds = self._predict(processed_inputs)
        self._postprocess(preds, processed_sizes, original_sizes)

    @staticmethod
    def process_boxes(boxes, processed_sizes, orig_sizes, keep_ratio, device):
        boxes = boxes.cpu().numpy()
        final_boxes = np.zeros_like(boxes)
        for idx, box in enumerate(boxes):
            final_boxes[idx] = norm_xywh_to_abs_xyxy(
                box, processed_sizes[idx][0], processed_sizes[idx][1]
            )

        for i in range(len(orig_sizes)):
            if keep_ratio:
                final_boxes[i] = scale_boxes_ratio_kept(
                    final_boxes[i],
                    processed_sizes[i],
                    orig_sizes[i],
                )
            else:
                final_boxes[i] = scale_boxes(
                    final_boxes[i],
                    orig_sizes[i],
                    processed_sizes[i],
                )
        return torch.tensor(final_boxes).to(device)

    def _preds_postprocess(
        self,
        outputs,
        processed_sizes,
        original_sizes,
        num_top_queries=300,
        use_focal_loss=True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        logits, boxes = outputs[0], outputs[1]
        boxes = self.process_boxes(
            boxes, processed_sizes, original_sizes, self.keep_ratio, self.device
        )  # B x TopQ x 4

        if use_focal_loss:
            scores = torch.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
            labels = index - index // self.n_outputs * self.n_outputs
            index = index // self.n_outputs
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
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        return results

    def _compute_nearest_size(self, shape, target_size, stride=32) -> Tuple[int, int]:
        """
        Get nearest size that is divisible by 32
        """
        scale = target_size / max(shape)
        new_shape = [int(round(dim * scale)) for dim in shape]

        # Make sure new dimensions are divisible by the stride
        new_shape = [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]
        return new_shape

    def _preprocess(self, img: NDArray, stride: int = 32) -> torch.tensor:
        if not self.keep_ratio:  # simple resize
            img = cv2.resize(
                img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
            )
        elif self.rect:  # keep ratio and cut paddings
            target_height, target_width = self._compute_nearest_size(
                img.shape[:2], max(*self.input_size)
            )
            img = letterbox(img, (target_height, target_width), stride=stride, auto=False)[0]
        else:  # keep ratio adding paddings
            img = letterbox(
                img, (self.input_size[0], self.input_size[1]), stride=stride, auto=False
            )[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0
        return img

    def _prepare_inputs(self, inputs):
        original_sizes = []
        processed_sizes = []

        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:  # single image
            processed_inputs = self._preprocess(inputs)[None]
            original_sizes.append((inputs.shape[0], inputs.shape[1]))
            processed_sizes.append((processed_inputs[0].shape[1], processed_inputs[0].shape[2]))

        elif isinstance(inputs, np.ndarray) and inputs.ndim == 4:  # batch of images
            processed_inputs = np.zeros(
                (inputs.shape[0], self.channels, self.input_size[0], self.input_size[1]),
                dtype=self.np_dtype,
            )
            for idx, image in enumerate(inputs):
                processed_inputs[idx] = self._preprocess(image)
                original_sizes.append((image.shape[0], image.shape[1]))
                processed_sizes.append(
                    (processed_inputs[idx].shape[1], processed_inputs[idx].shape[2])
                )
        return torch.tensor(processed_inputs).to(self.device), processed_sizes, original_sizes

    def _predict(self, img: torch.Tensor):
        """
        img shape: (B, C, H, W) already on CUDA
        returns a list of torch.Tensors [logits, boxes] sliced to batch B
        """
        B = img.shape[0]

        # 1) give TRT the input address and actual batch shape
        self.context.set_input_shape(self.input_name, img.shape)  # dynamic B
        self.context.set_tensor_address(self.input_name, int(img.data_ptr()))

        # 2) run
        self.context.execute_async_v3(self.stream)

        # 3) slice only the first B rows from our pre-allocated buffers
        return [buf[:B] for buf in self.out_bufs.values()]

    def _postprocess(
        self,
        preds: torch.tensor,
        processed_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
    ):
        output = self._preds_postprocess(preds, processed_sizes, original_sizes)
        output = filter_preds(output, self.conf_threshs)

        for res in output:
            res["labels"] = res["labels"].cpu().numpy()
            res["boxes"] = res["boxes"].cpu().numpy()
            res["scores"] = res["scores"].cpu().numpy()
        return output

    def __call__(self, inputs: NDArray[np.uint8]) -> List[Dict[str, np.ndarray]]:
        """
        Input image as ndarray (BGR, HWC) or BHWC
        Output:
            List of batch size length. Each element is a dict {"labels", "boxes", "scores"}
            labels: np.ndarray of shape (N,), dtype np.int64
            boxes: np.ndarray of shape (N, 4), dtype np.float32, abs values
            scores: np.ndarray of shape (N,), dtype np.float32
        """
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(inputs)
        preds = self._predict(processed_inputs)
        return self._postprocess(preds, processed_sizes, original_sizes)


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes_ratio_kept(boxes, img1_shape, img0_shape, ratio_pad=None, padding=True):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_boxes(boxes, orig_shape, resized_shape):
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes


def norm_xywh_to_abs_xyxy(boxes: np.ndarray, h: int, w: int, clip=True):
    """
    Normalised (x_c, y_c, w, h) -> absolute (x1, y1, x2, y2)
    Keeps full floating-point precision; no rounding.
    """
    x_c, y_c, bw, bh = boxes.T
    x_min = x_c * w - bw * w / 2
    y_min = y_c * h - bh * h / 2
    x_max = x_c * w + bw * w / 2
    y_max = y_c * h + bh * h / 2

    if clip:
        x_min = np.clip(x_min, 0, w - 1)
        y_min = np.clip(y_min, 0, h - 1)
        x_max = np.clip(x_max, 0, w - 1)
        y_max = np.clip(y_max, 0, h - 1)

    return np.stack([x_min, y_min, x_max, y_max], axis=1)


def filter_preds(preds, conf_threshs: List[float]):
    conf_threshs = torch.tensor(conf_threshs, device=preds[0]["scores"].device)
    for pred in preds:
        mask = pred["scores"] >= conf_threshs[pred["labels"]]
        pred["scores"] = pred["scores"][mask]
        pred["boxes"] = pred["boxes"][mask]
        pred["labels"] = pred["labels"][mask]
    return preds
