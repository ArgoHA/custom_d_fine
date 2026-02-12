import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch
from numpy.typing import NDArray


class TRT_model:
    def __init__(
        self,
        model_path: str,
        n_outputs: int,
        conf_thresh: float | List[float] = 0.5,
        binarize_masks: bool = True,
        mask_threshold: float = 0.5,
        rect: bool = False,
        half: bool = False,
        keep_ratio: bool = False,
        device: str = None,
    ) -> None:
        self.model_path = model_path
        self.n_outputs = n_outputs
        self.rect = rect
        self.half = half
        self.keep_ratio = keep_ratio
        self.channels = 3
        self.binarize_masks = binarize_masks
        self.mask_threshold = mask_threshold
        self.np_dtype = np.float32

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._load_model()
        self._read_engine_metadata()

        # Per-class confidence thresholds
        if isinstance(conf_thresh, float):
            self.conf_threshs = [conf_thresh] * self.n_outputs
        elif isinstance(conf_thresh, list):
            self.conf_threshs = conf_thresh

        self._test_pred()

    def _load_model(self):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def _read_engine_metadata(self):
        """Auto-read input_size and detect mask presence from the engine."""
        inp_name = self.engine.get_tensor_name(0)
        inp_shape = tuple(self.engine.get_tensor_shape(inp_name))
        self.input_size = (inp_shape[2], inp_shape[3])  # (H, W)

        n_outputs = 0
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                n_outputs += 1
        self.has_masks = n_outputs > 3

    @staticmethod
    def _torch_dtype_from_trt(trt_dtype):
        if trt_dtype == trt.float32:
            return torch.float32
        elif trt_dtype == trt.float16:
            return torch.float16
        elif trt_dtype == trt.int32:
            return torch.int32
        elif trt_dtype == trt.int64:
            return torch.int64
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
    def rescale_boxes(boxes, processed_sizes, orig_sizes, keep_ratio):
        """Rescale absolute xyxy boxes from input-size space to original image size."""
        out = boxes.clone()
        for i in range(boxes.shape[0]):
            if keep_ratio:
                out[i] = scale_boxes_ratio_kept(out[i], processed_sizes[i], orig_sizes[i])
            else:
                out[i] = scale_boxes(out[i], orig_sizes[i], processed_sizes[i])
        return out

    @staticmethod
    def process_masks(
        pred_masks,  # Tensor [B, Q, Hm, Wm] or [Q, Hm, Wm]
        processed_size,  # (H, W) of network input (after your A.Compose)
        orig_sizes,  # Tensor [B, 2] (H, W)
        keep_ratio: bool,
    ) -> List[torch.Tensor]:
        """
        Returns list of length B with masks resized to original image sizes:
        Each item: Float Tensor [Q, H_orig, W_orig] in [0,1] (no thresholding here).
        - Handles letterbox padding removal if keep_ratio=True.
        - Works for both batched and single-image inputs.
        """
        single = pred_masks.dim() == 3  # [Q,Hm,Wm]
        if single:
            pred_masks = pred_masks.unsqueeze(0)  # -> [1,Q,Hm,Wm]

        B, Q, Hm, Wm = pred_masks.shape
        proc_h, proc_w = int(processed_size[0]), int(processed_size[1])

        out = []
        for b in range(B):
            H0, W0 = int(orig_sizes[b, 0].item()), int(orig_sizes[b, 1].item())
            m = pred_masks[b]  # [Q, Hm, Wm]

            if keep_ratio:
                # Compute same gain/pad as in scale_boxes_ratio_kept
                gain = min(proc_h / H0, proc_w / W0)
                padw = round((proc_w - W0 * gain) / 2 - 0.1)
                padh = round((proc_h - H0 * gain) / 2 - 0.1)

                # Calculate crop region in mask space (scaled from processed_size to mask_size)
                scale_h, scale_w = Hm / proc_h, Wm / proc_w
                y1 = int(max(padh, 0) * scale_h)
                y2 = int((proc_h - max(padh, 0)) * scale_h)
                x1 = int(max(padw, 0) * scale_w)
                x2 = int((proc_w - max(padw, 0)) * scale_w)
                m = m[:, y1:y2, x1:x2]  # [Q, cropped_h, cropped_w]

            # Single resize directly to original size
            m = torch.nn.functional.interpolate(
                m.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False
            ).squeeze(0)  # [Q, H0, W0]
            out.append(m.clamp_(0, 1))

        if single:
            return [out[0]]
        return out

    def _compute_nearest_size(self, shape, target_size, stride=32) -> Tuple[int, int]:
        """Get nearest size that is divisible by 32"""
        scale = target_size / max(shape)
        new_shape = [int(round(dim * scale)) for dim in shape]
        return [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]

    def _preprocess(self, img: NDArray, stride: int = 32) -> NDArray:
        """
        Resize, RGB, CHW
        """
        if not self.keep_ratio:  # simple resize
            img = cv2.resize(
                img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR
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

        # BGR to RGB, HWC to CHW
        img = img[..., ::-1].transpose(2, 0, 1)
        return np.ascontiguousarray(img, dtype=np.uint8)

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
                dtype=np.uint8,
            )
            for idx, image in enumerate(inputs):
                processed_inputs[idx] = self._preprocess(image)
                original_sizes.append((image.shape[0], image.shape[1]))
                processed_sizes.append(
                    (processed_inputs[idx].shape[1], processed_inputs[idx].shape[2])
                )

        # Normalize to [0,1] on GPU
        if self.device == "cuda":
            tensor = torch.from_numpy(processed_inputs).to(self.device, non_blocking=True)
            tensor = tensor.to(dtype=torch.float32).div_(255.0)
        else:
            tensor = torch.from_numpy(processed_inputs).to(dtype=torch.float32).div_(255.0)
        return tensor, processed_sizes, original_sizes

    def _predict(self, img: torch.Tensor) -> List[torch.Tensor]:
        # 1) make contiguous and grab the full (B, C, H, W) shape
        img = img.contiguous()
        batch_shape = tuple(img.shape)

        # 2) prepare our buffer-pointer list
        n_io = self.engine.num_io_tensors
        bindings: List[int] = [None] * n_io
        outputs: List[torch.Tensor] = []

        # 3) for each I/O slot, either bind the input or allocate an output
        for i in range(n_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dims = tuple(self.engine.get_tensor_shape(name))
            dt = self.engine.get_tensor_dtype(name)
            t_dt = self._torch_dtype_from_trt(dt)

            if mode == trt.TensorIOMode.INPUT:
                # set our actual batch‐shape on the context
                ok = self.context.set_input_shape(name, batch_shape)
                assert ok, f"Failed to set input shape for {name} -> {batch_shape}"
                # point that binding at our tensor’s data ptr
                bindings[i] = img.data_ptr()
            else:
                # allocate a matching output tensor (B, *dims[1:])
                out_shape = (batch_shape[0],) + dims[1:]
                out = torch.empty(out_shape, dtype=t_dt, device=self.device)
                outputs.append(out)
                bindings[i] = out.data_ptr()

        # 4) run inference
        self.context.execute_v2(bindings)
        return outputs

    def _postprocess(
        self,
        outputs: List[torch.Tensor],
        processed_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, NDArray]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        labels = outputs[0]  # [B, K]
        boxes = outputs[1]  # [B, K, 4], absolute xyxy in input_size space
        scores = outputs[2]  # [B, K]
        pred_masks = outputs[3] if self.has_masks else None  # [B, K, Hm, Wm]
        B = labels.shape[0]

        boxes = self.rescale_boxes(boxes, processed_sizes, original_sizes, self.keep_ratio)

        results = []
        for b in range(B):
            sb, lb, bb = scores[b], labels[b], boxes[b]
            # Apply per-class confidence thresholds
            if self.conf_threshs is not None:
                conf_t = torch.tensor(self.conf_threshs, device=sb.device)
                keep = sb >= conf_t[lb]
            else:
                keep = sb >= self.conf_thresh
            sb, lb, bb = sb[keep], lb[keep], bb[keep]

            out = {"labels": lb, "boxes": bb, "scores": sb}

            if pred_masks is not None and lb.numel() > 0:
                mb = pred_masks[b][keep]  # [K, Hm, Wm] — already gathered for top-K
                # resize to original size (list of length 1)
                orig_sizes_tensor = torch.tensor([original_sizes[b]], device=mb.device)
                masks_list = self.process_masks(
                    mb.unsqueeze(0),
                    processed_size=processed_sizes[b],  # (Hin, Win)
                    orig_sizes=orig_sizes_tensor,  # [1,2]
                    keep_ratio=self.keep_ratio,
                )
                out["masks"] = masks_list[0]  # [K, H, W]
                if self.binarize_masks:
                    out["masks"] = (out["masks"] >= self.mask_threshold).to(torch.uint8)
                # clean up masks outside of the corresponding bbox
                out["masks"] = cleanup_masks(out["masks"], out["boxes"])

            results.append(out)
        return results

    def __call__(self, inputs: NDArray[np.uint8]) -> List[Dict[str, torch.Tensor]]:
        """
        Input image as ndarray (BGR, HWC) or BHWC
        Output:
            List of batch size length. Each element is a dict {"labels", "boxes", "scores"}
            labels: torch.Tensor of shape (N,), dtype int64
            boxes: torch.Tensor of shape (N, 4), dtype float32, abs values
            scores: torch.Tensor of shape (N,), dtype float32
            masks: torch.Tensor of shape (N, H, W), dtype float32. N = number of objects
        """
        processed_inputs, processed_sizes, original_sizes = self._prepare_inputs(inputs)
        preds = self._predict(processed_inputs)
        return self._postprocess(preds, processed_sizes, original_sizes)

    @staticmethod
    def mask2poly(masks: np.ndarray, img_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Convert binary masks to normalized polygon coordinates for YOLO segmentation format.

        Args:
            masks: Binary masks array of shape [N, H, W] where N is number of instances
            img_shape: Tuple of (height, width) of the original image

        Returns:
            List of normalized polygon coordinates, each as array of shape [num_points, 2]
            with values in range [0, 1]. Returns empty array for invalid masks.
        """
        h, w = img_shape[:2]
        polys = []

        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour
                contour = max(contours, key=cv2.contourArea)
                contour = contour.reshape(-1, 2)
                if len(contour) >= 3:  # Need at least 3 points for a valid polygon
                    # Normalize coordinates
                    norm_contour = contour.astype(np.float32)
                    norm_contour[:, 0] /= w
                    norm_contour[:, 1] /= h
                    polys.append(norm_contour)
                else:
                    polys.append(np.array([]))
            else:
                polys.append(np.array([]))

        return polys


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
    ratio = r, r  # initial uniform width, height ratios (may be updated below)
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
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(np.floor(dh)), int(np.ceil(dh))
    left, right = int(np.floor(dw)), int(np.ceil(dw))
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


def cleanup_masks(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    # clean up masks outside of the corresponding bbox
    N, H, W = masks.shape
    device = masks.device
    dtype = masks.dtype

    ys = torch.arange(H, device=device)[None, :, None]  # (1, H, 1)
    xs = torch.arange(W, device=device)[None, None, :]  # (1, 1, W)

    x1, y1, x2, y2 = boxes.T  # each (N,)
    inside = (
        (xs >= x1[:, None, None])
        & (xs < x2[:, None, None])
        & (ys >= y1[:, None, None])
        & (ys < y2[:, None, None])
    )  # (N, H, W), bool
    masks = masks * inside.to(dtype)
    return masks
