from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from openvino import Core


class OV_model:
    def __init__(
        self,
        model_path: str,
        n_outputs: int,
        input_width: int = 640,
        input_height: int = 640,
        conf_thresh: float = 0.5,
        rect: bool = False,  # cuts paddings, inference is faster, accuracy might be lower
        half: bool = False,
        keep_ratio: bool = False,
        max_batch_size: int = 1,
        device: str = None,
    ):
        self.input_size = (input_height, input_width)
        self.n_outputs = n_outputs
        self.model_path = model_path
        self.device = device.upper() if device else device
        self.rect = rect
        self.half = half
        self.keep_ratio = keep_ratio
        self.channels = 3
        self.max_batch_size = max_batch_size
        self.torch_device = "cpu"

        if isinstance(conf_thresh, float):
            self.conf_threshs = [conf_thresh] * self.n_outputs
        elif isinstance(conf_thresh, list):
            self.conf_threshs = conf_thresh

        self.np_dtype = np.float32

        self._load_model()
        self._test_pred()

    def _load_model(self):
        core = Core()
        det_ov_model = core.read_model(self.model_path)

        if not self.device:
            self.device = "CPU"
            if "GPU" in core.get_available_devices() and not self.rect:
                self.device = "GPU"

        if self.device != "CPU":
            det_ov_model.reshape({"input": [1, 3, *self.input_size]})

        inference_hint = "f16" if self.half else "f32"
        inference_mode = "CUMULATIVE_THROUGHPUT" if self.max_batch_size > 1 else "LATENCY"
        self.model = core.compile_model(
            det_ov_model,
            self.device,
            config={"PERFORMANCE_HINT": inference_mode, "INFERENCE_PRECISION_HINT": inference_hint},
        )
        logger.info(f"OpenVino running on {self.device}")

    def _test_pred(self):
        random_image = np.random.randint(0, 255, size=(1000, 1110, self.channels), dtype=np.uint8)
        self.model(self._prepare_inputs(random_image)[0])

    @staticmethod
    def process_boxes(boxes, processed_sizes, orig_sizes, keep_ratio):
        """All operations stay on torch tensors."""
        final_boxes = torch.zeros_like(boxes)
        for idx in range(boxes.shape[0]):
            final_boxes[idx] = norm_xywh_to_abs_xyxy(
                boxes[idx], processed_sizes[idx][0], processed_sizes[idx][1]
            )

        for i in range(len(orig_sizes)):
            if keep_ratio:
                final_boxes[i] = scale_boxes_ratio_kept(
                    final_boxes[i], processed_sizes[i], orig_sizes[i]
                )
            else:
                final_boxes[i] = scale_boxes(final_boxes[i], orig_sizes[i], processed_sizes[i])
        return final_boxes

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
        return processed_inputs, processed_sizes, original_sizes

    def _predict(self, img: NDArray) -> List[NDArray]:
        outputs = list(self.model(img).values())
        return outputs

    def _postprocess(
        self,
        outputs: torch.Tensor,
        processed_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
        num_top_queries=300,
        use_focal_loss=True,
    ) -> List[Dict[str, NDArray]]:
        """
        returns List with BS length. Each element is a dict {"labels", "boxes", "scores"}
        """
        logits, boxes = torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])
        has_masks = False
        if len(outputs) == 3:
            has_masks = True
        pred_masks = torch.from_numpy(outputs[2]) if has_masks else None  # [B,Q,Hm,Wm]
        B, Q = logits.shape[:2]

        boxes = self.process_boxes(
            boxes, processed_sizes, original_sizes, self.keep_ratio
        )  # B x TopQ x 4

        # scores/labels and preliminary topK over all Q*C
        if use_focal_loss:
            scores_all = torch.sigmoid(logits)  # [B,Q,C]
            flat = scores_all.flatten(1)  # [B, Q*C]
            # pre-topk to avoid scanning all queries later
            K = min(num_top_queries, flat.shape[1])
            topk_scores, topk_idx = torch.topk(flat, K, dim=-1)  # [B,K]
            topk_labels = topk_idx - (topk_idx // self.n_outputs) * self.n_outputs  # [B,K]
            topk_qidx = topk_idx // self.n_outputs  # [B,K]
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
            sb = topk_scores[b]
            lb = topk_labels[b]
            qb = topk_qidx[b]
            # Apply per-class confidence thresholds
            conf_threshs_tensor = torch.tensor(self.conf_threshs, device=sb.device)
            keep = sb >= conf_threshs_tensor[lb]

            sb = sb[keep]
            lb = lb[keep]
            qb = qb[keep]
            # gather boxes once
            bb = boxes[b].gather(0, qb.unsqueeze(-1).repeat(1, 4))

            out = {"labels": lb, "boxes": bb, "scores": sb}

            if has_masks and qb.numel() > 0:
                # gather only kept masks
                mb = pred_masks[b, qb]  # [K', Hm, Wm] logits or probs
                # resize to original size (list of length 1)
                orig_sizes_tensor = torch.tensor([original_sizes[b]])
                masks_list = self.process_masks(
                    mb.unsqueeze(0),  # [1,K',Hm,Wm]
                    processed_size=processed_sizes[b],  # (Hin, Win)
                    orig_sizes=orig_sizes_tensor,  # [1,2]
                    keep_ratio=self.keep_ratio,
                )
                out["mask_probs"] = masks_list[0]  # [K, H, W]
                # clean up masks outside of the corresponding bbox
                out["mask_probs"] = cleanup_masks(out["mask_probs"], out["boxes"])

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
            masks: np.ndarray of shape (N, H, W), dtype np.uint8. N = number of objects
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


if __name__ == "__main__":
    import time

    model = OV_model(
        model_path="/home/argo/Desktop/Projects/Veryfi/crops/output/models/teee_2025-12-17/model.xml",
        n_outputs=1,
        input_height=640,
        input_width=640,
        conf_thresh=0.4,
    )

    img = cv2.imread("/home/argo/Desktop/Projects/Veryfi/sign_det/data/test/test_image.jpg")

    latency = []
    for _ in range(30):
        t0 = time.perf_counter()
        res = model(img)
        latency.append((time.perf_counter() - t0) * 1000)

    print(res)
    print("LATENCY:", np.mean(latency[1:]))
